import tensorflow as tf
import numpy as np
from copy import deepcopy

from src.data.loader import load_gpu_datasets
from src.data.labeled_data import GPUDatasets
from src.configuration.constants import DatasetKeys, AlgorithmC

from src.network_gpu.rnn import RNN
from src.network_gpu.timer import Timer
from src.network_gpu.tools import print_config
from src.network_gpu.profiler import Profiler

from src.global_variable import set_nn_config, set_train_config, set_info_config
from src.t_metrics import save_to_file, print_results

# Mon: 50 min

class Experiment:
    def __init__(self, nn_config, data_config, info_config, train_config, task_id):
        self.rnn = None
        self.nn_config = nn_config
        self.train_config = train_config
        self.info_config = info_config
        self.datasets = load_gpu_datasets(data_config)
        self.gpu_dataset = GPUDatasets(data_config, self.datasets)
        self.timer = None
        self.result_dicts = []
        self.task_id = task_id

        set_nn_config(nn_config)
        set_info_config(info_config)
        set_train_config(train_config)

        with tf.variable_scope("this"):
            self.is_training = tf.placeholder(tf.int32)
            self.op = self.gpu_dataset.data["tr"][DatasetKeys.Y]
            self.op2 = self.gpu_dataset.data["tr"][DatasetKeys.X]

        self.create_nn(self.gpu_dataset, data_config)

    def create_nn(self, gpu_dataset, data_config):
        self.rnn = RNN(gpu_dataset)
        self.gpu_dataset = gpu_dataset
        self.data_config = data_config

    def train_multiple_runs(self, epochs, runs):
        for run in range(runs):
            self.train(epochs)
        self.save_results()

    def train(self, max_epoch):
        self.rnn.t_metrics.reset_results()
        self.timer = Timer(self.info_config.timer_enabled)
        print_config(rnn_config=self.nn_config,
                     training_config=self.train_config,
                     data_config=self.data_config,
                     info_config=self.info_config)

        with tf.Session() as sess:
            profiler = Profiler(info_config=self.info_config)
            writer = tf.summary.FileWriter(self.info_config.tensorboard_config.path + str(self.train_config.task_id))

            sess.run(tf.global_variables_initializer())

            if self.info_config.model_loader_config is not None:
                self.load_model_from_file(sess, self.info_config.model_loader_config.create_path())

            self.timer.start()

            # Load datasets from numpy into GPU
            for data_key in self.gpu_dataset.data.keys():
                sess.run(self.gpu_dataset.data[data_key][DatasetKeys.OP_LOAD],
                         feed_dict={self.gpu_dataset.data[data_key][DatasetKeys.X_PLACEHOLDER]:
                                        self.datasets[data_key][DatasetKeys.X],
                                    self.gpu_dataset.data[data_key][DatasetKeys.Y_PLACEHOLDER]:
                                        self.datasets[data_key][DatasetKeys.Y]})

            self.timer.restart("Loading data into GPU")


            current_epoch = 0
            gumbel_tau = self.train_config.gumbel_tau
            learning_rate = self.train_config.learning_rate

            if self.info_config.save_gradients:
                self.save_gradient_stats(sess, gumbel_tau)

            model_saver = tf.train.Saver(var_list=tf.trainable_variables())
            for epoch in range(max_epoch):
                # Shuffle data
                sess.run(self.gpu_dataset.data[DatasetKeys.TR_SET][DatasetKeys.OP_SHUFFLE])

                # Compute training metrics
                self.compute_performance(current_epoch=epoch, sess=sess, gumbel_tau=gumbel_tau)

                # Anneal learning rate if required
                if (epoch + 1) % self.train_config.learning_rate_anneal_period == 0:
                    learning_rate /= 2

                # AR and ARM require a learning rate variable which is initialized
                if self.train_config.algorithm in [AlgorithmC.AR, AlgorithmC.ARM]:
                    sess.run(self.rnn.assign_learning_rate, feed_dict={self.rnn.learning_rate: learning_rate})

                # Now iterate over minibatches to perform one epoch of learning
                for minibatch_idx in range(self.gpu_dataset.data[DatasetKeys.TR_SET][DatasetKeys.N_MINIBATCHES]):
                    # AR, ARM and log derivative use a different approach to training
                    # Here, the weights are samples n_forward_passes times and the
                    # gradients are computed using the respective algorithmic method
                    if self.train_config.algorithm in [AlgorithmC.AR, AlgorithmC.ARM, AlgorithmC.LOG_DERIVATIVE]:
                        accumulated_gradients = []

                        # For each forward pass we do the following:
                        # 1) sample new weights
                        # 2) compute gradients
                        # 3) add gradients to the gradients of the other forward passes
                        grad_test = []
                        for forward_pass_index in range(self.train_config.n_forward_passes):
                            sess.run(self.rnn.c_arm_sample_op)
                            gradients = sess.run(self.rnn.gradients, feed_dict={self.gpu_dataset.batch_idx: minibatch_idx,
                                                                                self.rnn.is_training: True})

                            if len(accumulated_gradients) == 0:
                                for gradient_idx in range(len(gradients)):
                                    accumulated_gradients.append(gradients[gradient_idx])
                            else:
                                for gradient_idx in range(len(gradients)):
                                    if gradient_idx == 0:
                                        grad_test.append(gradients[gradient_idx])
                                    accumulated_gradients[gradient_idx] += gradients[gradient_idx]

                        # Compute mean over gradients and update the weights with those gradients
                        if True:
                            print("M: " + str(np.mean(np.abs(np.mean(np.stack(grad_test, axis=-1), axis=-1)))))
                            print("S: " + str(np.mean(np.std(np.stack(grad_test, axis=-1), axis=-1))))
                        for gradient_idx in range(len(accumulated_gradients)):
                            accumulated_gradients[gradient_idx] /= self.train_config.n_forward_passes
                        sess.run(self.rnn.train_b_op, feed_dict={gradient_ph: grad for gradient_ph, grad in zip(self.rnn.gradient_ph, accumulated_gradients)})

                    # Every other training algorithm uses backpropagation to compute gradients
                    else:
                        sess.run(self.rnn.train_b_op,
                                 feed_dict={self.rnn.learning_rate: learning_rate, self.rnn.tau: gumbel_tau,
                                            self.gpu_dataset.batch_idx: minibatch_idx, self.rnn.is_training: True},
                                 options=profiler.options, run_metadata=profiler.run_metadata)

                self.timer.restart('Training epoch')

                # At the end of each epoch we update the profiler and we save the model
                profiler.add_trace()
                if self.info_config.model_saver_config is not None:
                    model_saver.save(sess, self.info_config.model_saver_config.create_path())

                writer.close()

            profiler.conclude_training(max_epoch)

        self.result_dicts.append(deepcopy(self.rnn.t_metrics.result_dict))
        return self.rnn.t_metrics.result_dict

    def run(self, number):
        with tf.Session() as sess:
            for key in self.datasets.keys():
                sess.run(self.gpu_dataset.data[key][DatasetKeys.OP_LOAD],
                         feed_dict={self.gpu_dataset.data[key][DatasetKeys.X_PLACEHOLDER]:
                                    self.datasets[key][DatasetKeys.X],
                                    self.gpu_dataset.data[key][DatasetKeys.Y_PLACEHOLDER]:
                                    self.datasets[key][DatasetKeys.Y]})

            sess.run(self.gpu_dataset.data["tr"][DatasetKeys.OP_SHUFFLE])
            for batch_idx in range(self.gpu_dataset.data["tr"][DatasetKeys.N_MINIBATCHES]):
                y = sess.run(self.op, feed_dict={self.gpu_dataset.batch_idx: batch_idx})

    def compute_performance(self, current_epoch, sess, gumbel_tau):
        if current_epoch % self.info_config.compute_tmetrics_every == 0:
            self.rnn.t_metrics.retrieve_results(sess, current_epoch, gumbel_tau)
            self.rnn.t_metrics.print(session_idx=0)

    # Loads the model from a file, that is, all weights of the model are initialized
    def load_model_from_file(self, sess, path):
        reader = tf.train.NewCheckpointReader(path)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
        if var.name.split(':')[0] in saved_shapes and 'batch_normalization' not in var.name])
        restore_vars = []
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = tf.get_variable(saved_var_name)
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
        opt_saver = tf.train.Saver(restore_vars)
        opt_saver.restore(sess, path)
        print("loaded")

    # This saves the weight probabilities in numpy format
    def save_weight_probs(self, path, epoch, run, weight_probs_dict):
        for layer_key in weight_probs_dict.keys():
            for var_key in weight_probs_dict[layer_key].keys():
                layer_weights = weight_probs_dict[layer_key]
                if len(layer_weights[var_key].keys()) == 2:
                    # Continuous weight with mean and variance
                    np.save(path + '_r' + str(run) + '_e' + str(epoch) + '_' + layer_key + '_' + var_key + '_m.npy',
                            layer_weights[var_key]['m'])
                    np.save(path + '_r' + str(run) + '_e' + str(epoch) + '_' + layer_key + '_' + var_key + '_v.npy',
                            layer_weights[var_key]['v'])
                else:
                    np.save(path + '_r' + str(run) + '_e' + str(epoch) + '_' + layer_key + '_' + var_key + '_p.npy',
                            layer_weights[var_key]['probs'])

    def save_results(self):
        if self.info_config.save_training_metrics:
            save_to_file(self.result_dicts, self.info_config.training_metrics_path + self.info_config.filename + "_" + str(self.task_id))
        print_results(self.result_dicts)

    def save_gradient_stats(self, sess, gumbel_tau):
        n_gradients = self.info_config.save_n_gradients

        debug1 = self.rnn.layers[-1].weights.tensor_dict["w"]
        debug2 = self.rnn.layers[-1].weights.logder_derivs["w"]
        debug_sb = self.rnn.layers[-1].weights.var_dict["w_sb"]

        vars = []
        rnn_gradients = []
        for gradient, var in zip(self.rnn.gradients, self.rnn.vars):
            if gradient is not None:
                rnn_gradients.append(gradient)
                vars.append(var)

        gradient_1st_mom = []
        gradient_2nd_mom = []
        samples = np.zeros((3,))
        for gradient_idx in range(n_gradients):
            sess.run(self.rnn.c_arm_sample_op)
            gradient = sess.run(rnn_gradients, feed_dict={self.gpu_dataset.batch_idx: 0,
                                                     self.rnn.is_training: True,
                                                     self.rnn.tau: gumbel_tau})
            if gradient_idx % 50 == 0:
                print("Processed: " + str(gradient_idx))

            if len(gradient_1st_mom) == 0:
                for idx, val in enumerate(gradient):
                    gradient_1st_mom.append(val)
                    gradient_2nd_mom.append(np.square(val))
            else:
                for idx, val in enumerate(gradient):
                    gradient_1st_mom[idx] += val
                    gradient_2nd_mom[idx] += np.square(val)

            if gradient_idx % 500 == 0:
                print(f"processed {gradient_idx}")

        for idx in range(len(gradient_1st_mom)):
            gradient_1st_mom[idx] /= n_gradients
            gradient_2nd_mom[idx] /= n_gradients
            var = vars[idx]

            suffix = '_' + var.name[:var.name.index('/')] + '_' + var.name[var.name.index('/')+1:-2] + '_' + str(self.task_id) + '.npy'
            np.save(file='../nr_grads/ge' + suffix, arr=gradient_1st_mom[idx])
            np.save(file='../nr_grads/gsqe' + suffix, arr=gradient_2nd_mom[idx])
        exit()


