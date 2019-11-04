import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
from src.data.loader import load_dataset
from src.data.labeled_data import LabeledData
from src.rnn import RNN
from src.tools import print_config, set_momentum
from src.timer import Timer
from src.global_variable import set_rnn_config, set_info_config, set_train_config
from tensorflow.python import debug as tf_debug
import time


class Experiment:
    def __init__(self):
        self.rnn = None
        self.l_data = None
        self.l_data_config = None
        self.data_dict = None
        self.rnn_config = None
        self.info_config = None
        self.train_config = None
        self.timer = None

    def create_rnn(self, l_data, l_data_config):
        set_momentum(self.train_config['batchnorm']['momentum'])
        self.rnn = RNN(l_data)
        self.l_data = l_data
        self.l_data_config = l_data_config

    # Creates a RNN using a modified l_data_config
    # Used e.g. by incremental sequence training, where the l_data_config is changed while training
    def create_modificated_model(self, l_data_config, session_idx):
        l_data_config['tr']['in_seq_len'] = self.train_config['mode']['in_seq_len'][session_idx]
        l_data_config['tr']['max_truncation'] = self.train_config['mode']['max_truncation'][session_idx]
        self.data_dict = load_dataset(l_data_config)
        labeled_data = LabeledData(l_data_config, self.data_dict)
        self.create_rnn(labeled_data, l_data_config)

    def train(self, rnn_config, l_data_config, train_config, info_config, run):
        print('what')
        self.rnn_config = rnn_config
        self.info_config = info_config
        self.train_config = train_config
        set_rnn_config(rnn_config)
        set_info_config(info_config)

        self.timer = Timer(info_config['timer']['enabled'])
        print_config(rnn_config, train_config, l_data_config)
        temp_model_path = '../models/temp' + str(train_config['task_id'])
        pretrained_model_path = '../tr_models/' + str(train_config['pretraining']['path'])

        if train_config['mode']['name'] == 'inc_lengths':
            n_sessions = len(train_config['mode']['in_seq_len'])
        elif train_config['mode']['name'] == 'classic':
            n_sessions = 1
        else:
            raise Exception('training mode not understood')

        self.timer.start()
        set_train_config(train_config)
        # Sessions refer to training with different architectures. If one RNN is used throughout the training process
        # then only one session is created. Training with incremental sequence lengths for example requires multiple
        # RNNs, one for each sequence lenghts. Evaluation datasets (validation and test) are always evaluated on a fixed
        # RNN, only the RNN structure used for the training set varies. current_epoch stores the total amounts of epochs
        # and epoch the epoch within a session
        current_epoch = 0
        tau = self.train_config['tau']
        learning_rate = self.train_config['learning_rate']
        best_weight_probs_dict = None
        for session_idx in range(n_sessions):
            tf.reset_default_graph()
            if self.train_config['mode']['name'] == 'inc_lengths':
                max_epochs = self.train_config['mode']['max_epochs'][session_idx]
                min_error = self.train_config['mode']['min_errors'][session_idx]
                self.create_modificated_model(l_data_config, session_idx)
            elif self.train_config['mode']['name'] == 'classic':
                self.data_dict = load_dataset(l_data_config)
                l_data = LabeledData(l_data_config, self.data_dict)
                self.create_rnn(l_data, l_data_config)
                max_epochs = self.train_config['mode']['max_epochs']
                min_error = self.train_config['mode']['min_error']
            self.timer.restart('Graph creation')

            # Saver is used for restoring weights for new session if more than one is used for training
            model_saver = tf.train.Saver(var_list=tf.trainable_variables())
            with tf.Session() as sess:
                if info_config['profiling']['enabled']:
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                else:
                    options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
                run_metadata = tf.RunMetadata()
                writer = tf.summary.FileWriter(info_config['tensorboard']['path'] + str(self.train_config['task_id']))
                sess.run(tf.global_variables_initializer())

                if session_idx != 0:
                    #self.optimistic_restore(sess, pretrained_model_path)
                    model_saver.restore(sess, temp_model_path)
                elif self.train_config['pretraining']['enabled'] == True:
                    self.optimistic_restore(sess, pretrained_model_path)
                    sess.run(self.rnn.init_op)
                #sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline")
                #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
                self.timer.restart('Intialization')

                # Loading datasets into GPU (via tf.Variables)
                for key in self.data_dict.keys():
                    sess.run(self.l_data.data[key]['load'],
                             feed_dict={self.l_data.data[key]['x_ph']: self.data_dict[key]['x'],
                                        self.l_data.data[key]['y_ph']: self.data_dict[key]['y']})
                    sess.run(self.l_data.data[key]['shuffle'])

                self.timer.restart('Loading data')

                traces = list()

                for epoch in range(max_epochs):
                    self.save_gradient_variance(sess, epoch, tau)
                    #quit()
                    # Evaluate performance on the different datasets and print some results on console
                    # Also check potential stopping critera
                    if current_epoch % info_config['calc_performance_every'] == 0:
                        self.rnn.t_metrics.retrieve_results(sess, current_epoch, tau)
                        self.rnn.t_metrics.print(session_idx)
                        #if self.rnn.t_metrics.result_dict['tr_b']['vfe'][-1] < min_error:
                            #break

                    if current_epoch + 1 % info_config['save_weights']['save_every'] == 0:
                        self.save_weight_probs(info_config['save_weights']['path'], current_epoch, run,
                                               sess.run(self.rnn.get_weights_op))

                    if info_config['save_weights']['save_best']:
                        if self.rnn.t_metrics.best_va['is_current']:
                            best_weight_probs_dict = sess.run(self.rnn.get_weights_op)

                    self.timer.restart('Metrics')

                    # Optionally store tensorboard summaries
                    if info_config['tensorboard']['enabled'] \
                            and current_epoch % info_config['tensorboard']['period'] == 0:
                        if info_config['tensorboard']['weights']:
                            weight_summary = sess.run(self.rnn.weight_summaries,
                                                      feed_dict={self.rnn.tau: (tau,), self.l_data.batch_idx: 0, self.rnn.is_training: False})
                            writer.add_summary(weight_summary, current_epoch)
                        if info_config['tensorboard']['gradients']:
                            gradient_summary = sess.run(self.rnn.gradient_summaries,
                                                        feed_dict={self.rnn.tau: (tau,), self.l_data.batch_idx: 0, self.rnn.is_training: False})
                            writer.add_summary(gradient_summary, current_epoch)
                        if info_config['tensorboard']['results']:
                            t_result_summaries = sess.run(self.rnn.t_metric_summaries,
                                                          feed_dict={self.rnn.tau: (tau,), self.l_data.batch_idx: 0, self.rnn.is_training: False})
                            writer.add_summary(t_result_summaries, current_epoch)
                        if info_config['tensorboard']['acts']:
                            act_summaries = sess.run(self.rnn.act_summaries, feed_dict={self.rnn.tau: (tau,), self.l_data.batch_idx: 0, self.rnn.is_training: False})
                            writer.add_summary(act_summaries, current_epoch)

                    self.timer.restart('Tensorboard')
                    # Train for one full epoch. First shuffle to create new minibatches from the given data and
                    # then do a training step for each minibatch.
                    # Also anneal learning rate and tau if necessary
                    if (current_epoch + 1) % self.train_config['learning_rate_tau'] == 0:
                        learning_rate /= 2

                    sess.run(self.l_data.data['tr']['shuffle'])
                    if 'c_ar' in self.train_config['algorithm'] or 'c_arm' in self.train_config['algorithm']:
                        sess.run(self.rnn.assign_learning_rate, feed_dict={self.rnn.learning_rate: learning_rate})
                    for minibatch_idx in range(self.l_data.data['tr']['n_minibatches']):
                        if 'c_ar' in self.train_config['algorithm'] or 'c_arm' in self.train_config['algorithm']\
                                or 'log_der' in self.train_config['algorithm']:
                            grads = None
                            for i in range(self.train_config['carm_iterations']):
                                sess.run(self.rnn.c_arm_sample_op)
                                gradients = sess.run(self.rnn.gradients, feed_dict={self.l_data.batch_idx: minibatch_idx, self.rnn.is_training:True})
                                if grads is None:
                                    grads = gradients
                                else:
                                    for j in range(len(grads)):
                                        if grads[j] is not None:
                                            grads[j] += gradients[j]
                            for j in range(len(grads)):
                                grads[j] /= self.train_config['carm_iterations']
                            sess.run(self.rnn.train_b_op, feed_dict={gradient_ph: grad for gradient_ph, grad in zip(self.rnn.gradient_ph, grads)})

                        else:
                            sess.run(self.rnn.train_b_op,
                                     feed_dict={self.rnn.learning_rate: learning_rate, self.rnn.tau: (tau,),
                                                self.l_data.batch_idx: minibatch_idx, self.rnn.is_training: True},
                                     options=options, run_metadata=run_metadata)

                    if info_config['profiling']['enabled']:
                        traces.append(timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format())
                    current_epoch += 1
                    self.timer.restart('Training')

                # Optionally store profiling results of this epoch in files
                if info_config['profiling']['enabled']:
                    for trace_idx, trace in enumerate(traces):
                        path = info_config['profiling']['path'] + '_' + str(current_epoch) + '_' + str(trace_idx)
                        with open(path + 'training.json', 'w') as f:
                            f.write(trace)

                # TODO: Clean the cell access code
                if info_config['cell_access']:
                    ca_1, ca_2 = sess.run([self.rnn.layers[0].cell_access_mat, self.rnn.layers[1].cell_access_mat],
                             feed_dict={self.l_data.batch_idx: 0})
                    np.save(file='../nr/ca_1_'+ str(self.train_config['task_id']), arr=ca_1)
                    np.save(file='../nr/ca_2_'+ str(self.train_config['task_id']), arr=ca_2)
                model_saver.save(sess, temp_model_path)

        if info_config['save_weights']['save_best']:
            self.save_weight_probs(self.info_config['save_weights']['path'], 'best', run, best_weight_probs_dict)
        writer.close()
        return self.rnn.t_metrics.result_dict

    # Empirically estimates variance of gradient, saves results and quits
    def save_gradient_variance(self, sess, epoch, tau):
        n_gradients = 23
        tf_grads = []
        tf_vars = []
        for tuple in self.rnn.gradients:
            if tuple is not None:
                tf_grads.append(tuple[0])
                tf_vars.append(tuple[1])

        gradients = {}
        for idx in range(len(tf_grads)):
            gradients[idx] = []

        for gradient_idx in range(n_gradients):
            gradient = sess.run(tf_grads, feed_dict={self.l_data.batch_idx: 0, self.rnn.tau: (tau,)})
            for idx, grad in enumerate(gradient):
                gradients[idx].append(np.expand_dims(grad, axis=0))

        variances = []
        expectations = []
        squared_expectations = []
        for idx in range(len(gradients)):
            gradients[idx] = np.concatenate(gradients[idx], axis=0)
            variances.append(np.var(gradients[idx], axis=0, ddof=1))
            expectations.append(np.mean(gradients[idx], axis=0))
            squared_expectations.append(np.mean(np.square(gradients[idx]), axis=0))
            var = tf_vars[idx]
            suffix = '_' + var.name[:var.name.index('/')] + '_' +  var.name[var.name.index('/')+1:-2] + '_' + str(self.train_config['task_id']) + '.npy'
            np.save(file='../numerical_results/gvar' + suffix, arr=variances[-1])
            np.save(file='../numerical_results/ge' + suffix, arr=expectations[-1])
            np.save(file='../numerical_results/gsqe' + suffix, arr=squared_expectations[-1])
        quit()
        #variances = np.concatenate(variances, axis=0)
        #normed_vars = np.concatenate(normed_vars, axis=0)
        np.save(file='../numerical_results/var_' + str(self.train_config['task_id']), arr=variances)
        #np.save(file='../numerical_results/normed_var_' + str(self.train_config['task_id']), arr=normed_vars)


    def optimistic_restore(self, sess, file):
        reader = tf.train.NewCheckpointReader(file)
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
        opt_saver.restore(sess, file)

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



