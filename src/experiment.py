import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
from src.data.loader import load_dataset
from src.data.labeled_data import LabeledData
from src.rnn import RNN
from src.tools import print_config
from src.timer import Timer
import time


class Experiment:
    def __init__(self):
        self.rnn = None
        self.l_data = None
        self.l_data_config = None
        self.rnn_config = None
        self.data_dict = None
        self.info_config = None
        self.timer = None

    def create_rnn(self, train_config, l_data, l_data_config):
        self.rnn = RNN(self.rnn_config, train_config, self.info_config, l_data)

        self.l_data = l_data
        self.l_data_config = l_data_config

    # Creates a RNN using a modified l_data_config
    # Used e.g. by incremental sequence training, where the l_data_config is changed while training
    def create_modificated_model(self, train_config, l_data_config, mod_data_config):
        incremental_idx = mod_data_config['session_idx']
        l_data_config['tr']['in_seq_len'] = mod_data_config['in_seq_len'][incremental_idx]
        l_data_config['tr']['out_seq_len'] = mod_data_config['out_seq_len'][incremental_idx]
        l_data_config['tr']['zero_padding'] = mod_data_config['zero_padding'][incremental_idx]
        self.data_dict = load_dataset(l_data_config)
        labeled_data = LabeledData(l_data_config, self.data_dict)
        self.create_rnn(train_config, labeled_data, l_data_config)

    def train(self, rnn_config, l_data_config, train_config, pretrain_config, info_config):
        self.rnn_config = rnn_config
        self.info_config = info_config
        self.timer = Timer(info_config['timer']['enabled'])
        print_config(rnn_config, train_config, l_data_config)
        temp_model_path = '../models/temp' + str(train_config['task_id'])
        pretrained_model_path = '../models/' + pretrain_config['path']

        if train_config['mode']['name'] == 'inc_lengths':
            n_sessions = len(train_config['mode']['in_seq_len'])
        elif train_config['mode']['name'] == 'classic':
            n_sessions = 1
        else:
            raise Exception('training mode not understood')

        self.timer.start()
        if pretrain_config['no_pretraining'] is False and pretrain_config['just_load'] is False:
            self.pretrain(l_data_config, pretrain_config, pretrained_model_path)
            print('pretraning is over')
        self.timer.restart('Pretraining')
        # Sessions refer to training with different architectures. If one RNN is used throughout the training process
        # then only one session is created. Training with incremental sequence lengths for example requires multiple
        # RNNs, one for each sequence lenghts. Evaluation datasets (validation and test) are always evaluated on a fixed
        # RNN, only the RNN structure used for the training set varies. current_epoch stores the total amounts of epochs
        # and epoch the epoch within a session
        current_epoch = 0
        for session_idx in range(n_sessions):
            tf.reset_default_graph()
            if train_config['mode']['name'] == 'inc_lengths':
                train_config['mode']['session_idx'] = session_idx
                max_epochs = train_config['mode']['max_epochs'][session_idx]
                min_error = train_config['mode']['min_errors'][session_idx]
                self.create_modificated_model(train_config, l_data_config, train_config['mode'])
            elif train_config['mode']['name'] == 'classic':
                self.data_dict = load_dataset(l_data_config)
                l_data = LabeledData(l_data_config, self.data_dict)
                self.create_rnn(train_config, l_data, l_data_config)
                max_epochs = train_config['mode']['max_epochs']
                min_error = train_config['mode']['min_error']

            self.timer.restart('Graph creation')

            # Saver is used for restoring weights for new session if more than one is used for training
            model_saver = tf.train.Saver(tf.trainable_variables())
            with tf.Session() as sess:
                if info_config['profiling']['enabled']:
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                else:
                    options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
                run_metadata = tf.RunMetadata()
                writer = tf.summary.FileWriter(info_config['tensorboard']['path'] + str(train_config['task_id']))
                sess.run(tf.global_variables_initializer())

                if session_idx != 0:
                    model_saver.restore(sess, temp_model_path)
                elif pretrain_config['no_pretraining'] is False:
                    model_saver.restore(sess, pretrained_model_path)
                    sess.run(self.rnn.init_op)

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
                    #self.save_gradient_variance(sess, train_config, epoch)
                    # Evaluate performance on the different datasets and print some results on console
                    # Also check potential stopping critera
                    if current_epoch % info_config['calc_performance_every'] == 0:
                        self.rnn.t_metrics.retrieve_results(sess, current_epoch)
                        self.rnn.t_metrics.print(session_idx)
                        if self.rnn.t_metrics.result_dict['tr_b']['nelbo'][-1] < min_error:
                            break
                    self.timer.restart('Metrics')

                    # Optionally store tensorboard summaries
                    if info_config['tensorboard']['enabled'] \
                            and current_epoch % info_config['tensorboard']['period'] == 0:
                        if info_config['tensorboard']['weights']:
                            weight_summary = sess.run(self.rnn.weight_summaries,
                                                      feed_dict={self.l_data.batch_idx: 0})
                            writer.add_summary(weight_summary, current_epoch)
                        if info_config['tensorboard']['gradients']:
                            gradient_summary = sess.run(self.rnn.gradient_summaries,
                                                        feed_dict={self.l_data.batch_idx: 0})
                            writer.add_summary(gradient_summary, current_epoch)
                        if info_config['tensorboard']['results']:
                            t_result_summaries = sess.run(self.rnn.t_metric_summaries,
                                                          feed_dict={self.l_data.batch_idx: 0})
                            writer.add_summary(t_result_summaries, current_epoch)
                        if info_config['tensorboard']['acts']:
                            act_summaries = sess.run(self.rnn.act_summaries, feed_dict={self.l_data.batch_idx: 0})
                            writer.add_summary(act_summaries, current_epoch)

                    self.timer.restart('Tensorboard')
                    # Train for one full epoch. First shuffle to create new minibatches from the given data and
                    # then do a training step for each minibatch.
                    sess.run(self.l_data.data['tr']['shuffle'])
                    for minibatch_idx in range(self.l_data.data['tr']['n_minibatches']):
                        sess.run(self.rnn.train_b_op,
                                 feed_dict={self.rnn.learning_rate: train_config['learning_rate'],
                                            self.l_data.batch_idx: minibatch_idx},
                                 options=options, run_metadata=run_metadata)
                    traces.append(timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format())
                    current_epoch += 1
                    self.timer.restart('Training')

                # Optionally store profiling results of this epoch in files
                if info_config['profiling']['enabled']:
                    for trace_idx, trace in enumerate(traces):
                        path = info_config['profiling']['path'] + '_' + str(current_epoch) + '_' + str(trace_idx)
                        with open(path + 'training.json', 'w') as f:
                            f.write(trace)

                model_saver.save(sess, temp_model_path)
        writer.close()
        return self.rnn.t_metrics.result_dict

    # Empirically estimates variance of gradient, saves results and quits
    def save_gradient_variance(self, sess, train_config, epoch):
        n_gradients = 20
        tf_grads = []

        for grad, var in self.rnn.gradients:
            if grad is not None:
                tf_grads.append(grad)

        gradients = {}
        for idx in range(len(tf_grads)):
            gradients[idx] = []

        for gradient_idx in range(n_gradients):
            gradient = sess.run(tf_grads, feed_dict={self.l_data.batch_idx: 0})
            for idx, grad in enumerate(gradient):
                gradients[idx].append(np.expand_dims(grad, axis=0))

        for grad_key in gradients.keys():
            grad_distribution = np.concatenate(gradients[grad_key], axis=0)
            variance = np.var(grad_distribution, axis=0, ddof=1)

            np.save(file='../numerical_results/g_var_' + str(train_config['task_id']) + '_' + str(grad_key) + '_' +
                         str(epoch), arr=variance)

    def pretrain(self, l_data_config, pretrain_config, model_path):
        if pretrain_config['mode']['name'] == 'inc_lengths':
            n_sessions = len(pretrain_config['mode']['in_seq_len'])
        elif pretrain_config['mode']['name'] == 'classic':
            n_sessions = 1
        else:
            raise Exception('pretraining mode not understood')

        current_epoch = 0
        for session_idx in range(n_sessions):
            tf.reset_default_graph()
            if pretrain_config['mode']['name'] == 'inc_lengths':
                pretrain_config['mode']['session_idx'] = session_idx
                max_epochs = pretrain_config['mode']['max_epochs'][session_idx]
                min_error = pretrain_config['mode']['min_errors'][session_idx]
                self.create_modificated_model(pretrain_config, l_data_config, pretrain_config['mode'])
            elif pretrain_config['mode']['name'] == 'classic':
                self.data_dict = load_dataset(l_data_config)
                l_data = LabeledData(l_data_config, self.data_dict)
                self.create_rnn(pretrain_config, l_data, l_data_config)
                max_epochs = pretrain_config['mode']['max_epochs']
                min_error = pretrain_config['mode']['min_error']

            # Saver is used for restoring weights for new session if more than one is used for training
            model_saver = tf.train.Saver(tf.trainable_variables())
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                if session_idx != 0:
                    model_saver.restore(sess, model_path)

                # Loading datasets into GPU (via tf.Variables)
                for key in self.data_dict.keys():
                    sess.run(self.l_data.data[key]['load'],
                             feed_dict={self.l_data.data[key]['x_ph']: self.data_dict[key]['x'],
                                        self.l_data.data[key]['y_ph']: self.data_dict[key]['y']})

                for epoch in range(max_epochs):
                    # Evaluate performance on the different datasets and print some results on console
                    # Also check potential stopping critera
                    self.rnn.t_metrics.retrieve_results(sess, current_epoch, is_pretrain=True)
                    if self.rnn.t_metrics.result_dict['tr_s']['m_loss'][-1] < min_error:
                        print(self.rnn.t_metrics.result_dict['tr_s']['m_acc'][-1])
                        break

                    # Train for one full epoch. First shuffle to create new minibatches from the given data and
                    # then do a training step for each minibatch.
                    sess.run(self.l_data.data['tr']['shuffle'])
                    for minibatch_idx in range(self.l_data.data['tr']['n_minibatches']):
                        sess.run(self.rnn.train_s_op,
                                 feed_dict={self.rnn.learning_rate: pretrain_config['learning_rate'],
                                            self.l_data.batch_idx: minibatch_idx})

                model_saver.save(sess, model_path)
                print(self.rnn.t_metrics.result_dict['tr_s']['m_acc'][-1])

