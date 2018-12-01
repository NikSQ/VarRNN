import tensorflow as tf
from tensorflow.python.client import timeline
from src.data.loader import load_dataset
from src.data.labeled_data import LabeledData
from src.rnn import RNN
from src.tools import print_config


class Experiment:
    def __init__(self):
        self.rnn = None
        self.l_data = None
        self.l_data_config = None
        self.rnn_config = None
        self.data_dict = None
        self.training_config = None
        self.info_config = None

    def create_rnn(self, rnn_config, l_data, l_data_config):
        if self.rnn_config is None:
            self.rnn_config = rnn_config
        self.rnn = RNN(rnn_config, self.training_config, self.info_config, l_data)

        self.l_data = l_data
        self.l_data_config = l_data_config

    # Creates a RNN using a modified l_data_config
    # Used e.g. by incremental sequence training, where the l_data_config is changed while training
    def create_modificated_model(self, rnn_config, l_data_config, mod_data_config):
        incremental_idx = mod_data_config['session_idx']
        l_data_config['tr']['in_seq_len'] = mod_data_config['in_seq_len'][incremental_idx]
        l_data_config['tr']['out_seq_len'] = mod_data_config['out_seq_len'][incremental_idx]
        l_data_config['tr']['zero_padding'] = mod_data_config['zero_padding'][incremental_idx]
        self.data_dict = load_dataset(l_data_config)
        labeled_data = LabeledData(l_data_config, self.data_dict)
        self.create_rnn(rnn_config, labeled_data, l_data_config)

    def train(self, rnn_config, l_data_config, training_config, info_config):
        self.training_config = training_config
        self.info_config = info_config
        print_config(rnn_config, training_config, l_data_config)
        temp_model_path = '../models/temp' + str(training_config['task_id'])

        if training_config['mode']['name'] == 'inc_lengths':
            n_sessions = len(training_config['mode']['in_seq_len'])
        elif training_config['mode']['name'] == 'classic':
            n_sessions = 1
        else:
            raise Exception('training mode not understood')

        # TODO: Train network to initialize discrete one

        # Sessions refer to training with different architectures. If one RNN is used throughout the training process
        # then only one session is created. Training with incremental sequence lengths for example requires multiple
        # RNNs, one for each sequence lenghts. Evaluation datasets (validation and test) are always evaluated on a fixed
        # RNN, only the RNN structure used for the training set varies. current_epoch stores the total amounts of epochs
        # and epoch the epoch within a session
        current_epoch = 0
        for session_idx in range(n_sessions):
            tf.reset_default_graph()
            if training_config['mode']['name'] == 'inc_lengths':
                training_config['mode']['session_idx'] = session_idx
                max_epochs = training_config['mode']['max_epochs'][session_idx]
                min_error = training_config['mode']['min_errors'][session_idx]
                self.create_modificated_model(rnn_config, l_data_config, training_config['mode'])
            elif training_config['mode']['name'] == 'classic':
                self.data_dict = load_dataset(l_data_config)
                l_data = LabeledData(l_data_config, self.data_dict)
                self.create_rnn(rnn_config, l_data, l_data_config)
                max_epochs = training_config['mode']['max_epochs']
                min_error = training_config['mode']['min_error']

            # Saver is used for restoring weights for new session if more than one is used for training
            model_saver = tf.train.Saver(tf.trainable_variables())
            with tf.Session() as sess:
                if info_config['profiling']['enabled']:
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                else:
                    options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
                run_metadata = tf.RunMetadata()
                writer = tf.summary.FileWriter(info_config['tensorboard']['path'] + str(training_config['task_id']))
                sess.run(tf.global_variables_initializer())
                if session_idx != 0:
                    model_saver.restore(sess, temp_model_path)

                # Loading datasets into GPU (via tf.Variables)
                for key in self.data_dict.keys():
                    sess.run(self.l_data.data[key]['load'],
                             feed_dict={self.l_data.data[key]['x_ph']: self.data_dict[key]['x'],
                                        self.l_data.data[key]['y_ph']: self.data_dict[key]['y']})
                for epoch in range(max_epochs):
                    # Evaluate performance on the different datasets and print some results on console
                    # Also check potential stopping critera
                    if current_epoch % info_config['calc_performance_every'] == 0:
                        self.rnn.t_metrics.retrieve_results(sess, current_epoch)
                        self.rnn.t_metrics.print(session_idx)
                        if self.rnn.t_metrics.result_dict['tr_b']['nelbo'][-1] < min_error:
                            break

                    # Train for one full epoch. First shuffle to create new minibatches from the given data and
                    # then do a training step for each minibatch.
                    sess.run(self.l_data.data['tr']['shuffle'])
                    traces = list()
                    for minibatch_idx in range(self.l_data.data['tr']['n_minibatches']):
                        sess.run(self.rnn.train_b_op,
                                 feed_dict={self.rnn.learning_rate: training_config['learning_rate'],
                                            self.l_data.batch_idx: minibatch_idx},
                                 options=options, run_metadata=run_metadata)
                    traces.append(timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format())

                    # Optionally store profiling results of this epoch in files
                    if info_config['profiling']['enabled']:
                        for trace_idx, trace in enumerate(traces):
                            path = info_config['profiling']['path'] + '_' + str(current_epoch) + '_' + str(trace_idx)
                            with open(path + 'training.json', 'w') as f:
                                f.write(trace)

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
                    current_epoch += 1

                model_saver.save(sess, temp_model_path)
        writer.close()
        return self.rnn.t_metrics.result_dict

