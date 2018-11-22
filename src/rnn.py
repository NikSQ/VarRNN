import tensorflow as tf
import numpy as np
from src.fc_layer import FCLayer
from src.lstm_layer import LSTMLayer
from src.data.t_metrics import TMetrics
import copy


class RNN:
    def __init__(self, rnn_config, training_config, info_config, l_data):
        self.c = None
        self.rnn_config = rnn_config
        self.training_config = training_config
        self.l_data = l_data
        self.layers = []
        self.train_b_op = None
        self.train_s_op = None
        self.accuracy = None
        self.gradients = None  # Used to find a good value to clip
        self.t_metrics = TMetrics(l_data.l_data_config, l_data, info_config)
        self.t_metric_summaries = None

        with tf.variable_scope('global'):
            self.learning_rate = tf.placeholder(tf.float32)

        weight_summaries = []
        sample_ops = []
        for layer_idx, layer_config in enumerate(self.rnn_config['layer_configs']):
            if layer_config['layer_type'] == 'fc':
                layer = FCLayer(rnn_config, layer_idx)
            elif layer_config['layer_type'] == 'lstm':
                layer = LSTMLayer(rnn_config, layer_idx)
            elif layer_config['layer_type'] == 'input':
                continue
            else:
                raise Exception("{} is not a valid layer type".format(layer_config['layer_type']))
            weight_summaries.append(layer.weight_summaries)
            sample_ops.append(layer.sample_op)
            self.layers.append(layer)

        self.sample_op = tf.group(*sample_ops)
        self.weight_summaries = tf.summary.merge(weight_summaries)
        self.gradient_summaries = None

        self.create_b_training_graph('tr')
        for data_key in l_data.data:
            if data_key != 'tr':
                self.create_b_evaluation_graph(data_key)
                self.create_s_evaluation_graph(data_key)

    # TODO: Make predictions based on predictive distribution rather than on mode
    def create_rnn_graph(self, data_key, mod_rnn_config, bayesian=True, record=True):
        x = self.l_data.data[data_key]['x']
        y = self.l_data.data[data_key]['y']
        x_shape = self.l_data.data[data_key]['x_shape']
        y_shape = self.l_data.data[data_key]['y_shape']

        # Elements of the list will be the means and variances of the output sequence
        m_outputs = []
        v_outputs = []
        outputs = []

        # This captures the seq_idx from which on the output will be computed
        start_output_idx = x_shape[2] - y_shape[2]

        # Create graph by connecting the appropriate layers unrolled in time
        for seq_idx in range(x_shape[2]):
            m = x[:, :, seq_idx]  # Mean of input to network at time seq_idx
            v = tf.fill(tf.shape(m), 0.)  # Variance of input to network at time seq_idx

            for layer_idx, layer in enumerate(self.layers, 1):
                if (seq_idx >= start_output_idx) or layer.layer_config['is_recurrent']:
                    if bayesian is False:
                        m = layer.create_fp(m, seq_idx == 0)
                    elif self.training_config['type'] == 'pfp':
                        m, v = layer.create_pfp(m, v, mod_rnn_config['layer_configs'][layer_idx], seq_idx == 0)
                    elif self.training_config['type'] == 'sampling':
                        m = layer.create_sampling_pass(m, mod_rnn_config['layer_configs'][layer_idx], seq_idx == 0)
                    else:
                        raise Exception('Training type not understood')

            if seq_idx >= start_output_idx:
                if bayesian:
                    m_outputs.append(tf.expand_dims(m, axis=2))
                    v_outputs.append(tf.expand_dims(v, axis=2))
                else:
                    outputs.append(tf.expand_dims(m, axis=2))

        # Process output of non bayesian network
        if bayesian is False:
            output = tf.cast(tf.concat(outputs, axis=2), dtype=tf.float64)
            if self.rnn_config['output_type'] == 'classification':
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y, dim=1))
                prediction = tf.argmax(output, axis=1)
                acc = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y, axis=1)), dtype=tf.float32))
            else:
                raise Exception('output type of RNN not understood')

            if record:
                self.t_metrics.add_s_vars(data_key + '_s', self.sample_op, loss, acc)
                if self.t_metric_summaries is not None:
                    self.t_metric_summaries = tf.summary.merge([self.t_metric_summaries,
                                                                tf.summary.scalar(data_key + '_s_loss', loss),
                                                                tf.summary.scalar(data_key + '_s_acc', acc)])
                else:
                    self.t_metric_summaries = tf.summary.merge([tf.summary.scalar(data_key + '_s_loss', loss),
                                                                tf.summary.scalar(data_key + '_s_acc', acc)])
            return loss, acc

        # Process output of bayesian network
        else:
            if self.rnn_config['output_type'] == 'classification' and self.training_config['type'] == 'pfp':
                m_output = tf.cast(tf.concat(m_outputs, axis=2), dtype=tf.float64)
                v_output = tf.cast(tf.concat(v_outputs, axis=2), dtype=tf.float64)
                smax = tf.nn.softmax(logits=m_output, axis=1)
                t = tf.argmax(y, axis=1)
                batch_range = np.arange(y_shape[0])
                seq_range = np.arange(y_shape[2])
                seqq, batchh = np.meshgrid(seq_range, batch_range)
                g_indices = tf.concat([np.expand_dims(batchh, axis=2), tf.expand_dims(t, axis=2),
                                      np.expand_dims(seqq, axis=2)], axis=2)
                elogl = tf.log(np.finfo(np.float64).eps + tf.reduce_mean(tf.gather_nd(smax, g_indices))) - \
                    0.5 * tf.reduce_mean(tf.reduce_sum(tf.multiply(v_output, tf.multiply(smax, 1 - smax)), axis=1))
                kl = 0
                for layer in self.layers:
                    kl = kl + layer.kl
                nelbo = tf.cast(kl, tf.float64) / self.rnn_config['data_multiplier'] - elogl
                prediction = tf.argmax(smax, axis=1)
                acc = tf.reduce_mean(tf.cast(tf.equal(prediction, t), dtype=tf.float32))
            elif self.rnn_config['output_type'] == 'classification' and self.training_config['type'] == 'sampling':
                m_output = tf.cast(tf.concat(m_outputs, axis=2), dtype=tf.float64)
                smax = tf.nn.softmax(logits=m_output, axis=1)
                t = tf.argmax(y, axis=1)
                kl = 0
                elogl = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=m_output, labels=y, dim=1))
                for layer in self.layers:
                    kl = kl + layer.kl
                nelbo = tf.cast(kl, dtype=tf.float64) / self.rnn_config['data_multiplier'] - elogl
                prediction = tf.argmax(smax, axis=1)
                acc = tf.reduce_mean(tf.cast(tf.equal(prediction, t), dtype=tf.float32))
            else:
                raise Exception('output type of RNN not understood')

            if record:
                self.t_metrics.add_b_vars(data_key + '_b', nelbo, kl, elogl, acc)

                if self.t_metric_summaries is not None:
                    self.t_metric_summaries = tf.summary.merge([self.t_metric_summaries,
                                                                tf.summary.scalar(data_key + '_b_nelbo', nelbo),
                                                                tf.summary.scalar(data_key + '_b_kl', kl),
                                                                tf.summary.scalar(data_key + '_b_elogl', elogl),
                                                                tf.summary.scalar(data_key + '_b_acc', acc)])
                else:
                    self.t_metric_summaries = tf.summary.merge([tf.summary.scalar(data_key + '_b_nelbo', nelbo),
                                                                tf.summary.scalar(data_key + '_b_kl', kl),
                                                                tf.summary.scalar(data_key + '_b_elogl', elogl),
                                                                tf.summary.scalar(data_key + '_b_acc', acc)])
            return nelbo, kl, elogl, acc

    # Creates Bayesian graph for training and the operations used for training.
    def create_b_training_graph(self, key):
        with tf.variable_scope(key + '_b'):
            nelbo, kl, elogl, acc = self.create_rnn_graph(key, self.rnn_config, record=True)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.gradients = optimizer.compute_gradients(nelbo)

            gradient_summaries = []
            for layer_idx in range(len(self.gradients)):
                if self.gradients[layer_idx][0] is not None:
                    gradient_summaries.append(tf.summary.histogram('gradients',
                                                                   self.gradients[layer_idx][0]))
            self.gradient_summaries = tf.summary.merge(gradient_summaries)

            clipped_gradients = [(grad, var) if grad is None else
                                 (tf.clip_by_value(grad, -self.rnn_config['gradient_clip_value'],
                                                   self.rnn_config['gradient_clip_value']), var)
                                 for grad, var in self.gradients]
            self.train_b_op = optimizer.apply_gradients(clipped_gradients)

    # Creates non-Bayesian graph for training the RNN
    def create_s_training_graph(self, key):
        with tf.variable_scope(key + '_s'):
            loss, accuracy = self.create_rnn_graph(key, None, bayesian=False, record=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.gradients = optimizer.compute_gradients(loss)
            clipped_gradients = [(grad, var) if grad is None else
                                 (tf.clip_by_value(grad, -self.rnn_config['gradient_clip_value'],
                                                   self.rnn_config['gradient_clip_value']), var)
                                 for grad, var in self.gradients]
            self.train_s_op = optimizer.apply_gradients(clipped_gradients)

    # Evaluation graphs are not used for training but only for evaluation. A modified RNN config file is used which
    # overwrites the one used for training. Can be used to modify the forward pass to turn off dropout while validating
    # and testing for example.
    def create_b_evaluation_graph(self, key):
        with tf.variable_scope(key + '_b'):
            mod_rnn_config = copy.deepcopy(self.rnn_config)
            for layer_config in mod_rnn_config['layer_configs']:
                if 'regularization' in layer_config:
                    layer_config['regularization']['mode'] = None

            self.create_rnn_graph(key, mod_rnn_config, record=True)

    def create_s_evaluation_graph(self, data_key):
        with tf.variable_scope(data_key + '_s'):
            self.create_rnn_graph(data_key, None, bayesian=False, record=True)





