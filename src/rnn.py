import tensorflow as tf
import numpy as np
from src.fc_layer import FCLayer
from src.lstm_layer import LSTMLayer
import copy




class RNN:
    def __init__(self, rnn_config, training_config, labelled_data):
        self.c = None
        self.rnn_config = rnn_config
        self.training_config = training_config
        self.labelled_data = labelled_data
        self.layers = []
        self.train_op = None
        self.accuracy = None
        self.gradients = None  # Used to find a good value to clip

        self.tr_acc = None
        self.tr_pred = None
        self.tr_loss = None
        self.tr_out = None
        self.tr_summary = None

        self.va_acc = None
        self.va_pred = None
        self.va_loss = None
        self.va_out = None
        self.va_summary = None

        with tf.variable_scope('global'):
            self.learning_rate = tf.placeholder(tf.float32)

        weight_summaries = []
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
            self.layers.append(layer)
        self.weight_summaries = tf.summary.merge(weight_summaries)
        self.gradient_summaries = None

    # TODO: Make predictions based on predictive distribution rather than on mode
    def create_rnn_graph(self, x, y, x_shape, y_shape, mod_rnn_config):
        # Elements of the list will be the means and variances of the output sequence
        m_outputs = []
        v_outputs = []

        # This captures the seq_idx from which on the output will be computed
        start_output_idx = x_shape[2] - y_shape[2]

        # Create graph by connecting the appropriate layers
        m = None
        v = None
        for seq_idx in range(x_shape[2]):
            m = x[:, :, seq_idx]  # Mean of input to network at time seq_idx
            v = tf.fill(tf.shape(m), 0.)  # Variance of input to network at time seq_idx

            for layer_idx, layer in enumerate(self.layers, 1):
                if (seq_idx >= start_output_idx) or layer.layer_config['is_recurrent']:
                    if self.training_config['type'] == 'pfp':
                        m, v = layer.create_pfp(m, v, mod_rnn_config['layer_configs'][layer_idx],
                                                seq_idx == 0)
                    elif self.training_config['type'] == 'sampling':
                        m = layer.create_sampling_pass(m, mod_rnn_config['layer_configs'][layer_idx], seq_idx == 0)
                    else:
                        raise Exception('Training type not understood')

            if seq_idx >= start_output_idx:
                m_outputs.append(tf.expand_dims(m, axis=2))
                v_outputs.append(tf.expand_dims(v, axis=2))

        m_output = tf.cast(tf.concat(m_outputs, axis=2), dtype=tf.float64)
        v_output = tf.cast(tf.concat(v_outputs, axis=2), dtype=tf.float64)

        if self.rnn_config['output_type'] == 'classification' and self.training_config['type'] == 'pfp':
            smax = tf.nn.softmax(logits=m_output, axis=1)
            t = tf.argmax(y, axis=1)
            batch_range = np.arange(y_shape[0])
            seq_range = np.arange(y_shape[2])
            seqq, batchh = np.meshgrid(seq_range, batch_range)
            g_indices = tf.concat([np.expand_dims(batchh, axis=2), tf.expand_dims(t, axis=2),
                                  np.expand_dims(seqq, axis=2)], axis=2)
            e_log_likelihood = tf.log(np.finfo(np.float64).eps + tf.reduce_mean(tf.gather_nd(smax, g_indices))) - \
                0.5 * tf.reduce_mean(tf.reduce_sum(tf.multiply(v_output, tf.multiply(smax, 1 - smax)), axis=1))
            kl_loss = 0
            for layer in self.layers:
                kl_loss = kl_loss + layer.kl_loss

            #var_free_energy = tf.cast(kl_loss, tf.float64) / self.rnn_config['data_multiplier'] - e_log_likelihood
            var_free_energy = - e_log_likelihood
            prediction = tf.argmax(smax, axis=1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, t), dtype=tf.float32))
            output = None
        elif self.rnn_config['output_type'] == 'classification' and self.training_config['type'] == 'sampling':
            smax = tf.nn.softmax(logits=m_output, axis=1)
            t = tf.argmax(y, axis=1)
            kl_loss = 0
            for layer in self.layers:
                kl_loss = kl_loss + layer.kl_loss
            var_free_energy = tf.cast(kl_loss, dtype=tf.float64) / self.rnn_config['data_multiplier'] \
                              + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=m_output, labels=y, dim=1))
            prediction = tf.argmax(smax, axis=1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, t), dtype=tf.float32))
            output = None
        else:
            raise Exception('output type of RNN not understood')

        return var_free_energy, prediction, accuracy, output

    def create_training_graph(self):
        with tf.variable_scope('training'):
            self.tr_loss, self.tr_pred, self.tr_acc, self.tr_out = \
                self.create_rnn_graph(self.labelled_data.x_tr_batch, self.labelled_data.y_tr_batch,
                                      self.labelled_data.x_tr_shape, self.labelled_data.y_tr_shape, self.rnn_config)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.gradients = optimizer.compute_gradients(self.tr_loss)

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
            self.train_op = optimizer.apply_gradients(clipped_gradients)
            self.tr_summary = tf.summary.merge([tf.summary.scalar('tr_loss', self.tr_loss),
                                                tf.summary.scalar('tr_acc', self.tr_acc)])

    def create_validation_graph(self):
        with tf.variable_scope('validation'):
            graph_config = copy.deepcopy(self.rnn_config)
            for layer_config in graph_config['layer_configs']:
                if 'regularization' in layer_config:
                    layer_config['regularization']['mode'] = None

            self.va_loss, self.va_pred, self.va_acc, self.va_out = \
                self.create_rnn_graph(self.labelled_data.x_va_batch, self.labelled_data.y_va_batch,
                                      self.labelled_data.x_va_shape, self.labelled_data.y_va_shape, graph_config)
            self.va_summary = tf.summary.merge([tf.summary.scalar('va_loss', self.va_loss),
                                                tf.summary.scalar('va_acc', self.va_acc)])

