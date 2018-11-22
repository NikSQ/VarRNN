import tensorflow as tf
import numpy as np
from src.fp_tools import approx_activation, sample_activation, get_kl_loss
from src.tools import get_mean_initializer, get_var_initializer, get_xavier_initializer


# FC LAYER..
# Can only be used as output
class FCLayer:
    def __init__(self, rnn_config, layer_idx):
        self.rnn_config = rnn_config
        self.layer_config = rnn_config['layer_configs'][layer_idx]
        self.w_shape = (rnn_config['layout'][layer_idx-1], rnn_config['layout'][layer_idx])
        self.b_shape = (1, self.w_shape[1])

        with tf.variable_scope(self.layer_config['var_scope']):
            self.w_m = tf.get_variable(name='w_m', shape=self.w_shape,
                                       initializer=get_mean_initializer(self.layer_config['w'], self.w_shape))
            self.w_v = tf.exp(tf.get_variable(name='w_v', shape=self.w_shape,
                                              initializer=get_var_initializer(self.layer_config['w'], self.w_shape)))
            self.b_m = tf.get_variable(name='b_m', shape=self.b_shape,
                                       initializer=get_mean_initializer(self.layer_config['b'], self.b_shape))
            self.b_v = tf.exp(tf.get_variable(name='b_v', shape=self.b_shape,
                                              initializer=get_var_initializer(self.layer_config['b'], self.b_shape)))

            self.w = tf.get_variable(name='w', shape=self.w_shape, initializer=get_xavier_initializer(self.w_shape))
            self.b = tf.get_variable(name='b', shape=self.b_shape, initializer=tf.zeros_initializer(dtype=tf.float32))
            gauss = tf.distributions.Normal(loc=0., scale=1.)
            w_sample_op = tf.assign(self.w, self.w_m + gauss.sample(self.w_shape) * tf.sqrt(self.w_v))
            b_sample_op = tf.assign(self.b, self.b_m + gauss.sample(self.b_shape) * tf.sqrt(self.b_v))
            self.sample_op = tf.group(*[w_sample_op, b_sample_op])

            self.kl = get_kl_loss(self.layer_config['w'], self.w_m, self.w_v) + \
                get_kl_loss(self.layer_config['b'], self.b_m, self.b_v)

            summary_ops = list()
            summary_ops.append(tf.summary.histogram('w_m', self.w_m))
            summary_ops.append(tf.summary.histogram('w_v', self.w_v))
            summary_ops.append(tf.summary.histogram('b_m', self.b_m))
            summary_ops.append(tf.summary.histogram('b_v', self.b_v))
            self.weight_summaries = tf.summary.merge(summary_ops)

    # Returns the output of the layer. If its the output layer, this only returns the activation
    # TODO: Implement it for non-ouput case
    def create_pfp(self, x_m, x_v, mod_layer_config, init_cell_state):
        a_m, a_v = approx_activation(self.w_m, self.w_v, self.b_m, self.b_v, x_m, x_v)
        if self.layer_config['is_output']:
            return a_m, a_v
        else:
            raise Exception('fc layer can only be used as output')

    def create_sampling_pass(self, x, mod_layer_config, init_cell_state):
        return sample_activation(self.w_m, self.w_v, self.b_m, self.b_v, x)

    def create_fp(self, x, init_cell_state):
        if self.layer_config['is_output']:
            return tf.matmul(x, self.w) + self.b



