import tensorflow as tf
import numpy as np
from src.fp_tools import approx_activation
from src.weights import Weights
from src.tools import get_batchnormalizer
from src.global_variable import get_train_config, get_info_config, get_rnn_config


# FC LAYER..
# Can only be used as output
class FCLayer:
    def __init__(self, layer_idx, is_training, tau):
        rnn_config = get_rnn_config()
        self.train_config = get_train_config()
        self.layer_config = rnn_config['layer_configs'][layer_idx]
        self.w_shape = (rnn_config['layout'][layer_idx-1], rnn_config['layout'][layer_idx])
        self.b_shape = (1, self.w_shape[1])
        self.is_training = is_training

        # Activation summaries and specific neurons to gather individual histograms
        self.acts = dict()
        self.act_neurons = np.random.choice(range(self.b_shape[1]),
                                            size=(get_info_config()['tensorboard']['single_acts'],), replace=False)

        self.bn_s_x = get_batchnormalizer()
        self.bn_b_x = get_batchnormalizer()

        with tf.variable_scope(self.layer_config['var_scope']):
            var_keys = ['w', 'b']
            self.weights = Weights(var_keys, self.layer_config, self.w_shape, self.b_shape, tau)

    def create_pfp(self, x_m, x_v, mod_layer_config, init):
        a_m, a_v = approx_activation(self.weights.var_dict['w_m'], self.weights.var_dict['w_v'], self.weights.var_dict['b_m'], self.weights.var_dict['b_v'], x_m, x_v)
        return a_m, a_v

    def create_l_sampling_pass(self, x, mod_layer_config, time_idx):
        if 'fc' in self.train_config['batchnorm']['modes']:
            x = self.bn_b_x(x, self.is_training)
        return self.weights.sample_activation('w', 'b', x, None, time_idx == 0)

    def create_g_sampling_pass(self, x, mod_layer_config, time_idx):
        if time_idx == 0:
            self.weights.create_tensor_samples()
        if 'fc' in self.train_config['batchnorm']['modes']:
            x = self.bn_b_x(x, self.is_training)
        return tf.matmul(x, self.weights.tensor_dict['w']) + self.weights.tensor_dict['b']

    def create_var_fp(self, x, time_idx):
        if 'fc' in self.train_config['batchnorm']['modes']:
            x = self.bn_s_x(x, self.is_training)
        act = tf.matmul(x, self.weights.var_dict['w']) + self.weights.var_dict['b']

        # Store activations over layer and over single neurons.
        if time_idx == 0:
            self.acts['n'] = act
            for neuron_idc in range(len(self.act_neurons)):
                self.acts['n' + '_' + str(neuron_idc)] = tf.slice(act, begin=(0, neuron_idc), size=(-1, 1))
        else:
            self.acts['n'] = tf.concat([self.acts['n'], act], 0)
            for neuron_idc in range(len(self.act_neurons)):
                self.acts['n' + '_' + str(neuron_idc)] = tf.concat([tf.slice(act, begin=(0, neuron_idc), size=(-1, 1)),
                                                                    self.acts['n_' + str(neuron_idc)]], axis=0)

        return act



