import tensorflow as tf
import numpy as np
from src.fp_tools import approx_activation
from src.weights import Weights
from src.tools import get_batchnormalizer
from src.global_variable import get_train_config, get_info_config, get_rnn_config


# FC LAYER..
# Can only be used as output
class FCLayer:
    def __init__(self, layer_idx, is_training, tau, prev_neurons = None):
        rnn_config = get_rnn_config()
        self.train_config = get_train_config()
        self.layer_config = rnn_config['layer_configs'][layer_idx]
        if prev_neurons is None:
            self.w_shape = (rnn_config['layout'][layer_idx-1], rnn_config['layout'][layer_idx])
        else:
            self.w_shape = (prev_neurons, rnn_config['layout'][layer_idx])
        self.b_shape = (1, self.w_shape[1])
        self.is_training = is_training

        # Activation summaries and specific neurons to gather individual histograms
        self.acts = dict()
        self.act_neurons = np.random.choice(range(self.b_shape[1]),
                                            size=(get_info_config()['tensorboard']['single_acts'],), replace=False)
        if self.train_config['batchnorm']['type'] == 'batch' and 'fc' in self.train_config['batchnorm']['modes']:
            self.bn_s_x = get_batchnormalizer()
            self.bn_b_x = get_batchnormalizer()

        with tf.variable_scope(self.layer_config['var_scope']):
            var_keys = ['w', 'b']
            self.weights = Weights(var_keys, self.layer_config, self.w_shape, self.b_shape, tau)

    def create_pfp(self, x_m, x_v, mod_layer_config, init):
        a_m, a_v = approx_activation(self.weights.var_dict['w_m'], self.weights.var_dict['w_v'], self.weights.var_dict['b_m'], self.weights.var_dict['b_v'], x_m, x_v)
        return a_m, a_v

    def create_l_sampling_pass(self, x, mod_layer_config, time_idx, init, init_cell=None):
        if self.train_config['batchnorm']['type'] == 'batch' and 'fc' in self.train_config['batchnorm']['modes']:
            x = self.bn_b_x(x, self.is_training)
        return self.weights.sample_activation('w', 'b', x, None, init,
                                              self.train_config['batchnorm']['type'] == 'layer'), None

    def create_g_sampling_pass(self, x, mod_layer_config, time_idx, init, second_arm_pass=False, data_key=None, init_cell=None):
        if init:
            self.weights.create_tensor_samples(second_arm_pass=second_arm_pass, data_key=data_key)
        if self.train_config['batchnorm']['type'] == 'batch' and 'fc' in self.train_config['batchnorm']['modes']:
            x = self.bn_b_x(x, self.is_training)

        act = tf.matmul(x, self.weights.tensor_dict['w'])
        if self.layer_config['bias_enabled']:
            act += self.weights.tensor_dict['b']
        if self.train_config['batchnorm']['type'] == 'layer':
            return tf.contrib.layers.layer_norm(act)
        else:
            return act, None

    def create_var_fp(self, x, time_idx, init, init_cell=None):
        if self.train_config['batchnorm']['type'] == 'batch' and 'fc' in self.train_config['batchnorm']['modes']:
            x = self.bn_s_x(x, self.is_training)

        act = tf.matmul(x, self.weights.var_dict['w'])
        if self.layer_config['bias_enabled']:
            act += self.weights.var_dict['b']
        if self.train_config['batchnorm']['type'] == 'layer':
            act = tf.contrib.layers.layer_norm(act)

        # Store activations over layer and over single neurons.
        if get_rnn_config()['architecture'] == 'encoder':
            return act, None
        if time_idx == 0:
            self.acts['n'] = act
            for neuron_idc in range(len(self.act_neurons)):
                self.acts['n' + '_' + str(neuron_idc)] = tf.slice(act, begin=(0, neuron_idc), size=(-1, 1))
        else:
            self.acts['n'] = tf.concat([self.acts['n'], act], 0)
            for neuron_idc in range(len(self.act_neurons)):
                self.acts['n' + '_' + str(neuron_idc)] = tf.concat([tf.slice(act, begin=(0, neuron_idc), size=(-1, 1)),
                                                                    self.acts['n_' + str(neuron_idc)]], axis=0)

        return act, None



