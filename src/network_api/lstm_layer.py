import tensorflow as tf
import numpy as np

from src.fp_tools import approx_activation, transform_tanh_activation, transform_sig_activation
from src.weights import Weights
from src.tools import get_batchnormalizer
from src.global_variable import get_train_config, get_info_config, get_nn_config
from src.configuration.constants import ActivationFunctionsC, GraphCreationKeys


@tf.custom_gradient
def disc_sigmoid(act, n_bins):
    s_act = tf.sigmoid(act)
    disc_output = tf.cast(tf.cast(s_act*n_bins, dtype=tf.int32), dtype=tf.float32) / n_bins

    def grad(dy):
        return dy * tf.multiply(s_act, 1-s_act), tf.zeros_like(n_bins)

    return disc_output, grad


@tf.custom_gradient
def disc_tanh(act, n_bins):
    disc_output = tf.cast(tf.cast(tf.sigmoid(act) * n_bins, dtype=tf.int32), dtype=tf.float32) * 2 / n_bins - 1

    def grad(dy):
        return dy * (1 - tf.square(tf.tanh(act))), tf.zeros_like(n_bins)

    return disc_output, grad


class LSTMLayer:
    def __init__(self, layer_idx, is_training, tau, bidirectional_inp=False, prev_neurons=None):
        self.nn_config = get_nn_config()
        self.train_config = get_train_config()
        self.layer_config = self.nn_config.layer_configs[layer_idx]

        p_layout = self.nn_config.layout[layer_idx]
        c_layout = self.nn_config.layout[layer_idx + 1]
        if prev_neurons is None:
            if bidirectional_inp:
                self.w_shape = (p_layout * 2 + c_layout, c_layout)
            else:
                self.w_shape = (p_layout + c_layout, c_layout)
        else:
            self.w_shape = (prev_neurons + c_layout, c_layout)

        self.b_shape = (1, self.w_shape[1])
        self.cell_access_mat = []
        self.is_training = is_training

        # Activation summaries and specific neurons to gather individual histograms
        self.acts = dict()
        self.act_neurons = np.random.choice(range(self.b_shape[1]),
                                            size=(get_info_config().tensorboard_config.record_n_neurons,), replace=False)

        """
        if self.train_config['batchnorm']['type'] == 'batch' and 'x' in self.train_config['batchnorm']['modes']:
            self.bn_b_x = []
            self.bn_s_x = []
        if self.train_config['batchnorm']['type'] == 'batch' and 'h' in self.train_config['batchnorm']['modes']:
            self.bn_b_h = []
            self.bn_s_h = []
        """

        with tf.variable_scope(self.layer_config.var_scope):
            var_keys = ['wi', 'bi', 'wc', 'bc', 'wo', 'bo']
            self.weights = Weights(var_keys, self.layer_config, self.w_shape, self.b_shape, tau)

    # TODO: Update PFP (currently does not work)
    def create_pfp(self, x_m, x_v, mod_layer_config, init, init_cell=None):
        if init:
            cell_shape = (tf.shape(x_m)[0], self.b_shape[1])
            self.weights.tensor_dict['cs_m'] = tf.zeros(cell_shape)
            self.weights.tensor_dict['cs_v'] = tf.zeros(cell_shape)
            self.weights.tensor_dict['co_m'] = tf.zeros(cell_shape)
            self.weights.tensor_dict['co_v'] = tf.zeros(cell_shape)

        if self.train_config['batchnorm']:
            raise Exception('Batchnorm not implemented for probabilistic forward pass')

        # Vector concatenation (input with recurrent)
        m = tf.concat([x_m, self.weights.tensor_dict['co_m']], axis=1)
        v = tf.concat([x_v, self.weights.tensor_dict['co_v']], axis=1)

        a_i_m, a_i_v = approx_activation(self.weights.var_dict['wi_m'], self.weights.var_dict['wi_v'], self.weights.var_dict['bi_m'], self.weights.var_dict['bi_v'], m, v)
        i_m, i_v = transform_sig_activation(a_i_m, a_i_v)
        a_c_m, a_c_v = approx_activation(self.weights.var_dict['wc_m'], self.weights.var_dict['wc_v'], self.weights.var_dict['bc_m'], self.weights.var_dict['bc_v'], m, v)
        c_m, c_v = transform_tanh_activation(a_c_m, a_c_v)
        a_o_m, a_o_v = approx_activation(self.weights.var_dict['wo_m'], self.weights.var_dict['wo_v'], self.weights.var_dict['bo_m'], self.weights.var_dict['bo_v'], m, v)
        o_m, o_v = transform_sig_activation(a_o_m, a_o_v)

        f_m = 1 - i_m
        f_v = i_v
        f_2nd_mom = tf.square(f_m) + f_v
        i_2nd_mom = tf.square(i_m) + i_v
        self.weights.tensor_dict['cs_v'] = tf.multiply(self.weights.tensor_dict['cs_v'], f_2nd_mom) + tf.multiply(c_v, i_2nd_mom) + \
                                           tf.multiply(tf.square(self.weights.tensor_dict['cs_m']), f_v) + tf.multiply(tf.square(c_m), i_v)
        self.weights.tensor_dict['cs_m'] = tf.multiply(f_m, self.weights.tensor_dict['cs_m']) + tf.multiply(i_m, c_m)

        c_tan_m, c_tan_v = transform_tanh_activation(self.weights.tensor_dict['cs_m'], self.weights.tensor_dict['cs_v'])
        o_2nd_mom = tf.square(o_m) + o_v
        self.weights.tensor_dict['co_m'] = tf.multiply(c_tan_m, o_m)
        self.weights.tensor_dict['co_v'] = tf.multiply(c_tan_v, o_2nd_mom) + tf.multiply(tf.square(c_tan_m), o_v)

        return self.weights.tensor_dict['co_m'], self.weights.tensor_dict['co_v']

    # Local reparametrization trick
    def create_l_sampling_pass(self, x, initialize, time_index, **kwargs):
        if initialize:
            self.init_cell_tensors(x, **kwargs)

        co = self.weights.tensor_dict['co']

        """
        if self.train_config['batchnorm']['type'] == 'batch':
            bn_idx = min(time_index, self.train_config['batchnorm']['tau'] - 1)
            if 'x' in self.train_config['batchnorm']['modes']:
                if len(self.bn_b_x) == bn_idx:
                    self.bn_b_x.append(get_batchnormalizer())
                x = self.bn_b_x[bn_idx](x, self.is_training)
            if 'h' in self.train_config['batchnorm']['modes'] and bn_idx > 0:
                if len(self.bn_b_h) == bn_idx - 1:
                    self.bn_b_h.append(get_batchnormalizer())
                co = self.bn_b_h[bn_idx - 1](co, self.is_training)
        """

        x = tf.concat([x, co], axis=1)

        if self.layer_config.i_gate_config.is_act_func_discrete:
            i = self.weights.sample_activation('wi', 'bi', x, ActivationFunctionsC.SIGMOID, initialize, layer_norm=False)
        else:
            a_i = self.weights.sample_activation('wi', 'bi', x, None, initialize, layer_norm=False)
            i = tf.sigmoid(a_i)
        f = 1. - i

        if self.layer_config.c_gate_config.is_act_func_discrete:
            c = self.weights.sample_activation('wc', 'bc', x, ActivationFunctionsC.TANH, initialize, layer_norm=False)
        else:
            a_c = self.weights.sample_activation('wc', 'bc', x, None, initialize, layer_norm=False)
            c = tf.tanh(a_c)

        if self.layer_config.o_gate_config.is_act_func_discrete:
            o = self.weights.sample_activation('wo', 'bo', x, ActivationFunctionsC.SIGMOID, initialize, layer_norm=False)
        else:
            a_o = self.weights.sample_activation('wo', 'bo', x, None, initialize, layer_norm=False)
            o = tf.sigmoid(a_o)

        self.weights.tensor_dict['cs'] = tf.multiply(f, self.weights.tensor_dict['cs']) + tf.multiply(i, c)

        if self.layer_config.i_gate_config.is_act_func_discrete:
            self.weights.tensor_dict['co'] = tf.multiply(self.weights.tensor_dict['cs'], o)
        else:
            self.weights.tensor_dict['co'] = tf.multiply(tf.tanh(self.weights.tensor_dict['cs']), o)

        return self.weights.tensor_dict['co'], self.weights.tensor_dict['cs']

    # Samples weights before computing activation
    def create_sampling_pass(self, x, initialize, time_index, **kwargs):
        if initialize:
            self.weights.create_tensor_samples(second_arm_pass=kwargs[GraphCreationKeys.SECOND_ARM_PASS],
                                               data_key=kwargs[GraphCreationKeys.DATA_KEY])
            self.init_cell_tensors(x, **kwargs)

        co = self.weights.tensor_dict['co']

        """
        if self.train_config['batchnorm']['type'] == 'batch':
            bn_idx = min(time_index, self.train_config['batchnorm']['tau'] - 1)
            if 'x' in self.train_config['batchnorm']['modes']:
                if len(self.bn_b_x) == bn_idx:
                    self.bn_b_x.append(get_batchnormalizer())
                x = self.bn_b_x[bn_idx](x, self.is_training)
            if 'h' in self.train_config['batchnorm']['modes'] and bn_idx > 0:
                if len(self.bn_b_h) == bn_idx - 1:
                    self.bn_b_h.append(get_batchnormalizer())
                co = self.bn_b_h[bn_idx - 1](co, self.is_training)
        """

        x = tf.concat([x, co], axis=1)

        i_act = tf.matmul(x, self.weights.tensor_dict['wi']) + self.weights.tensor_dict['bi']
        c_act = tf.matmul(x, self.weights.tensor_dict['wc']) + self.weights.tensor_dict['bc']
        o_act = tf.matmul(x, self.weights.tensor_dict['wo']) + self.weights.tensor_dict['bo']

        if self.train_config['batchnorm']['type'] == 'layer':
            i_act = tf.contrib.layers.layer_norm(i_act)
            c_act = tf.contrib.layers.layer_norm(c_act)
            o_act = tf.contrib.layers.layer_norm(o_act)

        if self.layer_config.i_gate_config.is_act_func_discrete:
            i = disc_sigmoid(i_act, self.layer_config.i_gate_config.n_act_bins)
        else:
            i = tf.sigmoid(i_act)

        f = 1. - i
        if self.layer_config.c_gate_config.is_act_func_discrete:
            c = disc_tanh(c_act, self.layer_config.c_gate_config.n_act_bins)
        else:
            c = tf.tanh(c_act)

        if self.layer_config.o_gate_config.is_act_func_discrete:
            o = disc_sigmoid(o_act, self.layer_config.o_gate_config.n_act_bins)
        else:
            o = tf.sigmoid(o_act)

        self.weights.tensor_dict['cs'] = tf.multiply(f, self.weights.tensor_dict['cs']) + tf.multiply(i, c)
        self.weights.tensor_dict['co'] = tf.multiply(o, tf.tanh(self.weights.tensor_dict['cs']))
        return self.weights.tensor_dict['co'], self.weights.tensor_dict['cs']

    def create_var_fp(self, x, initialize, time_index, **kwargs):
        if initialize:
            self.init_cell_tensors(x, **kwargs)

        co = self.weights.tensor_dict['co']
        """
        if self.train_config['batchnorm']['type'] == 'batch':
            bn_idx = min(time_index, self.train_config['batchnorm']['tau'] - 1)
            if 'x' in self.train_config['batchnorm']['modes']:
                if len(self.bn_s_x) == bn_idx:
                    self.bn_s_x.append(get_batchnormalizer())
                x = self.bn_s_x[bn_idx](x, self.is_training)
            if 'h' in self.train_config['batchnorm']['modes'] and bn_idx > 0:
                if len(self.bn_s_h) == bn_idx - 1:
                    self.bn_s_h.append(get_batchnormalizer())
                co = self.bn_s_h[bn_idx - 1](co, self.is_training)
        """
        x = tf.concat([x, co], axis=1)
        i_act = tf.matmul(x, self.weights.var_dict['wi']) + self.weights.var_dict['bi']
        c_act = tf.matmul(x, self.weights.var_dict['wc']) + self.weights.var_dict['bc']
        o_act = tf.matmul(x, self.weights.var_dict['wo']) + self.weights.var_dict['bo']

        """
        if self.train_config['batchnorm']['type'] == 'layer':
            i_act = tf.contrib.layers.layer_norm(i_act)
            c_act = tf.contrib.layers.layer_norm(c_act)
            o_act = tf.contrib.layers.layer_norm(o_act)
        """

        if initialize:
            for act_type, act in zip(['i', 'c', 'o'], [i_act, c_act, o_act]):
                self.acts[act_type] = act
                for neuron_idc in range(len(self.act_neurons)):
                    self.acts[act_type + '_' + str(neuron_idc)] = tf.slice(act, begin=(0, neuron_idc), size=(-1, 1))

        else:
            for act_type, act in zip(['i', 'c', 'o'], [i_act, c_act, o_act]):
                self.acts[act_type] = tf.concat([act, self.acts[act_type]], axis=0)
                for neuron_idc in range(len(self.act_neurons)):
                    self.acts[act_type + '_' + str(neuron_idc)] = \
                        tf.concat([tf.slice(act, begin=(0, neuron_idc), size=(-1, 1)),
                                   self.acts[act_type + '_' + str(neuron_idc)]], axis=0)

        if self.layer_config.i_gate_config.is_act_func_discrete:
            n_act_bins = self.layer_config.i_gate_config.n_act_bins
            i = tf.cast(tf.cast(tf.sigmoid(i_act) * n_act_bins, dtype=tf.int32), dtype=tf.float32) / n_act_bins
            if get_info_config()['cell_access']:
                self.cell_access_mat.append(i)

        else:
            i = tf.sigmoid(i_act)
        f = 1. - i

        if self.layer_config.c_gate_config.is_act_func_discrete:
            n_act_bins = self.layer_config.c_gate_config.n_act_bins
            c = tf.cast(tf.cast(tf.sigmoid(c_act) * n_act_bins, dtype=tf.int32), dtype=tf.float32) * \
                2 / n_act_bins - 1
        else:
            c = tf.tanh(c_act)

        if self.layer_config.o_gate_config.is_act_func_discrete:
            n_act_bins = self.layer_config.o_gate_config.n_act_bins
            o = tf.cast(tf.cast(tf.sigmoid(o_act) * n_act_bins, dtype=tf.int32), dtype=tf.float32) / n_act_bins
        else:
            o = tf.sigmoid(o_act)

        self.weights.tensor_dict['cs'] = tf.multiply(f, self.weights.tensor_dict['cs']) + tf.multiply(i, c)

        if self.layer_config.i_gate_config.is_act_func_discrete:
            self.weights.tensor_dict['co'] = tf.multiply(o, self.weights.tensor_dict['cs'])
        else:
            self.weights.tensor_dict['co'] = tf.multiply(o, tf.tanh(self.weights.tensor_dict['cs']))
        return self.weights.tensor_dict['co'], self.weights.tensor_dict['cs']

    # Initializes cell output and cell state for the first timestep
    def init_cell_tensors(self, x, **kwargs):
        cell_shape = (tf.shape(x)[0], self.b_shape[1])
        if GraphCreationKeys.CELL_INITIALIZATION in kwargs.keys():
            self.weights.tensor_dict['cs'] = kwargs[GraphCreationKeys.CELL_INITIALIZATION]
        else:
            self.weights.tensor_dict['cs'] = tf.zeros(cell_shape)
        self.weights.tensor_dict['co'] = tf.zeros(cell_shape)

