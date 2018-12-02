import tensorflow as tf
from src.fp_tools import approx_activation, transform_tanh_activation, transform_sig_activation, \
    sample_activation
from src.weights import Weights


class LSTMLayer:
    def __init__(self, rnn_config, layer_idx):
        self.rnn_config = rnn_config
        self.layer_config = rnn_config['layer_configs'][layer_idx]
        self.w_shape = (rnn_config['layout'][layer_idx-1] + rnn_config['layout'][layer_idx],
                        rnn_config['layout'][layer_idx])
        self.b_shape = (1, self.w_shape[1])
        self.acts = {'f': None, 'i': None, 'c': None, 'o': None}

        with tf.variable_scope(self.layer_config['var_scope']):
            var_keys = ['wf', 'bf', 'wi', 'bi', 'wc', 'bc', 'wo', 'bo']
            self.weights = Weights(var_keys, self.layer_config, self.w_shape, self.b_shape)
            
            #self.v_loss = tf.nn.l2_loss(self.weights.var_dict['wf_v']) + tf.nn.l2_loss(self.weights.var_dict['bf_v']) + \
            #              tf.nn.l2_loss(self.weights.var_dict['wi_v']) + tf.nn.l2_loss(self.weights.var_dict['bi_v']) + \
            #              tf.nn.l2_loss(self.weights.var_dict['wc_v']) + tf.nn.l2_loss(self.weights.var_dict['bc_v']) + \
            #              tf.nn.l2_loss(self.weights.var_dict['wo_v']) + tf.nn.l2_loss(self.weights.var_dict['bo_v'])
            self.v_loss = 0

    def create_pfp(self, x_m, x_v, mod_layer_config, init):
        if init:
            cell_shape = (tf.shape(x_m)[0], self.b_shape[1])
            self.weights.tensor_dict['cs_m'] = tf.zeros(cell_shape)
            self.weights.tensor_dict['cs_v'] = tf.zeros(cell_shape)
            self.weights.tensor_dict['co_m'] = tf.zeros(cell_shape)
            self.weights.tensor_dict['co_v'] = tf.zeros(cell_shape)

        # Vector concatenation (input with recurrent)
        m = tf.concat([x_m, self.weights.tensor_dict['co_m']], axis=1)
        v = tf.concat([x_v, self.weights.tensor_dict['co_v']], axis=1)

        a_f_m, a_f_v = approx_activation(self.weights.var_dict['wf_m'], self.weights.var_dict['wf_v'], self.weights.var_dict['bf_m'], self.weights.var_dict['bf_v'], m, v)
        f_m, f_v = transform_sig_activation(a_f_m, a_f_v)
        a_i_m, a_i_v = approx_activation(self.weights.var_dict['wi_m'], self.weights.var_dict['wi_v'], self.weights.var_dict['bi_m'], self.weights.var_dict['bi_v'], m, v)
        i_m, i_v = transform_sig_activation(a_i_m, a_i_v)
        a_c_m, a_c_v = approx_activation(self.weights.var_dict['wc_m'], self.weights.var_dict['wc_v'], self.weights.var_dict['bc_m'], self.weights.var_dict['bc_v'], m, v)
        c_m, c_v = transform_tanh_activation(a_c_m, a_c_v)
        a_o_m, a_o_v = approx_activation(self.weights.var_dict['wo_m'], self.weights.var_dict['wo_v'], self.weights.var_dict['bo_m'], self.weights.var_dict['bo_v'], m, v)
        o_m, o_v = transform_sig_activation(a_o_m, a_o_v)

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
    def create_l_sampling_pass(self, x_m, mod_layer_config, init):
        if init:
            cell_shape = (tf.shape(x_m)[0], self.b_shape[1])
            self.weights.tensor_dict['cs_m'] = tf.zeros(cell_shape)
            self.weights.tensor_dict['co_m'] = tf.zeros(cell_shape)

        m = tf.concat([x_m, self.weights.tensor_dict['co_m']], axis=1)

        a_f = sample_activation(self.weights.var_dict['wf_m'], self.weights.var_dict['wf_v'], self.weights.var_dict['bf_m'], self.weights.var_dict['bf_v'], m)
        f = tf.nn.sigmoid(a_f)
        a_i = sample_activation(self.weights.var_dict['wi_m'], self.weights.var_dict['wi_v'], self.weights.var_dict['bi_m'], self.weights.var_dict['bi_v'], m)
        i = tf.nn.sigmoid(a_i)
        a_c = sample_activation(self.weights.var_dict['wc_m'], self.weights.var_dict['wc_v'], self.weights.var_dict['bc_m'], self.weights.var_dict['bc_v'], m)
        c = tf.nn.tanh(a_c)
        a_o = sample_activation(self.weights.var_dict['wo_m'], self.weights.var_dict['wc_v'], self.weights.var_dict['bo_m'], self.weights.var_dict['bo_v'], m)
        o = tf.nn.sigmoid(a_o)

        self.weights.tensor_dict['cs_m'] = tf.multiply(f, self.weights.tensor_dict['cs_m']) + tf.multiply(i, c)
        self.weights.tensor_dict['co_m'] = tf.multiply(tf.tanh(self.weights.tensor_dict['cs_m']), o)
        return self.weights.tensor_dict['co_m']

    # Global reparametrization trick
    def create_g_sampling_pass(self, x, mod_layer_config, init):
        if init:
            self.weights.create_tensor_samples()
            cell_shape = (tf.shape(x)[0], self.b_shape[1])
            self.weights.tensor_dict['cs'] = tf.zeros(cell_shape)
            self.weights.tensor_dict['co'] = tf.zeros(cell_shape)

        x = tf.concat([x, self.weights.tensor_dict['co']], axis=1)
        f = tf.sigmoid(tf.matmul(x, self.weights.tensor_dict['wf']) + self.weights.tensor_dict['bf'])
        i = tf.sigmoid(tf.matmul(x, self.weights.tensor_dict['wi']) + self.weights.tensor_dict['bi'])
        c = tf.tanh(tf.matmul(x, self.weights.tensor_dict['wc']) + self.weights.tensor_dict['bc'])
        o = tf.sigmoid(tf.matmul(x, self.weights.tensor_dict['wo']) + self.weights.tensor_dict['bo'])

        self.weights.tensor_dict['cs'] = tf.multiply(f, self.weights.tensor_dict['cs']) + tf.multiply(i, c)
        self.weights.tensor_dict['co'] = tf.multiply(o, tf.tanh(self.weights.tensor_dict['cs']))
        return self.weights.tensor_dict['co']

    # Classic forward pass. Requires the execution of the sampling operation first.
    def create_fp(self, x, init):
        if init:
            cell_shape = (tf.shape(x)[0], self.b_shape[1])
            self.weights.tensor_dict['cs'] = tf.zeros(cell_shape)
            self.weights.tensor_dict['co'] = tf.zeros(cell_shape)

        x = tf.concat([x, self.weights.tensor_dict['co']], axis=1)

        f_act = tf.matmul(x, self.weights.var_dict['wf']) + self.weights.var_dict['bf']
        i_act = tf.matmul(x, self.weights.var_dict['wi']) + self.weights.var_dict['bi']
        c_act = tf.matmul(x, self.weights.var_dict['wc']) + self.weights.var_dict['bc']
        o_act = tf.matmul(x, self.weights.var_dict['wo']) + self.weights.var_dict['bo']

        if init:
            for act_type, act in zip(['f', 'i', 'c', 'o'], [f_act, i_act, c_act, o_act]):
                self.acts[act_type] = act
        else:
            for act_type, act in zip(['f', 'i', 'c', 'o'], [f_act, i_act, c_act, o_act]):
                self.acts[act_type] = tf.concat([act, self.acts[act_type]], 0)

        f = tf.sigmoid(f_act)
        i = tf.sigmoid(i_act)
        c = tf.tanh(c_act)
        o = tf.sigmoid(o_act)

        self.weights.tensor_dict['cs'] = tf.multiply(f, self.weights.tensor_dict['cs']) + tf.multiply(i, c)
        self.weights.tensor_dict['co'] = tf.multiply(o, tf.tanh(self.weights.tensor_dict['cs']))
        return self.weights.tensor_dict['co']

