import tensorflow as tf
from src.fp_tools import approx_activation, transform_tanh_activation, transform_sig_activation, get_kl_loss, \
    sample_activation
from src.tools import get_mean_initializer, get_var_initializer, get_xavier_initializer


class LSTMLayer:
    def __init__(self, rnn_config, layer_idx):
        self.rnn_config = rnn_config
        self.layer_config = rnn_config['layer_configs'][layer_idx]
        self.w_shape = (rnn_config['layout'][layer_idx-1] + rnn_config['layout'][layer_idx],
                        rnn_config['layout'][layer_idx])
        self.b_shape = (1, self.w_shape[1])

        with tf.variable_scope(self.layer_config['var_scope']):
            self.wf_m = tf.get_variable(name='wf_m', shape=self.w_shape,
                                        initializer=get_mean_initializer(self.layer_config['wf'], self.w_shape))
            self.wf_v = tf.exp(tf.get_variable(name='wf_v', shape=self.w_shape,
                                               initializer=get_var_initializer(self.layer_config['wf'], self.w_shape)))
            self.bf_m = tf.get_variable(name='bf_m', shape=self.b_shape,
                                        initializer=get_mean_initializer(self.layer_config['bf'], self.b_shape))
            self.bf_v = tf.exp(tf.get_variable(name='bf_v', shape=self.b_shape,
                                               initializer=get_var_initializer(self.layer_config['bf'], self.b_shape)))

            self.wi_m = tf.get_variable(name='wi_m', shape=self.w_shape,
                                        initializer=get_mean_initializer(self.layer_config['wi'], self.w_shape))
            self.wi_v = tf.exp(tf.get_variable(name='wi_v', shape=self.w_shape,
                                               initializer=get_var_initializer(self.layer_config['wi'], self.w_shape)))
            self.bi_m = tf.get_variable(name='bi_m', shape=self.b_shape,
                                        initializer=get_mean_initializer(self.layer_config['bi'], self.b_shape))
            self.bi_v = tf.exp(tf.get_variable(name='bi_v', shape=self.b_shape,
                                               initializer=get_var_initializer(self.layer_config['bi'], self.b_shape)))

            self.wc_m = tf.get_variable(name='wc_m', shape=self.w_shape,
                                        initializer=get_mean_initializer(self.layer_config['wc'], self.w_shape))
            self.wc_v = tf.exp(tf.get_variable(name='wc_v', shape=self.w_shape,
                                               initializer=get_var_initializer(self.layer_config['wc'], self.w_shape)))
            self.bc_m = tf.get_variable(name='bc_m', shape=self.b_shape,
                                        initializer=get_mean_initializer(self.layer_config['bc'], self.b_shape))
            self.bc_v = tf.exp(tf.get_variable(name='bc_v', shape=self.b_shape,
                                               initializer=get_var_initializer(self.layer_config['bc'], self.b_shape)))

            self.wo_m = tf.get_variable(name='wo_m', shape=self.w_shape,
                                        initializer=get_mean_initializer(self.layer_config['wo'], self.w_shape))
            self.wo_v = tf.exp(tf.get_variable(name='wo_v', shape=self.w_shape,
                                               initializer=get_var_initializer(self.layer_config['wo'], self.w_shape)))
            self.bo_m = tf.get_variable(name='bo_m', shape=self.b_shape,
                                        initializer=get_mean_initializer(self.layer_config['bo'], self.b_shape))
            self.bo_v = tf.exp(tf.get_variable(name='bo_v', shape=self.b_shape,
                                               initializer=get_var_initializer(self.layer_config['bo'], self.b_shape)))

            self.wf = tf.get_variable(name='wf', shape=self.w_shape, initializer=get_xavier_initializer(self.w_shape))
            self.bf = tf.get_variable(name='bf', shape=self.b_shape,
                                      initializer=tf.constant_initializer(value=1., dtype=tf.float32))
            self.wi = tf.get_variable(name='wi', shape=self.w_shape, initializer=get_xavier_initializer(self.w_shape))
            self.bi = tf.get_variable(name='bi', shape=self.b_shape, initializer=tf.zeros_initializer(dtype=tf.float32))
            self.wc = tf.get_variable(name='wc', shape=self.w_shape, initializer=get_xavier_initializer(self.w_shape))
            self.bc = tf.get_variable(name='bc', shape=self.b_shape, initializer=tf.zeros_initializer(dtype=tf.float32))
            self.wo = tf.get_variable(name='wo', shape=self.w_shape, initializer=get_xavier_initializer(self.w_shape))
            self.bo = tf.get_variable(name='bo', shape=self.b_shape, initializer=tf.zeros_initializer(dtype=tf.float32))

            gauss = tf.distributions.Normal(loc=0., scale=1.)
            wf_sample_op = tf.assign(self.wf, self.wf_m + gauss.sample(self.w_shape) * self.wf_v)
            bf_sample_op = tf.assign(self.bf, self.bf_m + gauss.sample(self.b_shape) * self.bf_v)
            wi_sample_op = tf.assign(self.wi, self.wi_m + gauss.sample(self.w_shape) * self.wi_v)
            bi_sample_op = tf.assign(self.bi, self.bi_m + gauss.sample(self.b_shape) * self.bi_v)
            wc_sample_op = tf.assign(self.wc, self.wc_m + gauss.sample(self.w_shape) * self.wc_v)
            bc_sample_op = tf.assign(self.bc, self.bc_m + gauss.sample(self.b_shape) * self.bc_v)
            wo_sample_op = tf.assign(self.wo, self.wo_m + gauss.sample(self.w_shape) * self.wo_v)
            bo_sample_op = tf.assign(self.bo, self.bo_m + gauss.sample(self.b_shape) * self.bo_v)
            self.sample_op = tf.group(*[wf_sample_op, bf_sample_op, wi_sample_op, bi_sample_op,
                                        wc_sample_op, bc_sample_op, wo_sample_op, bo_sample_op])

            summary_ops = list()
            summary_ops.append(tf.summary.histogram('wf_m', self.wf_m))
            summary_ops.append(tf.summary.histogram('wf_v', self.wf_v))
            summary_ops.append(tf.summary.histogram('bf_m', self.bf_m))
            summary_ops.append(tf.summary.histogram('bf_v', self.bf_v))

            summary_ops.append(tf.summary.histogram('wi_m', self.wi_m))
            summary_ops.append(tf.summary.histogram('wi_v', self.wi_v))
            summary_ops.append(tf.summary.histogram('bi_m', self.bi_m))
            summary_ops.append(tf.summary.histogram('bi_v', self.bi_v))

            summary_ops.append(tf.summary.histogram('wc_m', self.wc_m))
            summary_ops.append(tf.summary.histogram('wc_v', self.wc_v))
            summary_ops.append(tf.summary.histogram('bc_m', self.bc_m))
            summary_ops.append(tf.summary.histogram('bc_v', self.bc_v))

            summary_ops.append(tf.summary.histogram('wo_m', self.wo_m))
            summary_ops.append(tf.summary.histogram('wo_v', self.wo_v))
            summary_ops.append(tf.summary.histogram('bo_m', self.bo_m))
            summary_ops.append(tf.summary.histogram('bo_v', self.bo_v))
            self.weight_summaries = tf.summary.merge(summary_ops)

            self.cell_state_m = None
            self.cell_state_v = None
            self.cell_output_m = None
            self.cell_output_v = None
            self.cell_state = None
            self.cell_output = None

            self.kl = get_kl_loss(self.layer_config['wf'], self.wf_m, self.wf_v) + \
                get_kl_loss(self.layer_config['bf'], self.bf_m, self.bf_v) + \
                get_kl_loss(self.layer_config['wi'], self.wi_m, self.wi_v) + \
                get_kl_loss(self.layer_config['bi'], self.bi_m, self.bi_v) + \
                get_kl_loss(self.layer_config['wc'], self.wc_m, self.wc_v) + \
                get_kl_loss(self.layer_config['bc'], self.bc_m, self.bc_v) + \
                get_kl_loss(self.layer_config['wo'], self.wo_m, self.wo_v) + \
                get_kl_loss(self.layer_config['bo'], self.bo_m, self.bo_v)

    def create_pfp(self, x_m, x_v, mod_layer_config, init_cell_state):
        if init_cell_state:
            cell_shape = (tf.shape(x_m)[0], self.b_shape[1])
            self.cell_state_m = tf.zeros(cell_shape)
            self.cell_state_v = tf.zeros(cell_shape)
            self.cell_output_m = tf.zeros(cell_shape)
            self.cell_output_v = tf.zeros(cell_shape)

        # Vector concatenation (input with recurrent)
        m = tf.concat([x_m, self.cell_output_m], axis=1)
        v = tf.concat([x_v, self.cell_output_v], axis=1)

        a_f_m, a_f_v = approx_activation(self.wf_m, self.wf_v, self.bf_m, self.bf_v, m, v)
        f_m, f_v = transform_sig_activation(a_f_m, a_f_v)
        a_i_m, a_i_v = approx_activation(self.wi_m, self.wi_v, self.bi_m, self.bi_v, m, v)
        i_m, i_v = transform_sig_activation(a_i_m, a_i_v)
        a_c_m, a_c_v = approx_activation(self.wc_m, self.wc_v, self.bc_m, self.bc_v, m, v)
        c_m, c_v = transform_tanh_activation(a_c_m, a_c_v)

        f_2nd_mom = tf.square(f_m) + f_v
        i_2nd_mom = tf.square(i_m) + i_v
        self.cell_state_v = tf.multiply(self.cell_state_v, f_2nd_mom) + tf.multiply(c_v, i_2nd_mom) + \
                            tf.multiply(tf.square(self.cell_state_m), f_v) + tf.multiply(tf.square(c_m), i_v)
        self.cell_state_m = tf.multiply(f_m, self.cell_state_m) + tf.multiply(i_m, c_m)

        a_o_m, a_o_v = approx_activation(self.wo_m, self.wo_v, self.bo_m, self.bo_v, m, v)
        o_m, o_v = transform_sig_activation(a_o_m, a_o_v)
        c_tan_m, c_tan_v = transform_tanh_activation(self.cell_state_m, self.cell_state_v)
        o_2nd_mom = tf.square(o_m) + o_v
        self.cell_output_m = tf.multiply(c_tan_m, o_m)
        self.cell_output_v = tf.multiply(c_tan_v, o_2nd_mom) + tf.multiply(tf.square(c_tan_m), o_v)

        return self.cell_output_m, self.cell_output_v

    def create_sampling_pass(self, x_m, mod_layer_config, init_cell_state):
        if init_cell_state:
            cell_shape = (tf.shape(x_m)[0], self.b_shape[1])
            self.cell_state_m = tf.zeros(cell_shape)
            self.cell_output_m = tf.zeros(cell_shape)

        m = tf.concat([x_m, self.cell_output_m], axis=1)

        a_f = sample_activation(self.wf_m, self.wf_v, self.bf_m, self.bf_v, m)
        f = tf.nn.sigmoid(a_f)
        a_i = sample_activation(self.wi_m, self.wi_v, self.bi_m, self.bi_v, m)
        i = tf.nn.sigmoid(a_i)
        a_c = sample_activation(self.wc_m, self.wc_v, self.bc_m, self.bc_v, m)
        c = tf.nn.tanh(a_c)
        a_o = sample_activation(self.wo_m, self.wc_v, self.bo_m, self.bo_v, m)
        o = tf.nn.sigmoid(a_o)

        self.cell_state_m = tf.multiply(f, self.cell_state_m) + tf.multiply(i, c)
        self.cell_output_m = tf.multiply(tf.tanh(c), o)
        return self.cell_output_m

    def create_fp(self, x, init_cell_state):
        if init_cell_state:
            cell_shape = (tf.shape(x)[0], self.b_shape[1])
            self.cell_state = tf.zeros(cell_shape)
            self.cell_output = tf.zeros(cell_shape)

        x = tf.concat([x, self.cell_output], axis=1)
        f = tf.sigmoid(tf.matmul(x, self.wf) + self.bf)
        i = tf.sigmoid(tf.matmul(x, self.wi) + self.bi)
        o = tf.sigmoid(tf.matmul(x, self.wo) + self.bo)
        c = tf.tanh(tf.matmul(x, self.wc) + self.bc)

        self.cell_state = tf.multiply(f, self.cell_state) + tf.multiply(i, c)
        self.cell_output = tf.multiply(o, tf.tanh(self.cell_state))
        return self.cell_output



