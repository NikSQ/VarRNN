import tensorflow as tf
from src.pfp_tools import approx_activation, transform_sig_activation, transform_tanh_activation
import numpy as np


class LSTMCell:
    def __init__(self, lstm_shape, var_scope, gamma):
        self.weight_shape = (lstm_shape[1] + lstm_shape[0],  lstm_shape[1])
        self.output_shape = (1, lstm_shape[1])
        self.var_scope = var_scope

        # TODO: Ask about weight initialization
        with tf.variable_scope(var_scope):
            w_zero_init = tf.constant_initializer(np.zeros(self.weight_shape))
            w_mu_init = tf.random_normal_initializer(mean=0.0,
                                                     stddev=np.sqrt(2/(self.weight_shape[0] + self.weight_shape[1])))
            w_var_init = tf.constant_initializer(np.ones(self.weight_shape)/gamma)

            b_zero_init = tf.constant_initializer(np.zeros(self.output_shape))
            b_one_init = tf.constant_initializer(np.ones(self.output_shape))
            b_var_init = tf.constant_initializer(np.ones(self.output_shape)/gamma)

            self.Wf_mu = tf.get_variable('Wf_mu', shape=self.weight_shape, initializer=w_zero_init)
            self.Wf_var = tf.get_variable('Wf_var', shape=self.weight_shape, initializer=w_zero_init)
            self.bf_mu = tf.get_variable('bf_mu', shape=self.output_shape, initializer=b_one_init)
            self.bf_var = tf.get_variable('bf_var', shape=self.output_shape, initializer=b_one_init)

            self.Wi_mu = tf.get_variable('Wi_mu', shape=self.weight_shape, initializer=w_zero_init)
            self.Wi_var = tf.get_variable('Wi_var', shape=self.weight_shape, initializer=w_zero_init)
            self.bi_mu = tf.get_variable('bi_mu', shape=self.output_shape, initializer=b_zero_init)
            self.bi_var = tf.get_variable('bi_var', shape=self.output_shape, initializer=b_zero_init)

            self.Wc_mu = tf.get_variable('Wc_mu', shape=self.weight_shape, initializer=w_mu_init)
            self.Wc_var = tf.get_variable('Wc_var', shape=self.weight_shape, initializer=w_var_init)
            self.bc_mu = tf.get_variable('bc_mu', shape=self.output_shape, initializer=b_zero_init)
            self.bc_var = tf.get_variable('bc_var', shape=self.output_shape, initializer=b_var_init)

            self.Wo_mu = tf.get_variable('Wo_mu', shape=self.weight_shape, initializer=w_mu_init)
            self.Wo_var = tf.get_variable('Wo_var', shape=self.weight_shape, initializer=w_var_init)
            self.bo_mu = tf.get_variable('bo_mu', shape=self.output_shape, initializer=b_zero_init)
            self.bo_var = tf.get_variable('bo_var', shape=self.output_shape, initializer=b_var_init)

            self.cell_state_mu = None
            self.cell_state_var = None
            self.cell_output_mu = None
            self.cell_output_var = None

    def forward_pass(self, x_mu, x_var, init_cell_state):
        # Initialization of cell state
        if init_cell_state:
            shape = tf.shape(x_mu)[0], self.output_shape[1]
            self.cell_state_mu = tf.zeros(shape)
            self.cell_state_var = tf.zeros(shape)
            self.cell_output_mu = tf.zeros(shape)
            self.cell_output_var = tf.zeros(shape)

        # Vector concatenation
        mu = tf.concat([x_mu, self.cell_state_mu], axis=1)
        var = tf.concat([x_var, self.cell_state_var], axis=1)

        # Calculations of activations
        a_f_mu, a_f_var = approx_activation(self.Wf_mu, self.Wf_var, self.bf_mu, self.bf_var, mu, var)
        f_mu, f_var = transform_sig_activation(a_f_mu, a_f_var)
        a_i_mu, a_i_var = approx_activation(self.Wi_mu, self.Wi_var, self.bi_mu, self.bi_var, mu, var)
        i_mu, i_var = transform_sig_activation(a_i_mu, a_i_var)
        a_c_mu, a_c_var = approx_activation(self.Wc_mu, self.Wc_var, self.bc_mu, self.bc_var, mu, var)
        c_mu, c_var = transform_tanh_activation(a_c_mu, a_c_var)

        # Cell state update
        f_2nd_mom = tf.square(f_mu) + f_var
        i_2nd_mom = tf.square(i_mu) + i_var
        self.cell_state_var = tf.multiply(self.cell_state_var, f_2nd_mom) + tf.multiply(c_var, i_2nd_mom) + \
                              tf.multiply(tf.square(self.cell_state_mu), f_var) + tf.multiply(tf.square(c_mu), i_var)
        self.cell_state_mu = tf.multiply(f_mu, self.cell_state_mu) + tf.multiply(i_mu, c_mu)

        # Cell output update
        a_o_mu, a_o_var = approx_activation(self.Wo_mu, self.Wo_var, self.bo_mu, self.bo_var, mu, var)
        o_mu, o_var = transform_sig_activation(a_o_mu, a_o_var)
        c_tan_mu, c_tan_var = transform_tanh_activation(self.cell_state_mu, self.cell_state_var)
        o_2nd_mom = tf.square(o_mu) + o_var
        self.cell_output_mu = tf.multiply(c_tan_mu, o_mu)
        self.cell_output_var = tf.multiply(c_tan_var, o_2nd_mom) + tf.multiply(tf.square(c_tan_mu), o_var)

        return self.cell_output_mu, self.cell_output_var








