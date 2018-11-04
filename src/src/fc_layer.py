import tensorflow as tf
import math
from src.pfp_tools import approx_activation
import numpy as np

# TODO: Currently FCLayer can only be used as a regression-output layer
class FCLayer:
    def __init__(self, fc_shape, var_scope, gamma):
        weight_shape = (fc_shape[0], fc_shape[1])
        bias_shape = (1, fc_shape[1])
        with tf.variable_scope(var_scope):
            w_mu_init = tf.random_normal_initializer(mean=0.0, stddev=np.sqrt(2/(weight_shape[0] + weight_shape[1])))
            w_var_init = tf.constant_initializer(np.ones(weight_shape)/gamma)

            b_zero_init = tf.constant_initializer(np.zeros(bias_shape))
            b_var_init = tf.constant_initializer(np.ones(bias_shape)/gamma)

            self.W_mu = tf.get_variable('W_mu', shape=weight_shape, initializer=w_mu_init)
            self.W_var = tf.get_variable('W_var', shape=weight_shape, initializer=w_var_init)
            self.b_mu = tf.get_variable('b_mu', shape=bias_shape, initializer=b_zero_init)
            self.b_var = tf.get_variable('b_var', shape=bias_shape, initializer=b_var_init)

    def forward_pass(self, x_mu, x_var, y, beta):
        a_mu, a_var = approx_activation(self.W_mu, self.W_var, self.b_mu, self.b_var, x_mu, x_var)
        exp_log_likelihood = - tf.log(2 * math.pi * beta) / 2 - tf.divide(tf.square(y - a_mu), 2*beta) - x_var/(2*beta)
        return exp_log_likelihood