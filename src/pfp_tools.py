import tensorflow as tf
import math


# Returns statistics of the activations of a layer
def approx_activation(w_m, w_v, b_m, b_v, x_m, x_v):
    mean = tf.matmul(x_m, w_m) + b_m
    x_sec_moment = x_v + tf.square(x_m)
    variance = tf.matmul(x_sec_moment, w_v) + tf.matmul(x_v, tf.square(w_m)) + b_v
    return mean, variance


# Used article: arxiv 1703.00091
def transform_sig_activation(a_mu, a_var):
    pi_factor = 3/(math.pi ** 2)
    mu = sigmoid_div(a_mu, a_var, pi_factor)
    var = tf.multiply(tf.multiply(sigmoid_div(a_mu, a_var, pi_factor), 1 - sigmoid_div(a_mu, a_var, pi_factor)),
                      1 - tf.divide(1, tf.sqrt(1 + pi_factor * a_var)))
    return mu, var


def transform_tanh_activation(a_mu, a_var):
    mu, var = transform_sig_activation(2 * a_mu, 2 * a_var)
    mu = 2 * mu - 1
    var *= 2
    return mu, var


def sigmoid_div(mu, var, var_factor):
    return tf.sigmoid(tf.divide(mu, tf.sqrt(1 + var_factor * var)))


def get_kl_loss(w_config, m, v):
    return tf.reduce_sum(0.5 * tf.log(tf.divide(w_config['prior_v'], v)) +
                         tf.divide(v + tf.square(m - w_config['prior_m']), 2 * w_config['prior_v']) - 0.5)



