import tensorflow as tf
import numpy as np
from src.fp_tools import approx_activation


class ActivationLogic:
    def __init__(self, layer_config, weights, batchnorm):
        self.layer_config = layer_config
        self.weights = weights
        self.batchnorm = batchnorm

    # Used by local reparametrization for both sampling continuous and discrete activations
    # Supports continuous, binary and ternary weights
    # If act_func == None: Returns sample of activation
    #                Else: Returns sample of discretized tanh or sig
    def sample_activation(self, w_var_key, b_var_key, x_m, h_m, act_func, init):
        if self.layer_config['lr_adapt'] == False or init == True:
            w_m, w_v = self.weights.get_stats(w_var_key)
            b_m, b_v = self.weights.get_stats(b_var_key)
        else:
            w_m, w_v = self.weights.get_adapted_stats(w_var_key)
            b_m, b_v = self.weights.get_adapted_stats(w_var_key)

        if self.batchnorm:
            if self.layer_config['lr_adapt']:
                raise Exception('lr adaption not implemented for batchnorm')
            if self.layer_config['layer_type'] == 'fc':
                mean = tf.matmul(x_m, w_m)
                var = tf.matmul(tf.square(x_m), w_v)

                # mean_mean is the mean of the mean of the activations over the batch
                # mean_var is the variance of the mean of the activations over the batch
                mean_mean, mean_var = tf.nn.moments(mean, axes=0)
                mean = b_m + tf.multiply(self.weights.var_dict[w_var_key + '_gamma'],
                                         tf.divide(mean - mean_mean, tf.sqrt(mean_var + 0.0001)))
                std = tf.sqrt(tf.multiply(tf.square(self.weights.var_dict[w_var_key + '_gamma']),
                                          tf.divide(var, mean_var + 0.0001)) + b_v)
            else:
                mean_x = tf.matmul(x_m, tf.slice(w_m, begin=(0, 0),
                                                 size=(self.weights.w_shape[0] - self.weights.w_shape[1], -1)))
                var_x = tf.matmul(tf.square(x_m), tf.slice(w_v, begin=(0, 0),
                                                           size=(self.weights.w_shape[0] - self.weights.w_shape[1],
                                                                 -1)))
                mean_x_mean, mean_x_var = tf.nn.moments(mean_x, axes=0)

                mean_h = tf.matmul(h_m, tf.slice(w_m, begin=(self.weights.w_shape[0] - self.weights.w_shape[1],0),
                                                 size=(-1, -1)))
                var_h = tf.matmul(tf.square(h_m), tf.slice(w_v,
                                                           begin=(self.weights.w_shape[0] - self.weights.w_shape[1], 0),
                                                           size=(-1, -1)))
                mean_h_mean, mean_h_var = tf.nn.moments(mean_h, axes=0)

                mean = b_m + tf.multiply(self.weights.var_dict[w_var_key + '_gamma'],
                                         tf.divide(mean_x - mean_x_mean, tf.sqrt(mean_x_var + 0.0001))) + \
                       tf.multiply(self.weights.var_dict[w_var_key + '_gamma2'],
                                         tf.divide(mean_h - mean_h_mean, tf.sqrt(mean_h_var + 0.0001)))
                std = tf.sqrt(tf.multiply(tf.square(self.weights.var_dict[w_var_key + '_gamma']),
                                          tf.divide(var_x, mean_x_var + 0.0001)) +
                              tf.multiply(tf.square(self.weights.var_dict[w_var_key + '_gamma2']),
                                          tf.divide(var_h, mean_h_var + 0.0001)) + b_v)
        else:
            if self.layer_config['lr_adapt'] is False:
                mean = tf.matmul(x_m, w_m) + b_m
                std = tf.sqrt(tf.matmul(tf.square(x_m), w_v) + b_v)
            else:
                layer_inputs = tf.unstack(tf.expand_dims(x_m, axis=1), axis=0)
                means = []
                stds = []
                if init:
                    w_m = [w_m] * len(layer_inputs)
                    w_v = [w_v] * len(layer_inputs)
                    b_m = [b_m] * len(layer_inputs)
                    b_v = [b_v] * len(layer_inputs)
                    self.weights.create_adapted_weights(w_var_key, w_m, w_v)
                    self.weights.create_adapted_weights(b_var_key, b_m, b_v)
                for sample_w_m, sample_w_v, sample_b_m, sample_b_v, layer_input in zip(w_m, w_v, b_m, b_v, layer_inputs):
                    means.append(tf.matmul(layer_input, sample_w_m) + sample_b_m)
                    stds.append(tf.sqrt(tf.matmul(tf.square(layer_input), sample_w_v) + sample_b_v))
                mean = tf.squeeze(tf.stack(means, axis=0))
                std = tf.squeeze(tf.stack(stds, axis=0))

        shape = (tf.shape(x_m)[0], tf.shape(b_m)[1])

        if act_func is None:
            a = mean + tf.multiply(self.weights.gauss.sample(sample_shape=shape), std)
            if self.layer_config['lr_adapt']:
               self.weights.adapt_weights(x_m, w_var_key, b_var_key, a)
            return a
        else:
            if self.layer_config['lr_adapt']:
                raise Exception('not implemented for activation function sampling')
            prob_1 = 0.5 + 0.5 * tf.erf(tf.divide(mean, std * np.sqrt(2)))
            prob_1 = 0.0001 + (0.9999 - 0.0001) * prob_1
            output = tf.nn.tanh((tf.log(prob_1) - tf.log(1. - prob_1)
                                 - tf.log(-tf.log(self.weights.uniform.sample(shape)))
                                 + tf.log(-tf.log(self.weights.uniform.sample(shape))))
                                / (self.layer_config['tau'] * 2))
            if act_func == 'tanh':
                return output
            elif act_func == 'sig':
                return (output + 1.) / 2.
            else:
                raise Exception('activation function not understood')

    # Performs a batchnorm transformation for deterministic variables
    def batchnorm_transform(self, x, h, var_key, weight_dict):
        epsilon = .0001
        if self.layer_config['layer_type'] == 'fc':
            pre_acts = tf.matmul(x, weight_dict[var_key])
            mean, var = tf.nn.moments(pre_acts, axes=0)
            return tf.multiply(self.weights.var_dict[var_key + '_gamma'],
                                   tf.divide(pre_acts - mean, tf.sqrt(var + epsilon)))
        else:
            pre_acts_x = tf.matmul(x, tf.slice(weight_dict[var_key], begin=(0,0),
                                               size=(self.weights.w_shape[0] - self.weights.w_shape[1], -1)))
            mean_x, var_x = tf.nn.moments(pre_acts_x, axes=0)
            pre_acts_h = tf.matmul(h, tf.slice(weight_dict[var_key],
                                               begin=(self.weights.w_shape[0] - self.weights.w_shape[1], 0),
                                               size=(-1, -1)))
            mean_h, var_h = tf.nn.moments(pre_acts_h, axes=0)
            return tf.multiply(self.weights.var_dict[var_key + '_gamma'],
                               tf.divide(pre_acts_x - mean_x, tf.sqrt(var_x + epsilon))) + \
                   tf.multiply(self.weights.var_dict[var_key + '_gamma2'],
                               tf.divide(pre_acts_h - mean_h, tf.sqrt(var_h + epsilon)))




