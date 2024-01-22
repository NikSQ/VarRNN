import tensorflow as tf
import numpy as npü

from src.global_variable import get_rnn_config, get_train_config

from src.network_api.ff_layer import FFLayer
from src.network_api.lstm_layer import LSTMLayer
from src.configuration.constants import NetworkC

# NOTE Did not include varscope for sampling and bayesian pass

class RNN:
    def __init__(self, datasets):
        self.datasets = datasets

        with tf.variable_scope("global"):
            self.learning_rate = tf.get_variable(name="learning rate", shape=(1,), dtype=tf.float32)
            self.tau = tf.get_variable(name="tau", shape=(1,), dtype=tf.float32)
            self.is_training = tf.get_variable(name="is training", shape=(1,), dtype=tf.bool)

        self.nn_config = get_rnn_config()
        self.train_config = get_train_config()

        weight_summaries = []
        sample_ops = []
        c_arm_sample_ops = []
        init_ops = []
        self.layers = []
        for layer_idx, layer_config in enumerate(self.nn_config.layer_configs):
            if layer_config.layer_type == NetworkC.FEED_FORWARD_LAYER:
                layer = FFLayer(layer_idx=layer_idx,
                                is_training=self.is_training,
                                tau=self.is_tau)

            elif layer_config.layer_type == NetworkC.LSTM_LAYER:
                layer = LSTMLayer(layer_idx=layer_idx,
                                  is_training=self.is_training,
                                  tau=self.is_tau)

            else:
                raise Exception(f"Layer type {layer_config.layer_type} not understood")

            weight_summaries.append(layer.weights.weight_summaries)
            sample_ops.append(layer.weights.sample_op)
            c_arm_sample_ops.append(layer.weights.c_arm_sample_op)
            init_ops.append(layer.weights.init_op)
            self.layers.append(layer)

        self.sample_op = tf.group(*sample_ops)
        self.c_arm_sample_op = tf.group(*c_arm_sample_ops)
        self.init_op = tf.group(*init_ops)
        self.weight_summaries = tf.summary.merge(weight_summaries)
        self.get_weights_op = self.get_weights_op()

    # TODO: MAke predictions based on predictive distribution aswell
    def create_rnn_graph(self, data_key, mod_nn_config, bayesian=True):
        x, y, seqlen = self.dataset_iters[data_key].next()


    # Creates a Bayesian graph for training in probabilistic manner
    def create_bayesian_training_graph(self, key):
        pass
