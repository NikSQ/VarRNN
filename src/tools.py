import numpy as np
import tensorflow as tf
import pprint


def print_config(rnn_config, training_config, data_config):
    print('\n=============================\nCONFIG FILE')
    print('\nTRAINING CONFIG')
    pprint.pprint(training_config)
    print('\nDATA CONFIG')
    pprint.pprint(data_config)
    print('\nRNN CONFIG')
    pprint.pprint(rnn_config)
    print('==============================\n\n')


momentum = None

def set_momentum(value):
    global momentum
    momentum = value


def get_batchnormalizer():
    gamma_init = tf.constant_initializer(value=.1)
    return tf.keras.layers.BatchNormalization(center=False, gamma_initializer=gamma_init, momentum=momentum)

