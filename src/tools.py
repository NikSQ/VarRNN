import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pprint


def get_mean_initializer(w_config, shape):
    if w_config['init_m'] == 'xavier':
        init_vals = np.random.randn(shape[0], shape[1]) * np.sqrt(2/sum(shape))
    elif w_config['init_m'] == 'same':
        init_vals = np.ones(shape) * w_config['prior_m']
    else:
        raise Exception("{} is not a valid weight initialization".format(w_config['init_m']))
    return tf.constant_initializer(init_vals)


def get_var_initializer(w_config, shape):
    if w_config['init_v'] == 'xavier':
        w_config['prior_v'] = np.sqrt(2/sum(shape))
        init_vals = np.ones(shape) * w_config['prior_v']
    else:
        init_vals = np.ones(shape) * w_config['init_v']
        #init_vals = np.ones(shape) * 0.0001
    return tf.constant_initializer(init_vals)


def get_xavier_initializer(shape):
    init_vals = np.random.randn(shape[0], shape[1]) * np.sqrt(2/sum(shape))
    return tf.constant_initializer(init_vals)


def print_config(rnn_config, training_config, data_config):
    print('\n=============================\nCONFIG FILE')
    print('\nRNN CONFIG')
    pprint.pprint(rnn_config)
    print('\nDATA CONFIG')
    pprint.pprint(data_config)
    print('\nTRAINING CONFIG')
    pprint.pprint(training_config)
    print('==============================\n\n')
