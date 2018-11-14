import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pprint


def get_mean_initializer(w_config, shape):
    if w_config['init_m'] == 'xavier':
        init_vals = np.random.randn(shape[0], shape[1]) * np.sqrt(2/sum(shape)) + w_config['prior_m']
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
    return tf.constant_initializer(init_vals)


def process_results(result_config, result_dicts):
    tr_accs, tr_losses, tr_epochs = convert_to_array(result_dicts, 'tr')
    va_accs, va_losses, va_epochs = convert_to_array(result_dicts, 'va')

    if result_config['save_results']:
        np.save(result_config['filename'] + '_tr_accs', tr_accs)
        np.save(result_config['filename'] + '_tr_losses', tr_losses)
        np.save(result_config['filename'] + '_tr_epochs', tr_epochs)
        np.save(result_config['filename'] + '_va_accs', va_accs)
        np.save(result_config['filename'] + '_va_losses', va_losses)
        np.save(result_config['filename'] + '_va_epochs', va_epochs)

    if result_config['plot_results']:
        plt.plot(tr_epochs, np.mean(tr_accs, axis=0))
        plt.plot(va_epochs, np.mean(va_accs, axis=0))
        plt.legend(['training', 'validation'])
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.show()

    if result_config['print_final_stats']:
        print_stats('Final Tr Acc', tr_accs[:, -1])
        print_stats('Final Va Acc', va_accs[:, -1])
        print_stats('Final Tr Loss', tr_losses[:, -1])
        print_stats('Final Va Loss', va_losses[:, -1])


def convert_to_array(result_dicts, dict_key):
    accs = []
    losses = []
    for run, result_dict in enumerate(result_dicts):
        accs.append(np.expand_dims(np.asarray(result_dict[dict_key]['accs'], np.float32), axis=0))
        losses.append(np.expand_dims(np.asarray(result_dict[dict_key]['loss'], np.float32), axis=0))
    accs = np.concatenate(accs)
    losses = np.concatenate(losses)
    epochs = np.asarray(result_dicts[0][dict_key]['epochs'])
    return accs, losses, epochs


def print_stats(name, values):
    print('{:15s}: {:9.5f} +- {:7.5f}'.format(name, np.mean(values), np.std(values, ddof=1)))


def print_config(rnn_config, training_config, data_config):
    print('\n=============================\nCONFIG FILE')
    print('\nRNN CONFIG')
    pprint.pprint(rnn_config)
    print('\nDATA CONFIG')
    pprint.pprint(data_config)
    print('\nTRAINING CONFIG')
    pprint.pprint(training_config)
    print('==============================\n\n')
