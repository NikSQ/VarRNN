import sys
import os
import tensorflow as tf
import numpy as np
import copy

sys.path.append('../')

from src.experiment import Experiment
from src.data.t_metrics import save_to_file


try:
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
except KeyError:
    print('NO SLURM TASK ID FOUND')
    print('Using ID 0 instead')
    task_id = 0

runs = 1
filename = 'to_ignore'
tau = 0.5

timit_dataset = ['timit_tr_small_0', 'timit_va_small_0']
timit_s_dataset = ['timit_tr_s_0', 'timit_va_s_0']
timit_l_dataset = ['timit_tr_l_0', 'timit_va_l_0']
penstroke_dataset = 'red_penstroke'
# timit: 13 -> 54
# penstroke: 4 -> 10
labelled_data_config = {'dataset': penstroke_dataset,
                        'tr': {'in_seq_len': 30,
                               'out_seq_len': 5,
                               'zero_padding': 20,
                               'minibatch_enabled': True,
                               'minibatch_size': 200},
                        'va': {'in_seq_len': 30,
                               'out_seq_len': 5,
                               'zero_padding': 20,
                               'minibatch_enabled': True,
                               'minibatch_size': 200}}

priors = [[0.2, 0.6, 0.2],[0.1, 0.8, 0.1]]
input_config = {'layer_type': 'input'}
b_config = {'init_m': 'xavier', 'prior_m': 0., 'init_v': -4.5, 'prior_v': 0., 'type': 'continuous'}
w_config = {'priors': priors[0], 'type': 'ternary', 'pmin': .01, 'pmax':.99, 'p0min': .05, 'p0max': .95}

hidden_2_config = {'layer_type': 'lstm',
                   'var_scope': 'lstm_1',
                   'tau': tau,
                   'discrete_act': False,
                   'init_config': {'f': {'w': 'all_zero', 'b': 'all_one'},
                                   'i': {'w': 'xavier', 'b': 'all_zero'},
                                   'c': {'w': 'xavier', 'b': 'all_zero'},
                                   'o': {'w': 'xavier', 'b': 'all_zero'}},
                   'is_recurrent': True,
                   'is_output': False,
                   'wf': w_config,
                   'bf': b_config,
                   'wi': w_config,
                   'bi': b_config,
                   'wc': w_config,
                   'bc': b_config,
                   'wo': w_config,
                   'bo': b_config}

hidden_1_config = copy.deepcopy(hidden_2_config)
hidden_1_config['var_scope'] = 'lstm_0'

output_config = {'layer_type': 'fc',
                 'var_scope': 'output_layer',
                 'tau': tau,
                 'is_recurrent': False,
                 'is_output': True,
                 'regularization': {'mode': None,
                                    'strength': 0.02},
                 'w': w_config,
                 'b': b_config}

rnn_config = {'layout': [4, 60, 60, 10],
              'layer_configs': [input_config, hidden_1_config, hidden_2_config, output_config],
              'gradient_clip_value': .5,
              'output_type': 'classification',
              'data_multiplier': 1000.}

method = ['l_sampling', 'g_sampling']
training_config = {'learning_rate': 0.002,
                   'type': method[task_id % 2],
                   'is_pretrain': False,
                   'var_reg': 0.00,
                   'beta_reg': .0,
                   'mode': {'name': 'inc_lengths',
                            'in_seq_len': [1, 2, 4, 8, 30],
                            'out_seq_len': [1, 1, 2, 4, 5],
                            'zero_padding': [0, 0, 0, 2, 23],
                            'min_errors': [0., 0., 0., 0., 0.],
                            'max_epochs': [5, 20, 50, 100, 1000]},
                   'task_id': task_id}

training_config['mode'] = {'name': 'classic', 'min_error': 0., 'max_epochs': 0} 
pretrain_config = copy.deepcopy(training_config)


pretrain_config['is_pretrain'] = True
pretrain_config['mode'] = {'name': 'inc_lengths',
                           'in_seq_len': [1, 2, 4, 8, 30],
                           'out_seq_len': [1, 1, 2, 4, 5],
                           'zero_padding': [0, 0, 0, 2, 23],
                           'min_errors': [0., 0., 0., 0., 0.],
                           'max_epochs': [3, 10, 20, 50, 200]}
#pretrain_config['mode']['max_epochs'] = 10
pretrain_config['mode'] = {'name': 'classic', 'min_error': 0., 'max_epochs': 0} 
#reg = [.001, .01, .1, 1.]
pretrain_config['reg'] = 0.01
pretrain_config['path'] = 'trained_for_' + str(int(task_id/2) * 20) + '_epochs'
pretrain_config['just_load'] = True
pretrain_config['no_pretraining'] = False


info_config = {'calc_performance_every': 1,
               'n_samples': 20,
               'tensorboard': {'enabled': True, 'path': '../tb/' + filename + '_' + str(task_id), 'period': 50,
                               'weights': True, 'gradients': False, 'results': True, 'acts': True},
               'profiling': {'enabled': False, 'path': '../profiling/' + filename}}
               

result_config = {'save_results': True,
                 'path': '../numerical_results/' + filename + '_' + str(task_id),
                 'plot_results': False,
                 'print_final_stats': True}

result_dicts = []
for run in range(runs):
    experiment = Experiment()
    result_dicts.append(experiment.train(rnn_config, labelled_data_config, training_config, pretrain_config, info_config))
print('----------------------------')
if result_config['save_results']:
    save_to_file(result_dicts, result_config['path'])


