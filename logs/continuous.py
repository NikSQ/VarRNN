import sys
import os
import tensorflow as tf
import numpy as np
import copy

sys.path.append('../')

from src.experiment import Experiment
from src.data.t_metrics import save_to_file, print_results

try:
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
except KeyError:
    print('NO SLURM TASK ID FOUND')
    print('Using ID 0 instead')
    task_id = 0

runs = 1
filename = 'exp'

timit_dataset = ['timit_tr_small_0', 'timit_va_small_0']
timit_s_dataset = ['timit_tr_s_0', 'timit_va_s_0']
timit_l_dataset = ['timit_tr_l_0', 'timit_va_l_0']
penstroke_dataset = 'penstroke'

# timit: 13 -> 54
# penstroke: 4 -> 10
labelled_data_config = {'dataset': penstroke_dataset,
                        'tr': {'in_seq_len': 25,
                               'out_seq_len': 1,
                               'zero_padding': 0,
                               'minibatch_enabled': False,
                               'minibatch_size': 1000,
                               'extract_seqs': False},
                        'va': {'in_seq_len': 25,
                               'out_seq_len': 1,
                               'zero_padding': 0,
                               'minibatch_enabled': False,
                               'minibatch_size': 1000,
                               'extract_seqs': False}}

input_config = {'layer_type': 'input'}


w_config = {'init_m': 'xavier', 'prior_m': 0., 'init_v': -4.5, 'prior_v': 0., 'type': 'continuous'}

hidden_2_config = {'layer_type': 'lstm',
                   'var_scope': 'lstm_1',
                   'init_config': {'f': {'w': 'all_zero', 'b': 'all_one'},
                                   'i': {'w': 'xavier', 'b': 'all_zero'},
                                   'c': {'w': 'xavier', 'b': 'all_zero'},
                                   'o': {'w': 'xavier', 'b': 'all_zero'}},
                   'is_recurrent': True,
                   'discrete_act': False,
                   'normalize_mean': False,
                   'is_output': False,
                   'wf': w_config,
                   'bf': w_config,
                   'wi': w_config,
                   'bi': w_config,
                   'wc': w_config,
                   'bc': w_config,
                   'wo': w_config,
                   'bo': w_config}

hidden_1_config = copy.deepcopy(hidden_2_config)
hidden_1_config['var_scope'] = 'lstm_0'

output_config = {'layer_type': 'fc',
                 'var_scope': 'output_layer',
                 'is_recurrent': False,
                 'is_output': True,
                 'normalize_mean': False,
                 'regularization': {'mode': None,
                                    'strength': 0.02},
                 'w': w_config,
                 'b': w_config}

datam = [1., 1., None, 1., 1., None, 1., 1., None]
method = ['pfp']*3 + ['l_sampling']*3 + ['g_sampling']*3
var_reg = [0., 10., 0., 0., 10., 0., 0., 10., 0.]

rnn_config = {'layout': [4, 60, 60, 10],
              'layer_configs': [input_config, hidden_1_config, hidden_2_config, output_config],
              'gradient_clip_value': .5,
              'output_type': 'classification',
              'data_multiplier': datam[task_id]}

train_config = {'learning_rate': 0.1,
                'type': method[task_id],
                'is_pretrain': False,
                'var_reg': var_reg[task_id],
                'dir_reg': 0.,
                'ent_reg': 0.,
                'normalize_mean': False,
                'mode': {'name': 'inc_lengths',
                         'in_seq_len': [1, 2, 4, 8, 30],
                         'out_seq_len': [1, 1, 2, 4, 5],
                         'zero_padding': [0, 0, 0, 2, 23],
                         'min_errors': [0., 0., 0., 0., 0.],
                         'max_epochs': [2, 10, 30, 50, 1000]},
                'task_id': task_id}

train_config['mode'] = {'name': 'classic', 'min_error': 0., 'max_epochs': 1000}
pretrain_config = copy.deepcopy(train_config)
pretrain_config['path'] = 'continuous'
pretrain_config['mode']['max_epochs'] = 155
pretrain_config['is_pretrain'] = True
pretrain_config['no_pretraining'] = False
pretrain_config['just_load'] = False
pretrain_config['reg'] = 1.

info_config = {'calc_performance_every': 1,
               'n_samples': 10,
               'tensorboard': {'enabled': True, 'path': '../tb/' + filename + '_' + str(task_id), 'period': 20,
                               'weights': True, 'gradients': True, 'results': True, 'acts': True, 'single_acts': 1},
               'profiling': {'enabled': False, 'path': '../profiling/' + filename},
               'timer': {'enabled': False}}
               

result_config = {'save_results': True,
                 'path': '../numerical_results/' + filename + '_' + str(task_id),
                 'plot_results': False,
                 'print_final_stats': True}

result_dicts = []
for run in range(runs):
    experiment = Experiment()
    result_dicts.append(experiment.train(rnn_config, labelled_data_config, train_config, pretrain_config, info_config))
print('----------------------------')
if result_config['save_results']:
    save_to_file(result_dicts, result_config['path'])
print_results(result_dicts)

