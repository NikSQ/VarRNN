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

runs = 2
timit_dataset = ['timit_tr_small_0', 'timit_va_small_0']
timit_s_dataset = ['timit_tr_s_0', 'timit_va_s_0']
timit_l_dataset = ['timit_tr_l_0', 'timit_va_l_0']
red_penstroke_dataset = 'pen_stroke_small'
penstroke_dataset = 'penstroke'
# timit: 13 -> 54
# penstroke: 4 -> 10
labelled_data_config = {'dataset': penstroke_dataset,
                        'tr': {'in_seq_len': 6,
                               'out_seq_len': 1,
                               'zero_padding': 0,
                               'minibatch_enabled': True,
                               'minibatch_size': 5000},
                        'va': {'in_seq_len': 6,
                               'out_seq_len': 1,
                               'zero_padding': 0,
                               'minibatch_enabled': True,
                               'minibatch_size': 5000}}

input_config = {'layer_type': 'input'}

hidden_1_config = {'layer_type': 'fc',
                   'var_scope': 'fc_1',
                   'init_config': {'w': 'xavier', 'b': 'all_zero'},
                   'act_func': tf.nn.relu,
                   'is_recurrent': True,
                   'is_output': False,
                   'prior_w_m': 0,
                   'prior_w_v': 1,
                   'prior_b_m': 0,
                   'prior_b_v': 1}

initv = [-4.]
w_config = {'init_m': 'xavier', 'prior_m': 0, 'init_v': initv[task_id], 'prior_v': initv[task_id]}

hidden_2_config = {'layer_type': 'lstm',
                   'var_scope': 'lstm_1',
                   'init_config': {'f': {'w': 'all_zero', 'b': 'all_one'},
                                   'i': {'w': 'xavier', 'b': 'all_zero'},
                                   'c': {'w': 'xavier', 'b': 'all_zero'},
                                   'o': {'w': 'xavier', 'b': 'all_zero'}},
                   'is_recurrent': True,
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
                 'regularization': {'mode': None,
                                    'strength': 0.02},
                 'w': w_config,
                 'b': w_config}

n_lstm = [5, 10, 15, 20, 30, 40, 50, 65, 85, 110, 150, 180] #6 and 7
n_lstm1 = [20, 20, 20, 40, 40, 40, 60, 60, 60] # 60
n_lstm2 = [20, 40, 60, 20, 40, 60, 20, 40, 60] # 60
rnn_config = {'layout': [4, 60, 60, 10],
              'layer_configs': [input_config, hidden_1_config, hidden_2_config, output_config],
              'gradient_clip_value': .5,
              'output_type': 'classification',
              'data_multiplier': 1000000}

training_config = {'learning_rate': 0.002,
                   'max_epochs': 100,
                   'type': 'sampling',
                   'mode': {'name': 'inc_lengths',
                            'in_seq_len': [1, 2, 4, 8, 30],
                            'out_seq_len': [1, 1, 2, 4, 5],
                            'zero_padding': [0, 0, 0, 2, 23],
                            'min_errors': [0., 0., 0., 0., 0.],
                            'max_epochs': [2, 10, 30, 50, 1000]},
                   'task_id': task_id}

training_config['mode'] = {'name': 'classic', 'min_error': 0., 'max_epochs': 1}

filename = 'test'

info_config = {'calc_performance_every': 1,
               'n_samples': 10,
               'tensorboard': {'enabled': False, 'path': '../tb/' + filename, 'period': 1,
                               'weights': True, 'gradients': True, 'results': True},
               'profiling': {'enabled': False, 'path': '../profiling/pfp'}}

result_config = {'save_results': True,
                 'path': '../numerical_results/' + filename,
                 'plot_results': False,
                 'print_final_stats': True}

result_dicts = []
for run in range(runs):
    experiment = Experiment()
    result_dicts.append(experiment.train(rnn_config, labelled_data_config, training_config, info_config))
print('----------------------------')
save_to_file(result_dicts, result_config['path'])


