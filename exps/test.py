import sys
import os
import tensorflow as tf
import numpy as np

sys.path.append('../')

from src.experiment import Experiment
from src.tools import process_results


try:
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
except KeyError:
    print('NO SLURM TASK ID FOUND')
    print('Using ID 0 instead')
    task_id = 0

runs = 1
timit_dataset = ['timit_tr_small_0', 'timit_va_small_0']
timit_s_dataset = ['timit_tr_s_0', 'timit_va_s_0']
timit_l_dataset = ['timit_tr_l_0', 'timit_va_l_0']
penstroke_dataset = 'pen_stroke_small'
# timit: 13 -> 54
# penstroke: 4 -> 10
labelled_data_config = {'dataset': timit_dataset,
                        'tr': {'in_seq_len': 20,
                               'out_seq_len': 5,
                               'zero_padding': 12,
                               'mini_batch_mode': False,
                               'batch_size': 100},
                        'va': {'in_seq_len': 16,
                               'out_seq_len': 2,
                               'zero_padding': 12,
                               'mini_batch_mode': False,
                               'batch_size': 100}}

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

w_config = {'init_m': 'xavier', 'prior_m': 0, 'init_v': 'xavier'}

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

output_config = {'layer_type': 'fc',
                 'var_scope': 'output_layer',
                 'is_recurrent': False,
                 'is_output': True,
                 'regularization': {'mode': None,
                                    'strength': 0.02},
                 'w': w_config,
                 'b': w_config}

rnn_config = {'layout': [13, 5, 54],
              'layer_configs': [input_config, hidden_2_config, output_config],
              'gradient_clip_value': .5,
              'output_type': 'classification',
              'data_multiplier': 1}

training_config = {'learning_rate': 0.05,
                   'max_epochs': 1000,
                   'mode': {'name': 'inc_lengths',
                            'in_seq_len': [1, 2, 4, 8, 16],
                            'out_seq_len': [1, 1, 2, 4, 12],
                            'zero_padding': [0, 0, 0, 2, 2],
                            'min_errors': [3.5, 3.5, 3, 2, 0.1],
                            'max_epochs': [20, 20, 20, 20, 20]},
                   'task_id': task_id}

training_config['mode'] = {'name': 'classic', 'min_error': 0, 'max_epochs': 500}

info_config = {'calc_performance_every': 1,
               'include_pred': False,
               'include_out': False,
               'tensorboard': {'is_enabled': True, 'path': '../tb/run2', 'period': 1,
                               'weights': True, 'gradients': True, 'loss': True}}

result_config = {'save_results': False,
                 'filename': '../numerical_results/test',
                 'plot_results': False,
                 'print_final_stats': True}

result_dicts = []
for run in range(runs):
    experiment = Experiment()
    result_dicts.append(experiment.train(rnn_config, labelled_data_config, training_config, info_config))
print('----------------------------')
print(result_dicts[0])
process_results(result_config, result_dicts)


