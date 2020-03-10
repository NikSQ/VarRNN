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

runs = 6
filename = 'exp'
eval_grad = False
incremental_learning = False
save_results = True
datam = None

tau = 1.4
lr_tau = 9000
epochs = 10000
sampling_op = ['l_reparam', 'ste']
layout = [50, 40]
discrete_act = [[], ['i'], ['c'], ['o'], ['i', 'c'], ['i', 'o'], ['c', 'o'], ['i', 'c', 'o']][task_id % 8]
#discrete_act = [['i'], ['c', 'o'], ['i', 'c', 'o']][int(task_id / 3)]
weight_type = ['ternary', 'binary'][int(task_id / 8)]

#lr = [.03, .1, .003][task_id % 3]
lr = .1
n_samples_per_grad = 100

n_grads = 100000

batchnorm = []
batchnorm_type = 'none'

parametrization = 'sigmoid'
lr_adapt = False

timit_dataset = ['timit_tr_small_0', 'timit_va_small_0']
timit_s_dataset = ['timit_tr_s_0', 'timit_va_s_0']
timit_l_dataset = ['timit_tr_l_0', 'timit_va_l_0']
penstroke_dataset = 'penstroke'
# timit: 13 -> 54
# penstroke: 4 -> 10
labelled_data_config = {'dataset': 'timit_s',
                        'tr': {'in_seq_len': 20,
                               'max_truncation': 0,
                               'minibatch_enabled': False,
                               'minibatch_size': 5000},
                        'va': {'in_seq_len': 20,
                               'max_truncation': 0,
                               'minibatch_enabled': False,
                               'minibatch_size': 5000},
                        'te': {'in_seq_len': 20,
                               'max_truncation': 0,
                               'minibatch_enabled': False,
                               'minibatch_size': 1000}}

priors = [[0.2, 0.6, 0.2],[0.1, 0.8, 0.1]]
input_config = {'layer_type': 'input'}
b_config = {'init_m': 'xavier', 'prior_m': 0., 'init_v': -4.5, 'prior_v': 0., 'type': 'continuous'}
w_config = {'priors': priors[1], 'type': weight_type, 'pmin': .01, 'pmax':.99, 'p0min': .05, 'p0max': .95}

hidden_2_config = {'layer_type': 'lstm',
                   'var_scope': 'lstm_1',
                   'tau': tau,
                   'parametrization': parametrization,
                   'discrete_act': discrete_act,
                   'normalize_mean': False,
                   'lr_adapt': False,
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
hidden_1_config['layer_type'] = 'blstm'

output_config = {'layer_type': 'fc',
                 'var_scope': 'output_layer',
                 'parametrization': parametrization,
                 'normalize_mean': False,
                 'tau': tau,
                 'is_recurrent': False,
                 'is_output': True,
                 'lr_adapt': False,
                 'regularization': {'mode': None,
                                    'strength': 0.02},
                 'w': w_config,
                 'b': b_config}

rnn_config = {'layout': [13, layout[0], layout[1], 54],
              'act_disc': discrete_act,
              'layer_configs': [input_config, hidden_1_config, hidden_2_config, output_config],
              'gradient_clip_value': .5,
              'output_type': 'classification',
              'data_multiplier': datam}

training_config = {'learning_rate': lr, 
                   'learning_rate_tau': lr_tau,
                   'tau': tau,
                   'algorithm': sampling_op,
                   'pretraining': {'path': 't_model', 'enabled': True, 'sec_path': 't_bin'},
                   'batchnorm': {'modes': batchnorm,
                                 'type': batchnorm_type,
                                 'momentum': .98,
                                 'tau': 1},
                   'var_reg': 0.,
                   'ent_reg': 0.,
                   'dir_reg': 0., 
                   'mode': {'name': 'inc_lengths',
                            'in_seq_len': [1, 2, 4, 8, 30],
                            'out_seq_len': [1, 1, 2, 4, 5],
                            'zero_padding': [0, 0, 0, 2, 23],
                            'min_errors': [0., 0., 0., 0., 0.],
                            'max_epochs': [5, 20, 50, 100, 1000]},
                   'task_id': task_id}

if incremental_learning is False:
    training_config['mode'] = {'name': 'classic', 'min_error': 0., 'max_epochs': epochs} 


info_config = {'calc_performance_every': 1,
               'cell_access': False,
               'filename': filename,
               'gradient': {'samples': n_grads,
                            'grad_per_sample': n_samples_per_grad,
                            'evaluate': eval_grad},
               'save_weights': {'save_every': np.inf,
                                'save_best': False,
                                'path': '../t/w/' + filename + '_' + str(task_id)},
               'tensorboard': {'enabled': False, 'path': '../t/tb/' + filename + '_' + str(task_id), 'period': 200,
                               'weights': True, 'gradients': False, 'results': True, 'acts': True, 'single_acts': 1},
               'profiling': {'enabled': False, 'path': '../profiling/' + filename}, 
               'timer': {'enabled': False}}
               

result_config = {'save_results': save_results,
                 'path': '../t/nr/' + filename + '_' + str(task_id),
                 'plot_results': False,
                 'print_final_stats': True}

result_dicts = []
for run in range(runs):
    if weight_type == 'ternary':
        training_config['pretraining']['path'] = 'prob_m_ter' + str(run)
    elif weight_type == 'binary':
        training_config['pretraining']['path'] = 'prob_m_bin' + str(run)
    else:
        raise Exception('')
    experiment = Experiment()
    result_dicts.append(experiment.train(rnn_config, labelled_data_config, training_config, info_config, run))
print('----------------------------')
if result_config['save_results']:
    save_to_file(result_dicts, result_config['path'])
print_results(result_dicts)


