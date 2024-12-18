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

runs = 3
eval_grad = False
incremental_learning = False
save_results=True
datam = None
#sampling_op = [['c_reparam', 'gste'], ['c_reparam'], ['l_reparam'], ['l_reparam'], ['l_reparam', 'ste']][task_id % 5]
#sampling_op = [['c_reparam', 'gste'], ['c_reparam'], ['l_reparam']][task_id % 3]
#sampling_op = [['c_reparam', 'gste'], ['c_reparam']][int(task_id / 6)]
#sampling_op = ['l_reparam', 'ste']][int(task_id / 8)]
sampling_op = ['c_reparam', 'ste']
#sampling_op = [['c_reparam', 'gste'], ['l_reparam', 'ste']][task_id]
#if int(task_id / 5) == 0:
    #sampling_op = ['c_reparam']
#elif int(task_id / 5) == 1:
    #sampling_op = ['c_reparam', 'ste']

#weight_type=['ternary', 'binary'][int(task_id / 8)]
#weight_type = ['ternary', 'binary'][int(task_id / 8)]
weight_type = 'ternary'
forward_passes = 10

#batchnorm = ['fc', 'x+fc', 'h+fc', 'x+h+fc', 'none'][task_id]
discrete_act = [[], ['i'], ['c'], ['o'], ['i', 'c'], ['i', 'o'], ['c', 'o'], ['i', 'c','o']][int(task_id % 8)]
#discrete_act = []
act_bins = [2., 3., 4., 8.][int(task_id / 8)]
#act_bins = 2.

#tau = np.logspace(-1, 1.5, 6)[task_id % 6]
#tau = [.5, 1.5][int(task_id / 8)]
tau = 1.5
#tau = [1.5, .5][int(task_id / 4)]
batchnorm = []
batchnorm_type = 'none'
#lr_tau = [3000] * 4 + [10000] * 2
#lr_tau = lr_tau[task_id]
#epochs = [10000] * 4 + [35000] * 2
#epochs = epochs[task_id]
#layout = [[40, 40], [60, 60], [80, 80], [100, 100]][int(task_id / 8)]
layout = [40,60]

lr_tau = 15000
#lr = [1., .3, .1, .03][task_id % 4]
#lr = [.2, .1][int(task_id / 6)]
#lr = [.1, .2][int(task_id / 8)]
lr = .2
epochs = 25000

n_samples_per_grad = 1
n_grads = 200000

#filename = ['ar_var', 'arm_var', 'log_var'][task_id]
filename = 'exp'
parametrization = 'sigmoid'
lr_adapt = False

timit_dataset = ['timit_tr_small_0', 'timit_va_small_0']
timit_s_dataset = ['timit_tr_s_0', 'timit_va_s_0']
timit_l_dataset = ['timit_tr_l_0', 'timit_va_l_0']
penstroke_dataset = 'penstroke'
# timit: 13 -> 54
# penstroke: 4 -> 10
labelled_data_config = {'dataset': penstroke_dataset,
                        'remove_bias': True,
                        'tr': {'in_seq_len': 40,
                               'minibatch_enabled': False,
                               'minibatch_size': 1000,
                               'max_truncation': 5},
                        'va': {'in_seq_len': 40,
                               'minibatch_enabled': False,
                               'minibatch_size': 1000,
                               'max_truncation': 5},
                        'te': {'in_seq_len': 40,
                               'minibatch_enabled': False,
                               'minibatch_size': 1000,
                               'max_truncation': 5}}

priors = [[0.2, 0.6, 0.2],[0.1, 0.8, 0.1]]
input_config = {'layer_type': 'input'}
b_config = {'init_m': 'zeros', 'prior_m': 0., 'init_v': -4.5, 'prior_v': 0., 'type': 'continuous'}
bf_config = {'init_m': 'ones', 'prior_m': 1., 'init_v': -4.5, 'prior_v': 0., 'type': 'continuous'}
w_config = {'parametrization': 'sigmoid', 'priors': priors[1], 'type': weight_type, 'pmin': .05, 'pmax':.95, 'p0min': .05, 'p0max': .95}
w_config = {'parametrization': 'sigmoid', 'priors': priors[1], 'type': weight_type, 'pmin': .05, 'pmax':.95, 'p0min': .05, 'p0max': .95}


hidden_2_config = {'layer_type': 'lstm',
                   'var_scope': 'lstm_1',
                   'bias_enabled': True,
                   'parametrization': parametrization,
                   'tau': tau,
                   'discrete_act': discrete_act,
                   'lr_adapt': lr_adapt,
                   'wf': w_config,
                   'bf': bf_config,
                   'wi': w_config,
                   'bi': b_config,
                   'wc': w_config,
                   'bc': b_config,
                   'wo': w_config,
                   'bo': b_config}

hidden_1_config = copy.deepcopy(hidden_2_config)
hidden_1_config['var_scope'] = 'lstm_0'
hidden_1_config['layer_type'] = 'lstm'

output_config = {'layer_type': 'fc',
                 'bias_enabled': True,
                 'var_scope': 'output_layer',
                 'parametrization': parametrization,
                 'tau': tau,
                 'lr_adapt': lr_adapt,
                 'regularization': {'mode': None,
                                    'strength': 0.0},
                 'w': w_config,
                 'b': b_config}

rnn_config = {'layout': [4, layout[0], layout[1], 10],
              'architecture': 'casual',
              'act_disc': discrete_act,
              'act_bins': act_bins,
              'layer_configs': [input_config, hidden_1_config, hidden_2_config, output_config],
              'gradient_clip_value': .5,
              'output_type': 'classification',
              'data_multiplier': datam} 


training_config = {'learning_rate': lr, 
                   'learning_rate_tau': lr_tau,
                   'tau': tau,
                   'algorithm': sampling_op,
                   'carm_iterations': forward_passes,
                   'pretraining': {'path': 'm_model', 'enabled': True, 'sec_path': 'm_ter'},
                   'batchnorm': {'modes': batchnorm,
                                 'type': batchnorm_type,
                                 'momentum': .98,
                                 'tau': 1},
                   'var_reg': 0.,
                   'ent_reg': 0.,
                   'dir_reg': 0., 
                   'mode': {'name': 'inc_lengths',
                            'in_seq_len': [4, 8, 15, 40],
                            'max_truncation': [41, 37, 25, 5],
                            'min_errors': [0., 0., 0., 0.],
                            'max_epochs': [100, 500, 4400, 20000]},
                   'task_id': task_id}

if incremental_learning is False:
    training_config['mode'] = {'name': 'classic', 'min_error': 0., 'max_epochs': epochs} 

info_config = {'calc_performance_every': 1,
               'filename': filename,
               'cell_access': False,
               'gradient': {'samples': n_grads,
                            'grad_per_sample': n_samples_per_grad,
                            'evaluate': eval_grad},
               'save_weights': {'save_every': np.inf,
                                'save_best': False,
                                'path': '../m/w/' + filename + '_' + str(task_id)},
               'tensorboard': {'enabled': save_results, 'path': '../m/tb/' + filename + '_' + str(task_id), 'period': 200,
                               'weights': True, 'gradients': False, 'results': True, 'acts': True, 'single_acts': 1},
               'profiling': {'enabled': False, 'path': '../profiling/' + filename}, 
               'timer': {'enabled': False}}
               

result_config = {'save_results': save_results,
                 'path': '../m/nr/' + filename + '_' + str(task_id),
                 'plot_results': False,
                 'print_final_stats': True}

result_dicts = []
for run in range(runs):
    #if int(task_id / 8) == 0:
        #training_config['pretraining']['path'] = 'prob_m_ter' + str(run)
    #else:
        #training_config['pretraining']['path'] = 'prob_m_bin' + str(run)

    experiment = Experiment()
    result_dicts.append(experiment.train(rnn_config, labelled_data_config, training_config, info_config, run))
print('----------------------------')
if result_config['save_results']:
    save_to_file(result_dicts, result_config['path'])
print_results(result_dicts)


