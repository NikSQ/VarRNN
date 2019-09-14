# Manages results from training such as ELBO, accuracy, kl loss...
# Offers methods for retrieving these results from the rnn class and storing them

import numpy as np
import tensorflow as tf
from src.global_variable import get_train_config


def save_to_file(result_dicts, path):
    for process_key in result_dicts[0].keys():
        if process_key == 'epoch':
            continue

        if process_key.endswith('_s'):
            metrics = convert_to_array(result_dicts, process_key, ['acc', 'loss'])
        elif process_key.endswith('_b'):
            metrics = convert_to_array(result_dicts, process_key, ['vfe', 'kl', 'elogl', 'acc'])
        else:
            raise Exception('process key {} not understood'.format(process_key))

        for metric_key in metrics.keys():
            np.save(path + '_' + process_key + '_' + metric_key, metrics[metric_key])
    np.save(path + '_epochs', np.asarray(result_dicts[0]['epoch']))


def print_results(result_dicts):
    it_earlystop = []
    for process_key in result_dicts[0].keys():
        if process_key in ['va_b', 'va_s']:
            print('Results for {}'.format(process_key))
            if process_key.endswith('s'):
                metric_keys = ['acc', 'loss']
            else:
                metric_keys = ['elogl', 'acc']

            for metric_key in metric_keys:
                print('Metric: {:6s}'.format(metric_key))
                for run in range(len(result_dicts)):
                    extrema = np.max(result_dicts[run][process_key][metric_key])
                    if metric_key == 'elogl':
                        idx = np.argmin(result_dicts[run][process_key][metric_key])
                        print('{:6.3f} in iteration {:4d}'.format(-extrema, idx))
                    elif metric_key == 'acc':
                        idx = np.argmax(result_dicts[run][process_key][metric_key])
                        if process_key == 'va_s':
                            it_earlystop.append(idx)
                        print('{:6.2f} % in iteration {:4d}'.format(extrema*100, idx))

    for process_key in result_dicts[0].keys():
        if process_key in ['te_s']:
            print('EARLY STOP FOR {}'.format(process_key))
            metric_key = 'acc'
            idx = it_earlystop[0]
            value = result_dicts[0][process_key][metric_key][idx]
            print('{:6.2f} % in iteration {:4d}'.format(value*100, idx))


def convert_to_array(result_dicts, process_key, metric_keys):
    metrics = dict()
    for metric_key in metric_keys:
        metrics[metric_key] = list()

    for run, result_dict in enumerate(result_dicts):
        for metric_key in metric_keys:
            metrics[metric_key].append(np.expand_dims(np.asarray(result_dict[process_key][metric_key], np.float32),
                                       axis=0))

    for metric_key in metric_keys:
        metrics[metric_key] = np.concatenate(metrics[metric_key])

    return metrics


class TMetrics:
    def __init__(self, l_data_config, l_data, is_training, tau):
        self.is_training = is_training
        self.tau = tau
        self.l_data_config = l_data_config
        self.l_data = l_data
        self.result_dict = {'epoch': list()}
        self.s_batchnorm_stats_op = None
        self.b_batchnorm_stats_op = None
        self.op_dict = {}
        self.best_va = {'is_current': False,
                        'acc': 0.}


    # Connects metrics of datasets to TResults. Needs to be called for each dataset (training, validation and or test)
    # while building the graph
    def add_b_vars(self, process_key, vfe_op, kl_op, elogl_op, accs_op):
        self.result_dict.update({process_key: {'vfe': [], 'kl': [], 'elogl': [], 'acc': []}})
        self.op_dict.update({process_key: [vfe_op, kl_op, elogl_op, accs_op]})

    def add_s_vars(self, process_key, sample_op, loss_op, accs_op):
        self.result_dict.update({process_key: {'loss': [], 'acc': []}})
        self.op_dict.update({process_key: {'metrics': [loss_op, accs_op], 'sample': sample_op}})

    # Retrieves metrics of the performance of the processes which were added using add_vars()
    # A process is a method of operating a RNN (bayesian or sampled weights) combined with a dataset
    def retrieve_results(self, sess, epoch, tau, is_pretrain=False):
        if get_train_config()['batchnorm']:
            self.retrieve_s_results(sess, 'tr_s', is_pretrain, True)

        for process_key in self.op_dict.keys():
            if process_key.endswith('_s') and process_key is not 'tr_s':
                self.retrieve_s_results(sess, process_key, is_pretrain, False)
            elif process_key.endswith('_b'):
                self.retrieve_b_results(sess, process_key, False, tau)
            else:
                raise Exception('process key {} not understood'.format(process_key))
        self.result_dict['epoch'].append(epoch)

    def retrieve_b_results(self, sess, process_key, is_training, tau):
        data_key = process_key[:-2]
        cum_vfe = 0
        cum_acc = 0
        elogl = 0
        for minibatch_idx in range(self.l_data.data[data_key]['n_minibatches']):
            vfe, kl, elogl, acc = sess.run(self.op_dict[process_key],
                                             feed_dict={self.tau: (tau,), self.l_data.batch_idx: minibatch_idx,
                                                        self.is_training: is_training})
            cum_vfe += vfe
            cum_acc += acc
            elogl += elogl
        acc = cum_acc / self.l_data.data[data_key]['n_minibatches']
        vfe = cum_vfe / self.l_data.data[data_key]['n_minibatches']

        self.result_dict[process_key]['vfe'].append(vfe)
        self.result_dict[process_key]['kl'].append(kl)
        self.result_dict[process_key]['elogl'].append(elogl)
        self.result_dict[process_key]['acc'].append(acc)

    def retrieve_s_results(self, sess, process_key, is_pretrain, is_training):
        data_key = process_key[:-2]

        cum_loss = 0
        cum_acc = 0
        if is_pretrain is False:
            sess.run(self.op_dict[process_key]['sample'])
        for minibatch_idx in range(self.l_data.data[data_key]['n_minibatches']):
            loss, acc = sess.run(self.op_dict[process_key]['metrics'],
                                 feed_dict={self.l_data.batch_idx: minibatch_idx, self.is_training: is_training})
            cum_loss += loss
            cum_acc += acc
        if is_training:
            return
        loss = cum_loss / self.l_data.data[data_key]['n_minibatches']
        acc = cum_acc / self.l_data.data[data_key]['n_minibatches']

        if process_key == 'va_s':
            if self.best_va['acc'] < acc:
                self.best_va['acc'] = acc
                self.best_va['is_current'] = True
            else:
                self.best_va['is_current'] = False

        self.result_dict[process_key]['loss'].append(loss)
        self.result_dict[process_key]['acc'].append(acc)

    # Prints the latest metrics of the performance
    def print(self, session_idx):
        #print('{:3} | TrAcc: {:6.4f}, TrLoss: {:8.5f}'.format(self.result_dict['epoch'][-1], self.result_dict['tr_b']['acc'][-1],
                      #self.result_dict['tr_b']['vfe'][-1]))
        print('{:3}, {:2} | TrAcc: {:6.4f}, TrLoss: {:8.5f}, VaAcc: {:6.4f}, VaLoss: {:8.5f}'
              .format(self.result_dict['epoch'][-1], session_idx, self.result_dict['tr_b']['acc'][-1],
                      self.result_dict['tr_b']['vfe'][-1], self.result_dict['va_b']['acc'][-1],
                      self.result_dict['va_b']['vfe'][-1]) +
              '\t MAP NN | Acc: {:6.4f} | Loss: {:6.4f}'
              .format(self.result_dict['va_s']['acc'][-1],
                      self.result_dict['va_s']['loss'][-1]))
