# Manages results from training such as ELBO, accuracy, kl loss...
# Offers methods for retrieving these results from the rnn class and storing them

import numpy as np


def save_to_file(result_dicts, path):
    for process_key in result_dicts[0].keys():
        if process_key == 'epoch':
            continue

        if process_key.endswith('_s'):
            metrics = convert_to_array(result_dicts, process_key, ['m_loss', 's_loss', 'm_acc', 's_acc'])
        elif process_key.endswith('_b'):
            metrics = convert_to_array(result_dicts, process_key, ['vfe', 'kl', 'elogl', 'acc'])
        else:
            raise Exception('process key {} not understood'.format(process_key))

        for metric_key in metrics.keys():
            np.save(path + '_' + process_key + '_' + metric_key, metrics[metric_key])
    np.save(path + '_epochs', np.asarray(result_dicts[0]['epoch']))


def print_results(result_dicts):
    for process_key in result_dicts[0].keys():
        if process_key in ['va_b', 'va_s', 'te_b', 'te_s']:
            print('Results for {}'.format(process_key))
            if process_key.endswith('s'):
                metric_keys = ['m_acc']
            else:
                metric_keys = ['elogl', 'acc']
            for metric_key in metric_keys:
                print('Metric: {:6s}'.format(metric_key))
                for run in range(len(result_dicts)):
                    extrema = np.max(result_dicts[run][process_key][metric_key])
                    idx = np.argmax(result_dicts[run][process_key][metric_key])
                    if metric_key == 'elogl':
                        print('{:6.3f} in iteration {:4d}'.format(-extrema, idx))
                    else:
                        print('{:6.2f} % in iteration {:4d}'.format(extrema*100, idx))


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
    def __init__(self, l_data_config, l_data, info_config):
        self.info_config = info_config
        self.l_data_config = l_data_config
        self.l_data = l_data
        self.result_dict = {'epoch': list()}
        self.op_dict = {}

    # Connects metrics of datasets to TResults. Needs to be called for each dataset (training, validation and or test)
    # while building the graph
    def add_b_vars(self, process_key, vfe_op, kl_op, elogl_op, accs_op):
        self.result_dict.update({process_key: {'vfe': [], 'kl': [], 'elogl': [], 'acc': []}})
        self.op_dict.update({process_key: [vfe_op, kl_op, elogl_op, accs_op]})

    def add_s_vars(self, process_key, sample_op, loss_op, accs_op):
        self.result_dict.update({process_key: {'m_loss': [], 's_loss': [], 'm_acc': [], 's_acc': []}})
        self.op_dict.update({process_key: {'metrics': [loss_op, accs_op], 'sample': sample_op}})

    # Retrieves metrics of the performance of the processes which were added using add_vars()
    # A process is a method of operating a RNN (bayesian or sampled weights) combined with a dataset
    def retrieve_results(self, sess, epoch, is_pretrain=False):
        for process_key in self.op_dict.keys():
            if process_key.endswith('_s'):
                self.retrieve_s_results(sess, process_key, is_pretrain)
            elif process_key.endswith('_b'):
                self.retrieve_b_results(sess, process_key)
            else:
                raise Exception('process key {} not understood'.format(process_key))
        self.result_dict['epoch'].append(epoch)

    def retrieve_b_results(self, sess, process_key):
        data_key = process_key[:-2]
        cum_vfe = 0
        cum_acc = 0
        elogl = 0
        for minibatch_idx in range(self.l_data.data[data_key]['n_minibatches']):
            vfe, kl, elogl, acc = sess.run(self.op_dict[process_key],
                                             feed_dict={self.l_data.batch_idx: minibatch_idx})
            cum_vfe += vfe
            cum_acc += acc
            elogl += elogl
        acc = cum_acc / self.l_data.data[data_key]['n_minibatches']
        vfe = cum_vfe / self.l_data.data[data_key]['n_minibatches']

        self.result_dict[process_key]['vfe'].append(vfe)
        self.result_dict[process_key]['kl'].append(kl)
        self.result_dict[process_key]['elogl'].append(elogl)
        self.result_dict[process_key]['acc'].append(acc)

    def retrieve_s_results(self, sess, process_key, is_pretrain):
        data_key = process_key[:-2]
        losses = list()
        accs = list()

        if is_pretrain is False:
            loop_range = range(self.info_config['n_samples'])
        else:
            loop_range = range(1)

        for sample_idx in loop_range:
            cum_loss = 0
            cum_acc = 0
            if is_pretrain is False:
                sess.run(self.op_dict[process_key]['sample'])
            for minibatch_idx in range(self.l_data.data[data_key]['n_minibatches']):
                loss, acc = sess.run(self.op_dict[process_key]['metrics'],
                                     feed_dict={self.l_data.batch_idx: minibatch_idx})
                cum_loss += loss
                cum_acc += acc
            accs.append(cum_acc / self.l_data.data[data_key]['n_minibatches'])
            losses.append(cum_loss / self.l_data.data[data_key]['n_minibatches'])
        losses = np.asarray(losses)
        accs = np.asarray(accs)
        self.result_dict[process_key]['m_loss'].append(np.mean(losses))
        self.result_dict[process_key]['m_acc'].append(np.mean(accs))

        if is_pretrain is False:
            self.result_dict[process_key]['s_loss'].append(np.std(losses, ddof=1))
            self.result_dict[process_key]['s_acc'].append(np.std(accs, ddof=1))

    # Prints the latest metrics of the performance
    def print(self, session_idx):
        print('{:3}, {:2} | TrAcc: {:6.4f}, TrLoss: {:8.5f}, VaAcc: {:6.4f}, VaLoss: {:8.5f}'
              .format(self.result_dict['epoch'][-1], session_idx, self.result_dict['tr_b']['acc'][-1],
                      self.result_dict['tr_b']['vfe'][-1], self.result_dict['va_b']['acc'][-1],
                      self.result_dict['va_b']['vfe'][-1]) +
              '\t {} sampled NNs | VaAcc: {:6.4f} +- {:6.4f} | VaLoss: {:6.4f} +- {:6.4f}'
              .format(self.info_config['n_samples'], self.result_dict['va_s']['m_acc'][-1],
                      self.result_dict['va_s']['s_acc'][-1], self.result_dict['va_s']['m_loss'][-1],
                      self.result_dict['va_s']['s_loss'][-1]))
