from scipy.io import loadmat
import numpy as np
import tensorflow as tf
import itertools

from src.data.preprocessing import extract_seqs, extract_sequences
from src.configuration.data_config import DataConfig
from src.configuration.constants import DatasetKeys


class Dataset:
    def __init__(self, batches, ds_name):
        self.batches = batches
        self.ds_name = ds_name


class BatchMap:
    x = 0
    Y = 1
    SEQLEN = 2


def load_api_datasets(data_config):
    datasets = {}
    for data_key in data_config.ds_configs.keys():
        dataset_config = data_config.ds_configs[data_key]
        unprocessed_data = loadmat(dataset_config.filename)
        x, y, seqlens = extract_sequences(unprocessed_data, dataset_config)

        ds = tf.data.Dataset.from_tensor_slices((x, y, seqlens))

        ds = ds.cache()
        if dataset_config.do_shuffle:
            ds = ds.shuffle(buffer_size=dataset_config.shuffle_buffer)
        if dataset_config.minibatch_enabled:
            ds = ds.batch(dataset_config.minibatch_size)

        ds.prefetch(1)
        dataset = Dataset(ds, data_key)
        datasets[data_key] = dataset
    return datasets

# Loads Dataset into GPU
def load_gpu_datasets(data_config):
    datasets = {}
    
    for data_key in data_config.ds_configs.keys():
        dataset_config = data_config.ds_configs[data_key]
        unprocessed_data = loadmat(dataset_config.filename)

        dataset = {}
        dataset[DatasetKeys.X], dataset[DatasetKeys.Y], dataset[DatasetKeys.SEQLEN] = \
            extract_sequences(unprocessed_data, dataset_config)
        datasets[data_key] = dataset
        print(data_key)
        print(dataset[DatasetKeys.X].shape)
    
    #datasets = toy_samples(data_config)
    return datasets

def toy_samples(data_config):
    datasets = {}
    for data_key in data_config.ds_configs.keys():
        dataset = {}
        n_samples = 60
        n_timesteps = 11
        n_features = 2

        x = np.zeros((n_samples, n_timesteps, n_features))
        y = np.zeros((n_samples, 3))
        seqlen = (np.ones((n_samples,)) * (n_timesteps -2)).astype(np.int32)
        for idx in range(0, 20):
            x[idx, :, 0] = np.arange(n_timesteps) * np.random.uniform(-1./n_timesteps, 0) + np.random.randn(n_timesteps) * .1
            y[idx] = np.array([1, 0, 0])

        for idx in range(20, 40):
            x[idx, :, 0] = np.arange(n_timesteps) * np.random.uniform(0, 1.0/n_timesteps) + np.random.randn(n_timesteps) * .1
            y[idx] = np.array([0, 1, 0])

        for idx in range(40, 60):
            x[idx, :, 0] = np.cos(np.arange(n_timesteps)) + np.random.randn(n_timesteps) * .1
            y[idx] = np.array([0, 0, 1])
        dataset[DatasetKeys.X] = np.transpose(x, axes=[0, 2, 1])
        dataset[DatasetKeys.Y] = y
        dataset[DatasetKeys.SEQLEN] = seqlen
        datasets[data_key] = dataset
    return datasets


filenames = {'penstroke_tr': '../datasets/mnist_pen_strokes/mps_full1_tr.mat',
             'penstroke_va': '../datasets/mnist_pen_strokes/mps_full1_va.mat',
             'penstroke_te': '../datasets/mnist_pen_strokes/mps_full1_te.mat',
             'sign_language_tr': '../../datasets/sign_language/sign_language_tr.mat',
             'sign_language_va': '../../datasets/sign_language/sign_language_va.mat',
             'sign_language_te': '../../datasets/sign_language/sign_language_te.mat',
             'syn_dataset_tr': '../../datasets/synthetic/syn_tr.mat',
             'syn_dataset_va': '../../datasets/synthetic/syn_va.mat',
             'syn_dataset_te': '../../datasets/synthetic/syn_te.mat'}


def load_dataset(l_data_config):
    entry = l_data_config['dataset']
    if entry.startswith('timit'):
        data_dict = load_timit(l_data_config)
    elif entry.startswith('penstroke'):
        data_dict = get_datadict(l_data_config, 'penstroke')
    elif entry.startswith('sign'):
        data_dict = get_datadict(l_data_config, 'sign_language')
    elif entry.startswith('syn'):
        data_dict = get_datadict(l_data_config, 'syn_dataset')
    elif entry.startswith('t_seq'):
        data_dict = load_timit_seq_seq(l_data_config)

    return data_dict

def load_timit_seq_seq(l_data_config):
    if l_data_config['dataset'] == 't_seq_s':
        n_te_phonems = 500
        n_tv_phonems = 1000
        # in_seq_len = 637 (tr + va), 756 (te)
    path = '../../datasets/timit_seq/'
    data_dict = {'tr': dict(), 'va': dict(), 'te': dict()}
    partial_dict = {'x': [], 'y': [], 'seqlen': [], 'y_seqlen': []}
    x_max_len = 0
    y_max_len = 0
    speaker_lens = []
    n_phonems = 0
    for mat_idx in range(16):
        print(mat_idx)
        partial_set = loadmat(path + 'seq_train_' + str(mat_idx + 1) + '.mat')
        seqlen = np.squeeze(partial_set['seqlen']).astype(np.int32)
        y_seqlen = np.squeeze(partial_set['y_seqlen']).astype(np.int32)
        partial_dict['seqlen'] = np.concatenate([partial_dict['seqlen'], seqlen], axis=0)
        partial_dict['y_seqlen'] = np.concatenate([partial_dict['y_seqlen'], y_seqlen], axis=0)
        partial_dict['x'].append(partial_set['x'])
        partial_dict['y'].append(partial_set['y'])
        speaker_lens.append(partial_set['x'].shape[0])
        if x_max_len < partial_set['x'].shape[2]:
            x_max_len = partial_set['x'].shape[2]
        if y_max_len < partial_set['y'].shape[2]:
            y_max_len = partial_set['y'].shape[2]
        n_phonems += partial_set['x'].shape[0]
        if n_phonems > n_tv_phonems:
            break
    partial_dict['y_seqlen'] = partial_dict['y_seqlen'].astype(np.int32)
    xs = []
    ys = []
    for x, y in itertools.zip_longest(partial_dict['x'], partial_dict['y']):
        x_shape = x.shape
        xs.append(np.concatenate([np.zeros((x_shape[0], x_shape[1], x_max_len - x_shape[2])), x], axis=2))
        y_shape = y.shape
        ys.append(np.concatenate([np.zeros((y_shape[0], y_shape[1], y_max_len - y_shape[2])), y], axis=2))
    partial_dict['x'] = np.concatenate(xs, axis=0)
    partial_dict['y'] = np.concatenate(ys, axis=0)
    # x_max_len = 637

    permuted_indices = np.arange(n_phonems)
    n_tr_phonems = int(len(permuted_indices) * 0.8)
    tr_speaker_idc = permuted_indices[:n_tr_phonems]
    tr_idc = tr_speaker_idc
    va_idc = permuted_indices[n_tr_phonems:]
    data_dict['tr']['x'], data_dict['tr']['y'], data_dict['tr']['end'] = \
        extract_seqs(partial_dict['x'][tr_idc], partial_dict['y'][tr_idc],
                     partial_dict['seqlen'][tr_idc], l_data_config['tr'], False, partial_dict['y_seqlen'][tr_idc])

    data_dict['va']['x'], data_dict['va']['y'], data_dict['va']['end'] = \
        extract_seqs(partial_dict['x'][va_idc], partial_dict['y'][va_idc],
                     partial_dict['seqlen'][va_idc], l_data_config['va'], False, partial_dict['y_seqlen'][va_idc])

    data_dict['te'] = {'end': [], 'x': [], 'y': []}
    partial_dict = {'x': [], 'y':[], 'seqlen': [], 'y_seqlen': []}
    n_phonems = 0
    x_max_len = 0
    y_max_len = 0
    for mat in range(6):
        partial_set = loadmat(path + 'seq_test_' + str(mat + 1) + '.mat')
        seqlen = np.squeeze(partial_set['seqlen']).astype(np.int32)
        y_seqlen = np.squeeze(partial_set['seqlen'].astype(np.int32))

        partial_dict['seqlen'] = np.concatenate([partial_dict['seqlen'], seqlen], axis=0)
        partial_dict['y_seqlen'] = np.concatenate([partial_dict['y_seqlen'], y_seqlen], axis=0)
        partial_dict['x'].append(partial_set['x'])
        partial_dict['y'].append(partial_set['y'])
        speaker_lens.append(partial_set['x'].shape[0])
        if x_max_len < partial_set['x'].shape[2]:
            x_max_len = partial_set['x'].shape[2]
        if y_max_len < partial_set['y'].shape[2]:
            y_max_len = partial_set['y'].shape[2]

        n_phonems += x.shape[0]
        if n_phonems > n_te_phonems:
            break
    partial_dict['y_seqlen'] = partial_dict['y_seqlen'].astype(np.int32)
    xs = []
    ys = []
    for x, y in itertools.zip_longest(partial_dict['x'], partial_dict['y']):
        x_shape = x.shape
        xs.append(np.concatenate([np.zeros((x_shape[0], x_shape[1], x_max_len - x_shape[2])), x], axis=2))
        y_shape = y.shape
        ys.append(np.concatenate([np.zeros((y_shape[0], y_shape[1], y_max_len - y_shape[2])), y], axis=2))
    partial_dict['x'] = np.concatenate(xs, axis=0)
    partial_dict['y'] = np.concatenate(ys, axis=0)

    data_dict['te']['x'], data_dict['te']['y'], data_dict['te']['end'] = \
        extract_seqs(partial_dict['x'], partial_dict['x'], partial_dict['seqlen'],
                     l_data_config['te'], False, partial_dict['y_seqlen'])
    print(data_dict['tr']['x'].shape)
    print(data_dict['va']['x'].shape)
    print(data_dict['te']['x'].shape)
    return data_dict


def load_timit(l_data_config):
    if l_data_config['dataset'] == 'timit_s':
        n_te_phonems = 3000
        n_tv_phonems = 30000
    timit_path = '../../datasets/timit/'
    data_dict = {'tr': dict(), 'va': dict(), 'te': dict()}

    partial_dict = {'x': [], 'y': [], 'seqlen': []}
    max_len = 0
    speaker_lens = []
    n_phonems = 0
    for mat_idx in range(16):
        path = timit_path + 'tr' + str(mat_idx + 1) + '.mat'
        partial_set = loadmat(path)
        seqlen = np.squeeze(partial_set['seqlen']).astype(np.int32)
        partial_dict['seqlen'] = np.concatenate([partial_dict['seqlen'], seqlen], axis=0)
        partial_dict['x'].append(partial_set['x'])
        partial_dict['y'].append(partial_set['y'])
        speaker_lens.append(partial_set['x'].shape[0])
        if max_len < partial_set['x'].shape[2]:
            max_len = partial_set['x'].shape[2]
        n_phonems += partial_set['x'].shape[0]
        if n_phonems > n_tv_phonems:
            break
    xs = []
    ys = []
    for x, y in itertools.zip_longest(partial_dict['x'], partial_dict['y']):
        x_shape = x.shape
        xs.append(np.concatenate([np.zeros((x_shape[0], x_shape[1], max_len - x_shape[2])), x], axis=2))
        y_shape = y.shape
        ys.append(np.concatenate([np.zeros((y_shape[0], y_shape[1], max_len - y_shape[2])), y], axis=2))
    partial_dict['x'] = np.concatenate(xs, axis=0)
    partial_dict['y'] = np.concatenate(ys, axis=0)
    permuted_indices = np.arange(n_phonems)
    n_tr_phonems = int(len(permuted_indices) * 0.8)
    tr_speaker_idc = permuted_indices[:n_tr_phonems]
    tr_idc = tr_speaker_idc
    va_idc = permuted_indices[n_tr_phonems:]
    data_dict['tr']['x'], data_dict['tr']['y'], data_dict['tr']['end'] = \
        extract_seqs(partial_dict['x'][tr_idc], partial_dict['y'][tr_idc],
                     partial_dict['seqlen'][tr_idc], l_data_config['tr'], False)

    data_dict['va']['x'], data_dict['va']['y'], data_dict['va']['end'] = \
        extract_seqs(partial_dict['x'][va_idc], partial_dict['y'][va_idc],
                     partial_dict['seqlen'][va_idc], l_data_config['va'], False)

    data_dict['te'] = {'end': [], 'x': [], 'y': []}
    n_phonems = 0
    for mat in range(7):
        partial_set = loadmat(timit_path + 'te' + str(mat + 1) + '.mat')
        seqlen = np.squeeze(partial_set['seqlen']).astype(np.int32)
        x, y, end = extract_seqs(partial_set['x'], partial_set['y'], seqlen, l_data_config['te'], False)
        data_dict['te']['x'].append(x)
        data_dict['te']['y'].append(y)
        data_dict['te']['end'].append(end)
        n_phonems += x.shape[0]
        if n_phonems > n_te_phonems:
            break

    data_dict['te']['x'] = np.concatenate(data_dict['te']['x'], axis=0)
    data_dict['te']['y'] = np.concatenate(data_dict['te']['y'], axis=0)
    data_dict['te']['end'] = np.concatenate(data_dict['te']['end'], axis=0)
    te_idc = np.random.permutation(np.arange(n_phonems))
    data_dict['te']['x'] = data_dict['te']['x'][te_idc]
    data_dict['te']['y'] = data_dict['te']['y'][te_idc]
    data_dict['te']['end'] = data_dict['te']['end'][te_idc]
    print(data_dict['tr']['x'].shape)
    print(data_dict['va']['x'].shape)
    print(data_dict['te']['x'].shape)
    return data_dict


def get_datadict(l_data_config, name):
    keys = ['tr', 'va', 'te']
    data_dict = dict()
    for data_key in keys:
        if data_key in l_data_config.keys():
            if data_key == "tr":
               used_data_key = "te"
            else:
                used_data_key = data_key

            dataset = loadmat(filenames[name + '_' + used_data_key])
            data_dict[data_key] = dict()
            data_dict[data_key]['x'], data_dict[data_key]['y'], data_dict[data_key]['end'] = \
                extract_seqs(dataset['x'], dataset['y'], np.squeeze(dataset['seqlen']).astype(np.int32),
                             l_data_config[data_key], remove_bias=l_data_config['remove_bias'])
            print(data_key)
            print(data_dict[data_key]['x'].shape)
    return data_dict

if __name__ == '__main__':
    #filenames = {'penstroke_tr': '../datasets/mnist_pen_strokes/mps_full1_tr.mat',
    data_config = DataConfig()
    data_config.add_mnist_small()
    load_api_datasets(data_config)
    #for key in ["tr", "va", "te"]:
        #print(key)
        #dataset_dict = loadmat(f"../../datasets/mnist_pen_strokes/mps_all_{key}.mat")
        #dataset_config = DatasetConfig("penstroke", key, 45, True, 500)
        #extract_sequences(dataset_dict, dataset_config)




