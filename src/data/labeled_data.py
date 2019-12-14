import numpy as np
import tensorflow as tf


# Container for handling labeled data on GPU
class LabeledData:
    def __init__(self, l_data_config, data_dict):
        with tf.variable_scope('labeled_data'):
            self.batch_idx = tf.placeholder(dtype=tf.int32)  # indexer for minibatch
            self.l_data_config = l_data_config
            self.data = dict()

            # the keys refer to the different datasets (training, validation and / or test)
            for data_key in data_dict.keys():
                l_data = dict()
                l_data['x_shape'] = data_dict[data_key]['x'].shape
                l_data['y_shape'] = data_dict[data_key]['y'].shape
                l_data['x_ph'] = tf.placeholder(dtype=tf.float32, shape=l_data['x_shape'])
                l_data['y_ph'] = tf.placeholder(dtype=tf.float32, shape=l_data['y_shape'])
                l_data['x'] = tf.get_variable(name='x_' + data_key, shape=l_data['x_shape'], dtype=tf.float32,
                                                   trainable=False)
                l_data['y'] = tf.get_variable(name='y_' + data_key, shape=l_data['y_shape'], dtype=tf.float32,
                                                   trainable=False)
                l_data['end'] = tf.get_variable(name='end_' + data_key, shape=data_dict[data_key]['end'].shape,
                                                   dtype=tf.int32, trainable=False)
                assign_x = tf.assign(l_data['x'], l_data['x_ph'])
                assign_y = tf.assign(l_data['y'], l_data['y_ph'])
                assign_seqlens = tf.assign(l_data['end'], data_dict[data_key]['end'])
                l_data['load'] = tf.group(*[assign_x, assign_y, assign_seqlens])

                if l_data_config[data_key]['minibatch_enabled']:
                    batch_size = l_data_config[data_key]['minibatch_size']

                    # Number of samples is expanded, such that the number of samples is a multiple of the batch size
                    l_data['n_minibatches'] = int(np.ceil(float(l_data['x_shape'][0]) / float(batch_size)))
                    n_samples = batch_size * l_data['n_minibatches']

                    # A shuffled list of sample indices. Iterating over the complete list will be one epoch
                    sample_list = tf.get_variable(name=data_key + '_sample_list', shape=n_samples, dtype=tf.int32,
                                                  trainable=False, initializer=tf.constant_initializer(np.arange(n_samples).astype(np.int32), dtype=np.int32))
                    l_data['list'] = sample_list
                    samples = tf.tile(tf.random_shuffle(tf.range(l_data['x_shape'][0])),
                                      multiples=[int(np.ceil(n_samples / l_data['x_shape'][0]))])
                    l_data['shuffle'] = tf.assign(sample_list, samples[:n_samples])

                    l_data['x'] = tf.gather(l_data['x'], indices=sample_list[self.batch_idx:self.batch_idx+batch_size])
                    l_data['y'] = tf.gather(l_data['y'], indices=sample_list[self.batch_idx:self.batch_idx+batch_size])
                    l_data['end'] = tf.gather(l_data['end'], indices=sample_list[self.batch_idx:self.batch_idx+batch_size])
                    l_data['x_shape'] = (l_data_config[data_key]['minibatch_size'],) + l_data['x_shape'][1:]
                    l_data['y_shape'] = (l_data_config[data_key]['minibatch_size'],) + l_data['y_shape'][1:]
                else:
                    l_data['minibatch_size'] = data_dict[data_key]['x'].shape[0]
                    l_data['n_minibatches'] = 1
                    l_data['shuffle'] = tf.no_op()

                self.data.update({data_key: l_data})
