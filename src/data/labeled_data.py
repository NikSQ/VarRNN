import numpy as np
import tensorflow as tf

from src.configuration.constants import DatasetKeys


# Container for handling labeled data on GPU
class GPUDatasets:
    def __init__(self, data_config, datasets):
        with tf.variable_scope('labeled_data'):
            self.batch_idx = tf.placeholder(dtype=tf.int32)  # indexer for minibatch
            self.data_config = data_config
            self.data = dict()

            # the keys refer to the different datasets (training, validation and / or test)
            for data_key in datasets.keys():
                l_data = dict()
                x_shape = datasets[data_key][DatasetKeys.X].shape
                y_shape = datasets[data_key][DatasetKeys.Y].shape

                l_data[DatasetKeys.X_SHAPE] = x_shape
                l_data[DatasetKeys.Y_SHAPE] = y_shape

                l_data[DatasetKeys.X_PLACEHOLDER] = tf.placeholder(dtype=tf.float32, shape=x_shape)
                l_data[DatasetKeys.Y_PLACEHOLDER] = tf.placeholder(dtype=tf.float32, shape=y_shape)

                l_data[DatasetKeys.X] = tf.get_variable(name=f"{DatasetKeys.X}_" + data_key,
                                                        shape=x_shape,
                                                        dtype=tf.float32,
                                                        trainable=False)
                l_data[DatasetKeys.Y] = tf.get_variable(name=f"{DatasetKeys.Y}_" + data_key,
                                                        shape=y_shape,
                                                        dtype=tf.float32,
                                                        trainable=False)
                l_data[DatasetKeys.SEQLEN] = tf.get_variable(name=f"{DatasetKeys.SEQLEN}_" + data_key,
                                                             shape=datasets[data_key][DatasetKeys.SEQLEN].shape,
                                                             dtype=tf.int32,
                                                             trainable=False)
                # Create operation for loading data into GPU
                assign_x = tf.assign(l_data[DatasetKeys.X], l_data[DatasetKeys.X_PLACEHOLDER])
                assign_y = tf.assign(l_data[DatasetKeys.Y], l_data[DatasetKeys.Y_PLACEHOLDER])
                assign_seqlens = tf.assign(l_data[DatasetKeys.SEQLEN], datasets[data_key][DatasetKeys.SEQLEN])
                l_data[DatasetKeys.OP_LOAD] = tf.group(*[assign_x, assign_y, assign_seqlens])

                d_config = data_config.ds_configs[data_key]
                if d_config.minibatch_enabled:
                    batch_size = d_config.minibatch_size

                    # Number of samples is expanded, such that the number of samples is a multiple of the batch size
                    l_data[DatasetKeys.N_MINIBATCHES] = int(np.ceil(float(x_shape[0]) / float(batch_size)))
                    n_samples = batch_size * l_data[DatasetKeys.N_MINIBATCHES]

                    # A shuffled list of sample indices. Iterating over the complete list will be one epoch
                    sample_list = tf.get_variable(name=data_key + f"_{DatasetKeys.SAMPLE_LIST}",
                                                  shape=n_samples,
                                                  dtype=tf.int32,
                                                  trainable=False,
                                                  initializer=tf.constant_initializer(np.arange(x_shape[0]).astype(np.int32),
                                                                                      dtype=np.int32))
                    l_data[DatasetKeys.SAMPLE_LIST] = sample_list
                    samples = tf.tile(tf.random_shuffle(tf.range(x_shape[0])),
                                      multiples=[int(np.ceil(n_samples / x_shape[0]))])
                    l_data[DatasetKeys.OP_SHUFFLE] = tf.assign(sample_list, samples[:n_samples])

                    start_idx = self.batch_idx * batch_size
                    l_data[DatasetKeys.X] = tf.gather(l_data[DatasetKeys.X],
                                                      indices=sample_list[start_idx:start_idx+batch_size])
                    l_data[DatasetKeys.Y] = tf.gather(l_data[DatasetKeys.Y],
                                                      indices=sample_list[start_idx:start_idx+batch_size])
                    l_data[DatasetKeys.SEQLEN] = tf.gather(l_data[DatasetKeys.SEQLEN],
                                                           indices=sample_list[start_idx:start_idx+batch_size])

                    l_data[DatasetKeys.X_SHAPE] = (batch_size,) + l_data[DatasetKeys.X_SHAPE][1:]
                    l_data[DatasetKeys.Y_SHAPE] = (batch_size,) + l_data[DatasetKeys.Y_SHAPE][1:]
                else:
                    #l_data['minibatch_size'] = data_dict[data_key]['x'].shape[0]
                    l_data[DatasetKeys.N_MINIBATCHES] = 1
                    l_data[DatasetKeys.OP_SHUFFLE] = tf.no_op()

                self.data.update({data_key: l_data})
