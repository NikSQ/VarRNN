import tensorflow as tf

from src.data.loader import load_api_datasets


class Experiment:
    def __init__(self, data_config):
        self.datasets = load_api_datasets(data_config)
        #sample_dataset = self.datasets[list(self.datasets.keys())[0]].batches
        #self.dataset_iter = tf.data.Iterator.from_structure(sample_dataset.output_types, sample_dataset.output_shapes)
        #self.init_iters_ops = {}
        #for key in self.datasets.keys():
            #self.init_iters_ops[key] = self.dataset_iter.make_initializer(self.datasets[key].batches)
        self.dataset_iterators = {}
        for data_key in self.datasets.keys():
            self.dataset_iterators[data_key] = self.datasets[data_key].batches.make_initializable_iterator()

        with tf.variable_scope("this"):
            self.is_training = tf.placeholder(tf.int32)
            self.op = self.is_training + 5
            features, labels, seqlen = self.dataset_iter.get_next()
            self.op = tf.reduce_mean(features)

    def run(self, number):
        with tf.Session() as sess:
            sess.run(self.dataset_iterators["tr"].initializer)
            print(sess.run(self.op, feed_dict={self.is_training: 10}))
