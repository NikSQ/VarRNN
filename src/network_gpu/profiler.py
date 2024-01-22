# This class implements the profiling from tensorflow

import tensorflow as tf
from tensorflow.python.client import timeline


class Profiler:
    def __init__(self, info_config):
        self.enabled = info_config.profiling_enabled
        self.path = info_config.profiling_path
        self.options = None
        self.run_metadata = None
        self.traces = []

        self.initialize()

    def initialize(self):
        if self.enabled:
            self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        else:
            self.options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        self.run_metadata = tf.RunMetadata()

    def add_trace(self):
        if self.enabled:
            self.traces.append(timeline.Timeline(self.run_metadata.step_stats).generate_chrome_trace_format())

    def conclude_training(self, epoch):
        if self.enabled:
            for trace_idx, trace in enumerate(self.traces):
                path = self.path + '_' + str(epoch) + '_' + str(trace_idx)
                with open(path + 'training.json', 'w') as f:
                    f.write(trace)