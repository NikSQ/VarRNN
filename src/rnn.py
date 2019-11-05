import tensorflow as tf
import numpy as np
from src.fc_layer import FCLayer
from src.lstm_layer import LSTMLayer
from src.data.t_metrics import TMetrics
from src.global_variable import get_rnn_config, get_train_config
import copy




class RNN:
    def __init__(self, l_data):
        self.c = None
        self.l_data = l_data
        self.layers = []
        self.train_b_op = None
        self.train_s_op = None
        self.accuracy = None
        self.gradients = None  # Used to find a good value to clip
        self.act_summaries = None
        self.gradient_summaries = None

        with tf.variable_scope('global'):
            self.learning_rate = tf.placeholder(dtype=tf.float32)
            self.tau = tf.placeholder(shape=(1,), dtype=tf.float32)
            self.is_training = tf.placeholder(tf.bool)

        self.rnn_config = get_rnn_config()
        self.train_config = get_train_config()
        self.t_metrics = TMetrics(l_data.l_data_config, l_data, self.is_training, self.tau)
        self.t_metric_summaries = None

        weight_summaries = []
        sample_ops = []
        c_arm_sample_ops = []
        init_ops = []
        for layer_idx, layer_config in enumerate(self.rnn_config['layer_configs']):
            if layer_config['layer_type'] == 'fc':
                layer = FCLayer(layer_idx, self.is_training, self.tau)
            elif layer_config['layer_type'] == 'lstm':
                layer = LSTMLayer(layer_idx, self.is_training, self.tau)
            elif layer_config['layer_type'] == 'input':
                continue
            else:
                raise Exception("{} is not a valid layer type".format(layer_config['layer_type']))
            weight_summaries.append(layer.weights.weight_summaries)
            sample_ops.append(layer.weights.sample_op)
            c_arm_sample_ops.append(layer.weights.c_arm_sample_op)
            init_ops.append(layer.weights.init_op)
            self.layers.append(layer)

        self.sample_op = tf.group(*sample_ops)
        self.c_arm_sample_op = tf.group(*c_arm_sample_ops)
        self.init_op = tf.group(*init_ops)
        self.weight_summaries = tf.summary.merge(weight_summaries)
        self.get_weights_op = self.get_weights_op()

        self.create_b_training_graph('tr')
        for data_key in l_data.data:
            if data_key not in ['tr', 'te']:
                self.create_b_evaluation_graph(data_key)
            self.create_s_evaluation_graph(data_key)
            self.get_activation_summaries()

    def get_activation_summaries(self):
        for layer in self.layers:
            with tf.variable_scope(layer.layer_config['var_scope']):
                for act_key in layer.acts.keys():
                    if self.act_summaries is None:
                        if len(act_key) == 1:
                            self.act_summaries = tf.summary.histogram(act_key, tf.clip_by_value(layer.acts[act_key], -20, 20))
                        else:
                            self.act_summaries = tf.summary.histogram(act_key, tf.clip_by_value(layer.acts[act_key], -7, 7))
                    else:
                        if len(act_key) == 1:
                            self.act_summaries = tf.summary.merge([self.act_summaries,
                                                               tf.summary.histogram(act_key, tf.clip_by_value(layer.acts[act_key], -20, 20))])
                        else:
                            self.act_summaries = tf.summary.merge([self.act_summaries,
                                                                   tf.summary.histogram(act_key, tf.clip_by_value(layer.acts[act_key], -7, 7))])

    # TODO: Make predictions based on predictive distribution rather than on mode
    def create_rnn_graph(self, data_key, mod_rnn_config, bayesian=True):
        x = self.l_data.data[data_key]['x']
        y = self.l_data.data[data_key]['y']

        x_shape = self.l_data.data[data_key]['x_shape']
        y_shape = self.l_data.data[data_key]['y_shape']

        # Elements of the list will be the means and variances of the output sequence
        m_outputs = []
        m2_outputs = []
        v_outputs = []

        # This captures the seq_idx from which the output will be computed
        output_idcs = self.l_data.data[data_key]['end']

        # Create graph by connecting the appropriate layers unrolled in time
        for time_idx in range(x_shape[2]):
            m = x[:, :, time_idx]  # Mean of input to network at time seq_idx
            v = tf.fill(tf.shape(m), 0.)  # Variance of input to network at time seq_idx

            for layer_idx, layer in enumerate(self.layers, 1):
                if bayesian is False:
                    m = layer.create_var_fp(m, time_idx)
                elif 'pfp' in self.train_config['algorithm']:
                    m, v = layer.create_pfp(m, v, mod_rnn_config['layer_configs'][layer_idx])
                elif 'l_reparam' in self.train_config['algorithm']:
                    m = layer.create_l_sampling_pass(m, mod_rnn_config['layer_configs'][layer_idx], time_idx)
                elif 'c_reparam' in self.train_config['algorithm'] or 'c_ar' in self.train_config['algorithm'] or \
                        'c_arm' in self.train_config['algorithm'] or 'log_der' in self.train_config['algorithm']:
                    m = layer.create_g_sampling_pass(m, mod_rnn_config['layer_configs'][layer_idx], time_idx, data_key=data_key)
                else:
                    raise Exception('Training type not understood')

            m_outputs.append(m)
            if 'pfp' in self.train_config['algorithm']:
                v_outputs.append(v)

        for time_idx in range(x_shape[2]):
            m = x[:, :, time_idx]  # Mean of input to network at time seq_idx
            if bayesian is True and 'c_arm' in self.train_config['algorithm'] and data_key == 'tr':
                for layer_idx, layer in enumerate(self.layers, 1):
                    m = layer.create_g_sampling_pass(m, mod_rnn_config['layer_configs'][layer_idx], time_idx,
                                                     second_arm_pass=True, data_key=data_key)
                m2_outputs.append(m)

        m_outputs = tf.stack(m_outputs, axis=1)
        gather_idcs = tf.stack([tf.range(y_shape[0]), output_idcs], axis=1)
        m_outputs = tf.gather_nd(m_outputs, gather_idcs)

        # Process output of non bayesian network
        if bayesian is False:
            output = tf.cast(m_outputs, dtype=tf.float64)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y, dim=1))
            prediction = tf.argmax(output, axis=1)
            acc = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y, axis=1)), dtype=tf.float32))

            self.t_metrics.add_s_vars(data_key + '_s', self.sample_op, loss, acc)
            if self.t_metric_summaries is not None:
                self.t_metric_summaries = tf.summary.merge([self.t_metric_summaries,
                                                            tf.summary.scalar(data_key + '_s_loss', loss),
                                                            tf.summary.scalar(data_key + '_s_acc', acc)])
            else:
                self.t_metric_summaries = tf.summary.merge([tf.summary.scalar(data_key + '_s_loss', loss),
                                                            tf.summary.scalar(data_key + '_s_acc', acc)])
            return loss, acc

        # Process output of bayesian network
        else:
            if 'pfp' in self.train_config['algorithm']:
                m_output = tf.cast(dtype=tf.float64)
                v_output = tf.cast(tf.stack(v_outputs, axis=0), dtype=tf.float64)
                smax = tf.nn.softmax(logits=m_output, axis=1)
                t = tf.argmax(y, axis=1)
                batch_range = np.arange(y_shape[0])
                seq_range = np.arange(y_shape[2])
                seqq, batchh = np.meshgrid(seq_range, batch_range)
                g_indices = tf.concat([np.expand_dims(batchh, axis=2), tf.expand_dims(t, axis=2),
                                      np.expand_dims(seqq, axis=2)], axis=2)
                elogl = tf.reduce_mean(tf.log(np.finfo(np.float64).eps + tf.gather_nd(smax, g_indices))) - \
                    0.5 * tf.reduce_mean(tf.reduce_sum(tf.multiply(v_output, tf.multiply(smax, 1 - smax)), axis=1))
                kl = 0
                if self.rnn_config['data_multiplier'] is not None:
                    for layer in self.layers:
                        kl += layer.weights.get_kl_loss()
                    kl /= (self.rnn_config['data_multiplier'] *
                           self.l_data.data[data_key]['minibatch_size'] *
                           self.l_data.data[data_key]['n_minibatches'])
                    vfe = kl - elogl
                else:
                    kl = tf.zeros(())
                    vfe = - elogl

                prediction = tf.argmax(smax, axis=1)
                acc = tf.reduce_mean(tf.cast(tf.equal(prediction, t), dtype=tf.float32))
            elif 'l_reparam' in self.train_config['algorithm'] or 'c_reparam' in self.train_config['algorithm'] or data_key != 'tr':
                smax = tf.nn.softmax(logits=m_outputs, axis=1)
                t = tf.argmax(y, axis=1)
                elogl = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=m_outputs, labels=y, dim=1))

                kl = 0
                if self.rnn_config['data_multiplier'] is not None:
                    for layer in self.layers:
                        kl += layer.weights.get_kl_loss()
                    kl /= (self.rnn_config['data_multiplier'] *
                           self.l_data.l_data_config[data_key]['minibatch_size'] *
                           self.l_data.data[data_key]['n_minibatches'])
                    vfe = kl - elogl
                else:
                    kl = tf.zeros(())
                    vfe = -elogl

                prediction = tf.argmax(smax, axis=1)
                acc = tf.reduce_mean(tf.cast(tf.equal(prediction, t), dtype=tf.float32))
            elif 'c_arm' in self.train_config['algorithm'] and data_key == 'tr':
                m2_outputs = tf.stack(m2_outputs, axis=1)
                m2_outputs = tf.gather_nd(m2_outputs, gather_idcs)

                return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=m_outputs, labels=y, dim=1)) + \
                        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=m2_outputs, labels=y, dim=1))
            elif ('c_ar' in self.train_config['algorithm'] or 'log_der' in self.train_config['algorithm']) and data_key == 'tr':
                return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=m_outputs, labels=y, dim=1))*2

            self.t_metrics.add_b_vars(data_key + '_b', vfe, kl, elogl, acc)

            if self.t_metric_summaries is not None:
                self.t_metric_summaries = tf.summary.merge([self.t_metric_summaries,
                                                            tf.summary.scalar(data_key + '_b_vfe', vfe),
                                                            tf.summary.scalar(data_key + '_b_kl', kl),
                                                            tf.summary.scalar(data_key + '_b_elogl', elogl),
                                                            tf.summary.scalar(data_key + '_b_acc', acc)])
            else:
                self.t_metric_summaries = tf.summary.merge([tf.summary.scalar(data_key + '_b_vfe', vfe),
                                                            tf.summary.scalar(data_key + '_b_kl', kl),
                                                            tf.summary.scalar(data_key + '_b_elogl', elogl),
                                                            tf.summary.scalar(data_key + '_b_acc', acc)])
            return vfe, kl, elogl, acc

    # Creates Bayesian graph for training and the operations used for training.
    def create_b_training_graph(self, key):
        with tf.variable_scope(key + '_b'):
            if 'c_ar' in self.train_config['algorithm'] or 'c_arm' in self.train_config['algorithm'] or 'log_der' in self.train_config['algorithm']:
                loss = self.create_rnn_graph(key, self.rnn_config)
                layer_samples = dict()
                variables = []
                for var in tf.trainable_variables():
                    if 'sb' in var.name:
                        variables.append(var)

                for layer in self.layers:
                    var_scope = layer.layer_config['var_scope']
                    if 'log_der' in self.train_config['algorithm']:
                        layer_samples[var_scope] = layer.weights.logder_derivs
                    else:
                        layer_samples[var_scope] = layer.weights.arm_samples

                grads = []
                grad_vars = []
                for var in variables:
                    for var_scope in layer_samples.keys():
                        if var_scope in var.name:
                            for var_key in layer_samples[var_scope].keys():
                                if var_key + '_sb' == var.name[var.name.index('/')+1:-2]:
                                    if 'log_der' in self.train_config['algorithm']:
                                        grads.append(loss * layer_samples[var_scope][var_key])
                                    else:
                                        grads.append(loss * (layer_samples[var_scope][var_key] - .5))
                                    grad_vars.append(var)

                self.gradients = list(zip(grads, grad_vars))
                self.gradient_ph = []
                self.vars = grad_vars
                gradient_summaries = []
                for grad, var in zip(grads, grad_vars):
                    if grad is not None:
                        gradient_summaries.append(tf.summary.histogram('g_' + var.name[var.name.index('/')+1:-2], grad))
                        self.gradient_ph.append(tf.placeholder(shape=grad.shape, dtype=tf.float32, name='gradient_ph_' + var.name[var.name.index('/')+1:-2]))
                    else:
                        self.gradient_ph.append(tf.placeholder(shape=None, dtype=tf.float32, name='gradient_ph_'  + var.name[var.name.index('/')+1:-2]))
                learning_rate = tf.get_variable(name='lr', shape=(), dtype=tf.float32)
                self.assign_learning_rate = tf.assign(learning_rate, self.learning_rate)
                self.train_b_op = tf.train.AdamOptimizer(learning_rate=learning_rate).apply_gradients(zip(self.gradient_ph, self.vars))
                self.gradient_summaries = tf.summary.merge(gradient_summaries)
                return

            vfe, kl, elogl, acc = self.create_rnn_graph(key, self.rnn_config)

            dir_reg = 0
            var_reg = 0
            ent_reg = 0
            for layer in self.layers:
                if self.train_config['var_reg'] != 0:
                    var_reg += layer.weights.get_var_reg()
                if self.train_config['dir_reg'] != 0:
                    dir_reg += layer.weights.get_dir_reg()
                if self.train_config['ent_reg'] != 0:
                    ent_reg += layer.weights.get_entropy_reg()

            var_reg *= self.train_config['var_reg']
            dir_reg *= self.train_config['dir_reg']
            ent_reg *= self.train_config['ent_reg']
            if type(self.train_config['learning_rate']) is list:
                opt1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate[0])
                opt2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate[1])
                opt3 = tf.train.AdamOptimizer(learning_rate=self.learning_rate[2])
            else:
                opt1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                opt2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                opt3 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            vars1 = []
            vars2 = []
            vars3 = []
            for var in tf.trainable_variables():
                if var.name[:var.name.index('/')] == 'lstm_0':
                    vars1.append(var)
                if var.name[:var.name.index('/')] == 'lstm_1':
                    vars2.append(var)
                if var.name[:var.name.index('/')] == 'output_layer':
                    vars3.append(var)

            self.gradients = tf.gradients(vfe + dir_reg + var_reg + ent_reg, vars1 + vars2 + vars3)

            gradient_summaries = []
            for gradient, var in zip(self.gradients, vars1+vars2+vars3):
                if gradient is not None:
                    gradient_summaries.append(tf.summary.histogram('g_' + var.name, gradient))
            self.gradient_summaries = tf.summary.merge(gradient_summaries)

            clipped_gradients = [grad if grad is None else
                                 tf.clip_by_value(grad, -self.rnn_config['gradient_clip_value'],
                                                  self.rnn_config['gradient_clip_value'])
                                 for grad in self.gradients]

            grads1 = clipped_gradients[:len(vars1)]
            grads2 = clipped_gradients[len(vars1):len(vars1)+len(vars2)]
            grads3 = clipped_gradients[len(vars1) + len(vars2):]
            train_ops = []

            if len(vars1) != 0:
                train_ops.append(opt1.apply_gradients(zip(grads1, vars1)))
            if len(vars2) != 0:
                train_ops.append(opt2.apply_gradients(zip(grads2, vars2)))
            if len(vars3) != 0:
                train_ops.append(opt3.apply_gradients(zip(grads3, vars3)))
            self.train_b_op = tf.group(*train_ops)

    # Creates non-Bayesian graph for training the RNN
    def create_s_training_graph(self, key):
        with tf.variable_scope(key + '_s'):
            loss, accuracy = self.create_rnn_graph(key, None, bayesian=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            reg = 0
            for layer in self.layers:
                reg += layer.weights.get_pretraining_reg()
            reg *= self.train_config['reg']
            gradients = optimizer.compute_gradients(loss + reg)

            clipped_gradients = [(grad, var) if grad is None else
                                 (tf.clip_by_value(grad, -self.rnn_config['gradient_clip_value'],
                                                   self.rnn_config['gradient_clip_value']), var)
                                 for grad, var in gradients]
            self.train_s_op = optimizer.apply_gradients(clipped_gradients)

    # Evaluation graphs are not used for training but only for evaluation. A modified RNN config file is used which
    # overwrites the one used for training. Can be used to modify the forward pass to turn off dropout while validating
    # and testing for example.
    def create_b_evaluation_graph(self, key):
        with tf.variable_scope(key + '_b'):
            mod_rnn_config = copy.deepcopy(self.rnn_config)
            for layer_config in mod_rnn_config['layer_configs']:
                if 'regularization' in layer_config:
                    layer_config['regularization']['mode'] = None

            self.create_rnn_graph(key, mod_rnn_config)

    def create_s_evaluation_graph(self, data_key):
        with tf.variable_scope(data_key + '_s'):
            self.create_rnn_graph(data_key, None, bayesian=False)

    def get_weights_op(self):
        weight_probs = {}
        for layer in self.layers:
            weight_probs[layer.layer_config['var_scope']] = layer.weights.get_weight_probs()
        return weight_probs



