import tensorflow as tf
import numpy as np
from copy import deepcopy

from src.global_variable import get_nn_config, get_train_config
from src.network_gpu.ff_layer import FFLayer
from src.network_gpu.lstm_layer import LSTMLayer
from src.data.t_metrics import  TMetrics

from src.configuration.constants import NetworkC, DatasetKeys, AlgorithmC, VarNames

# NOTE Did not include varscope for sampling and bayesian pass
class RNN:
    def __init__(self, datasets):
        self.datasets = datasets

        with tf.variable_scope("global"):
            self.learning_rate = tf.get_variable(name="learning_rate", shape=(), dtype=tf.float32, trainable=False)
            self.tau = tf.get_variable(name="tau", shape=(), dtype=tf.float32, trainable=False)
            self.is_training = tf.get_variable(name="is_training", shape=(), dtype=tf.bool, trainable=False)

        self.rnn_config = get_nn_config()
        self.train_config = get_train_config()
        self.data_config = self.datasets.data_config

        weight_summaries = []
        sample_ops = []
        c_arm_sample_ops = []
        init_ops = []
        self.layers = []
        for layer_idx, layer_config in enumerate(self.rnn_config.layer_configs):
            if layer_config.layer_type == NetworkC.FEED_FORWARD_LAYER:
                layer = FFLayer(layer_idx=layer_idx,
                                is_training=self.is_training,
                                tau=self.tau)

            elif layer_config.layer_type == NetworkC.LSTM_LAYER:
                layer = LSTMLayer(layer_idx=layer_idx,
                                  is_training=self.is_training,
                                  tau=self.tau)

            elif layer_config.layer_type == NetworkC.INPUT_LAYER:
                continue
            else:
                raise Exception("Layer type " + layer_config.layer_type + " not understood")

            weight_summaries.append(layer.weights.weight_summaries)
            sample_ops.append(layer.weights.sample_op)
            c_arm_sample_ops.append(layer.weights.c_arm_sample_op)
            init_ops.append(layer.weights.init_op)
            self.layers.append(layer)

        self.sample_op = tf.group(*sample_ops)
        self.c_arm_sample_op = tf.group(*c_arm_sample_ops)
        self.init_op = tf.group(*init_ops)
        self.weight_summaries = tf.summary.merge(weight_summaries)
        #self.get_weights_op = self.get_weights_op()

        self.t_metrics = TMetrics(data_config=self.data_config,
                                  gpu_dataset=self.datasets,
                                  is_training=self.is_training,
                                  tau=self.tau)
        self.train_b_op = None
        self.train_s_op = None
        self.grad_vars = None
        self.gradient_summaries = None

        self.create_bayesian_training_graph(DatasetKeys.TR_SET)
        if self.train_config.algorithm in [AlgorithmC.LOG_DERIVATIVE, AlgorithmC.AR, AlgorithmC.ARM]:
            self.create_bayesian_evaluation_graph(DatasetKeys.TR_SET)
        for data_key in self.datasets.data.keys():
            if data_key not in [DatasetKeys.TR_SET]:
                self.create_bayesian_evaluation_graph(data_key)
            self.create_sampling_evaluation_graph(data_key)

    def unfold_rnn_layer(self, bayesian, data_key, layer, layer_idx, layer_input, x_shape, mod_nn_config,
                         reverse=False, second_arm_pass=False, annotations=False):
        timesteps = np.arange(x_shape[2])
        layer_outputs = []
        layer_states = []

        for timestep in timesteps:
            input_m = layer_input[timestep]
            input_v = tf.fill(tf.shape(input_m), 0.)
            do_initialize = timestep == 0

            if not bayesian:
                input_m, state = layer.create_var_fp(x=input_m,
                                                     do_initialize=do_initialize,
                                                     timestep=timestep)
            elif self.train_config.algorithm == AlgorithmC.LOCAL_REPARAMETRIZATION:
                input_m, state = layer.create_l_sampling_pass(x=input_m,
                                                              do_initialize=do_initialize,
                                                              timestep=timestep,
                                                              mod_layer_config=mod_nn_config.layer_configs[layer_idx])
            elif self.train_config.algorithm in [AlgorithmC.REPARAMETRIZATION, AlgorithmC.AR,
                                                 AlgorithmC.ARM, AlgorithmC.LOG_DERIVATIVE]:
                input_m, state = layer.create_sampling_pass(x=input_m,
                                                            timestep=timestep,
                                                            do_initialize=do_initialize,
                                                            data_key=data_key,
                                                            mod_layer_config=mod_nn_config.layer_configs[layer_idx],
                                                            second_arm_pass=second_arm_pass)
            else:
                raise Exception("Training algorithm " + self.train_config.algorithm + " not implemented")

            layer_outputs.append(input_m)
            layer_states.append(state)
        return layer_outputs, layer_states

    def unfold_rnn(self, bayesian, data_key, x, x_shape, mod_nn_config, second_arm_pass=False):
        layer_input = tf.transpose(x, perm=[2, 0, 1]) # new shape (n_timesteps, batch_size, n_features)
        for layer_idx, layer in enumerate(self.layers, 1):
            layer_input, _ = self.unfold_rnn_layer(bayesian=bayesian,
                                                   data_key=data_key,
                                                   layer=layer,
                                                   layer_idx=layer_idx,
                                                   layer_input=layer_input,
                                                   x_shape=x_shape,
                                                   mod_nn_config=mod_nn_config,
                                                   reverse=False,
                                                   second_arm_pass=second_arm_pass)
        return layer_input

    # TODO: Make predictions based on predictive distribution as well
    # TODO: Make sure extraction idcs work correctly
    def create_rnn_graph(self, data_key, mod_nn_config, bayesian=True, is_training_graph=True):
        dataset = self.datasets.data[data_key]

        x = dataset[DatasetKeys.X]
        y = dataset[DatasetKeys.Y]
        x_shape = dataset[DatasetKeys.X_SHAPE] # (batch_size, n_features, n_timesteps)
        y_shape = dataset[DatasetKeys.Y_SHAPE] # (batch_size, n_categories)

        seq_lens = dataset[DatasetKeys.SEQLEN]

        # =========================================================
        # Pass data through network to gather the output
        # =========================================================
        output_m = self.unfold_rnn(bayesian=bayesian,
                                   data_key=data_key,
                                   x=x,
                                   x_shape=x_shape,
                                   mod_nn_config=mod_nn_config,
                                   second_arm_pass=False)

        output_m = tf.stack(output_m, axis=-1)
        one_hot = tf.one_hot(seq_lens, x_shape[2])[:, tf.newaxis, :]
        output_m = tf.multiply(output_m, one_hot)
        output_m = tf.reduce_sum(output_m, axis=2)

        if bayesian and self.train_config.algorithm == AlgorithmC.ARM and is_training_graph:
            output_m_2 = self.unfold_rnn(bayesian=bayesian,
                                         data_key=data_key,
                                         x=x,
                                         x_shape=x_shape,
                                         mod_nn_config=mod_nn_config,
                                         second_arm_pass=True)
            output_m_2 = tf.stack(output_m_2, axis=-1)
            output_m_2 = tf.multiply(output_m_2, one_hot)
            output_m_2 = tf.reduce_sum(output_m_2, axis=2)
        # =========================================================
        # Process output of network
        # =========================================================
        if not bayesian:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_m, labels=y, dim=1))
            prediction = tf.argmax(output_m, axis=1)
            acc = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y, axis=1)), dtype=tf.float32))
            self.t_metrics.add_s_vars(process_key=data_key + "_s", sample_op=self.sample_op, loss_op=loss, accs_op=acc)
            return loss, acc

        elif self.train_config.algorithm in [AlgorithmC.LOCAL_REPARAMETRIZATION, AlgorithmC.REPARAMETRIZATION] or \
                not is_training_graph:
            smax = tf.nn.softmax(logits=output_m, axis=1)
            t = tf.argmax(y, axis=1)
            elogl = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_m, labels=y, dim=1))
            kl = 0
            if self.train_config.data_multiplier is not None:
                # TODO check whether kl loss has right scaling
                for layer in self.layers:
                    kl += layer.weights.get_kl_loss()
                kl /= (self.train_config.data_multiplier *
                       self.datasets.data_config.ds_configs[data_key].minibatch_size *
                       self.datasets.data[data_key][DatasetKeys.N_MINIBATCHES])
                vfe = kl - elogl
            else:
                kl = tf.zeros(())
                vfe = -elogl
            prediction = tf.argmax(smax, axis=1)
            acc = tf.reduce_mean(tf.cast(tf.equal(prediction, t), dtype=tf.float32))

        elif self.train_config.algorithm == AlgorithmC.ARM:
            # 0.5 * [f(1_u<sig(phi)) - f(1_u>sig(-phi))
            # 0.5 * [f(1_u<p1) - f(1_u>pn1)
            return .5 * (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_m, labels=y, dim=1)) - \
                    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_m_2, labels=y, dim=1)))

        elif self.train_config.algorithm in [AlgorithmC.AR, AlgorithmC.LOG_DERIVATIVE]:
            # In AR case:
            # f(1_u<sig(phi))
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_m, labels=y, dim=1))

        self.t_metrics.add_b_vars(process_key=data_key + "_b", vfe_op=vfe, kl_op=kl, elogl_op=elogl, accs_op=acc)
        return vfe, kl, elogl, acc

    # Creates a Bayesian graph for training in probabilistic manner
    def create_bayesian_training_graph(self, key):
        with tf.variable_scope(key + '_b'):
            if self.train_config.algorithm in [AlgorithmC.AR, AlgorithmC.ARM, AlgorithmC.LOG_DERIVATIVE]:
                loss = self.create_rnn_graph(key, self.rnn_config)

                layer_samples = dict()
                variables = []
                acceptable_vars = [VarNames.SIGMOID_A,
                                   VarNames.SIGMOID_B,
                                   VarNames.LOGITS_NEG,
                                   VarNames.LOGITS_ZER,
                                   VarNames.LOGITS_POS]
                # TODO: Check discrepancy, previously there was only 'sb'
                for var in tf.trainable_variables():
                    for acceptable_var in acceptable_vars:
                        if acceptable_var in var.name:
                            variables.append(var)
                            print("DEBUG")
                            print("Added var: " + var.name)
                            break

                for layer in self.layers:
                    var_scope = layer.layer_config.var_scope
                    if self.train_config.algorithm in [AlgorithmC.LOG_DERIVATIVE]:
                        layer_samples[var_scope] = layer.weights.logder_derivs
                    else:
                        layer_samples[var_scope] = layer.weights.arm_samples

                grads = []
                vars = []
                for var in variables:
                    for var_scope in layer_samples.keys():
                        if var_scope in var.name:
                            for var_key in layer_samples[var_scope].keys():
                                if var_key + '_sb' == var.name[var.name.index('/') + 1:-2]:
                                    if AlgorithmC.LOG_DERIVATIVE in self.train_config.algorithm:
                                        grads.append(loss * layer_samples[var_scope][var_key])
                                    else:
                                        grads.append(loss * (1 - 2 * layer_samples[var_scope][var_key]))
                                    vars.append(var)

                self.gradient_ph = []
                self.vars = vars
                self.gradients = grads
                self.grad_vars = list(zip(grads, vars))
                gradient_summaries = []
                for grad, var in zip(grads, vars):
                    if grad is not None:
                        gradient_summaries.append(
                            tf.summary.histogram('g_' + var.name[var.name.index('/') + 1:-2], grad))
                        self.gradient_ph.append(tf.placeholder(shape=grad.shape, dtype=tf.float32,
                                                               name='gradient_ph_' + var.name[var.name.index(
                                                                   '/') + 1:-2]))
                    else:
                        self.gradient_ph.append(tf.placeholder(shape=None, dtype=tf.float32,
                                                               name='gradient_ph_' + var.name[var.name.index(
                                                                   '/') + 1:-2]))
                learning_rate = tf.get_variable(name='lr', shape=(), dtype=tf.float32)
                self.assign_learning_rate = tf.assign(learning_rate, self.learning_rate)
                self.train_b_op = tf.train.AdamOptimizer(learning_rate=learning_rate).apply_gradients(
                    zip(self.gradient_ph, self.vars))
                self.gradient_summaries = tf.summary.merge(gradient_summaries)
                return

            # Create RNN graph
            vfe, kl, elogl, acc = self.create_rnn_graph(key, self.rnn_config)

            dir_reg = 0
            var_reg = 0
            ent_reg = 0
            for layer in self.layers:
                if self.train_config.variance_regularization != 0:
                    var_reg += layer.weights.get_var_reg()
                if self.train_config.dirichlet_regularization != 0:
                    dir_reg += layer.weights.get_dir_reg()
                if self.train_config.entropy_regularization != 0:
                    ent_reg += layer.weights.get_entropy_reg()

            var_reg *= self.train_config.variance_regularization
            dir_reg *= self.train_config.dirichlet_regularization
            ent_reg *= self.train_config.entropy_regularization

            """
            if type(self.train_config.learning_rate) is list:
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

            gradients = tf.gradients(vfe + dir_reg + var_reg + ent_reg, vars1 + vars2 + vars3)
            self.gradients = zip(gradients, vars1 + vars2 + vars3)

            # Create summary of gradients for Tensorboard visualizations
            gradient_summaries = []
            for gradient, var in zip(gradients, vars1 + vars2 + vars3):
                if gradient is not None:
                    gradient_summaries.append(tf.summary.histogram('g_' + var.name, gradient))
            self.gradient_summaries = tf.summary.merge(gradient_summaries)

            # Add gradient clipping if desired
            if self.train_config.gradient_clipping_enabled:
                clipped_gradients = [grad if grad is None else
                                     tf.clip_by_value(grad, -self.train_config.gradient_clip_value,
                                                      self.train_config.gradient_clip_value)
                                     for grad in gradients]
            else:
                clipped_gradients = gradients

            grads1 = clipped_gradients[:len(vars1)]
            grads2 = clipped_gradients[len(vars1):len(vars1) + len(vars2)]
            grads3 = clipped_gradients[len(vars1) + len(vars2):]
            train_ops = []

            if len(vars1) != 0:
                train_ops.append(opt1.apply_gradients(zip(grads1, vars1)))
            if len(vars2) != 0:
                train_ops.append(opt2.apply_gradients(zip(grads2, vars2)))
            if len(vars3) != 0:
                train_ops.append(opt3.apply_gradients(zip(grads3, vars3)))
            """

            # Collect all trainable variables
            trainable_vars = []
            for var in tf.trainable_variables():
                if "global" not in var.name:
                    trainable_vars.append(var)

            # Compute gradientens
            gradients = tf.gradients(vfe + dir_reg + var_reg + ent_reg, trainable_vars)

            # Clip gradients (optional)
            if self.train_config.gradient_clipping_enabled:
                gradients = [grad if grad is None else
                             tf.clip_by_value(grad, -self.train_config.gradient_clip_value,
                                              self.train_config.gradient_clip_value)
                             for grad in gradients]

            # Set grad vars
            self.grad_vars = list(zip(gradients, trainable_vars))
            self.gradients = gradients
            self.vars = trainable_vars

            # Set up gradient summaries for tensorboard
            gradient_summaries = []
            for gradient, var in self.grad_vars:
                if gradient is not None:
                    gradient_summaries.append(tf.summary.histogram('g_' + var.name, gradient))
                    break
            self.gradient_summaries = tf.summary.merge(gradient_summaries)

            # Optimize
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_b_op = opt.apply_gradients(self.grad_vars)

    # Creates non-Bayesian graph for training the RNN
    def create_sampling_training_graph(self, key):
        with tf.variable_scope(key + '_s'):
            loss, accuracy = self.create_rnn_graph(key, None, bayesian=False)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            reg = 0
            for layer in self.layers:
                reg += layer.weights.get_pretraining_reg()
            reg *= self.train_config.pretraining_reg
            gradients = optimizer.compute_gradients(loss + reg)

            clipped_gradients = [(grad, var) if grad is None else
                                 (tf.clip_by_value(grad, -self.rnn_config['gradient_clip_value'],
                                                   self.rnn_config['gradient_clip_value']), var)
                                 for grad, var in gradients]
            self.train_s_op = optimizer.apply_gradients(clipped_gradients)

    # Evaluation graphs are not used for training but only for evaluation. A modified RNN config file is used which
    # overwrites the one used for training. Can be used to modify the forward pass to turn off dropout while validating
    # and testing for example.
    def create_bayesian_evaluation_graph(self, key):
        with tf.variable_scope(key + '_b'):
            self.create_rnn_graph(key, mod_nn_config=self.rnn_config, is_training_graph=False)

    def create_sampling_evaluation_graph(self, data_key):
        with tf.variable_scope(data_key + '_s'):
            self.create_rnn_graph(data_key, mod_nn_config=self.rnn_config, bayesian=False, is_training_graph=False)

# Here we summarize
