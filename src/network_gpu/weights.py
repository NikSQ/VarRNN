import tensorflow as tf
#import tensorflow_probability as tfp
import numpy as np
import copy

from src.global_variable import get_train_config
from src.configuration.constants import WeightC, AlgorithmC, VarNames, ActivationFunctionsC


# Returns a tf variable initializer, dependent on initializer_type which is either a scalar or an understood string
def get_initializer(initializer_type, shape):
    if type(initializer_type) in [int, float]:
        init_vals = np.ones(shape) * initializer_type
    elif initializer_type[0] == WeightC.XAVIER_INIT:
        init_vals = np.random.randn(shape[0], shape[1]) * np.sqrt(2/sum(shape))
    elif initializer_type[0] == WeightC.BINARY_INIT:
        init_vals = 2 * np.random.binomial(n=1, p=0.5, size=shape) - 1
    else:
        raise Exception("{} is not a valid weight initialization".format(initializer_type))
    return tf.constant_initializer(init_vals)


def get_dirichlet_init(initializer_type, shape, parametrization):
    probs = np.random.dirichlet(initializer_type[1], size=shape)
    probs = probs.astype(dtype=np.float32)
    probs = create_stable_probs(probs)
    if parametrization == WeightC.LOGIT:
        return get_logit_params_from_probs(probs)
    elif parametrization == WeightC.SIGMOID:
        return get_sigmoid_params_from_probs(probs)
    else:
        raise Exception("{} is not an understood weight parametrization".format(parametrization))


def create_stable_probs(probs):
    probs += np.ones_like(probs) * .001
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    return probs


def get_sigmoid_params_from_probs(probs):
    if probs.shape[-1] == 2:
        return tf.constant_initializer(-np.log(np.divide(1. - probs[:, :, 1], probs[:, :, 1])))
    else:
        sa = -np.log(np.divide(1. - probs[:, :, 1], probs[:, :, 1]))
        cond_prob = np.divide(probs[:, :, 2], 1 - probs[:, :, 1])
        sb = -np.log(np.divide(1. - cond_prob, cond_prob))

        sa = tf.constant_initializer(sa)
        sb = tf.constant_initializer(sb)
        return sa, sb


def get_logit_params_from_probs(probs):
    if probs.shape[-1] == 2:
        log_neg = np.zeros((probs.shape[0], probs.shape[1]))
        log_pos = np.log(np.divide(probs[:, :, 1], 1 - probs[:, :, 1]))

        log_neg = tf.constant_initializer(log_neg)
        log_pos = tf.constant_initializer(log_pos)
        return log_neg, log_pos
    else:
        log_neg = np.log(np.divide(1 - probs[:, :, 2] - probs[:, :, 1], probs[:, :, 1]))
        log_zer = np.zeros((probs.shape[0], probs.shape[1]))
        log_pos = np.log(np.divide(probs[:, :, 2], probs[:, :, 1]))

        log_neg = tf.constant_initializer(log_neg)
        log_zer = tf.constant_initializer(log_zer)
        log_pos = tf.constant_initializer(log_pos)
        return log_neg, log_zer, log_pos


def get_bin_prob_from_pretrained(weight, w_config):
    return tf.clip_by_value(.5 * (1. + weight),
                            w_config.from_pretrained_init_p_min,
                            w_config.from_pretrained_init_p_max)


def get_ter_prob_from_pretrained(weight, w_config):
    prob_0 = tf.clip_by_value(1 - tf.abs(weight),
                              w_config.from_pretrained_init_p_min,
                              w_config.from_pretrained_init_p_max)
    prob_p = tf.clip_by_value(.5 * (1 - prob_0 + weight),
                              w_config.from_pretrained_init_p_min,
                              w_config.from_pretrained_init_p_max)
    prob_n = tf.clip_by_value(1 - prob_0 - prob_p,
                              w_config.from_pretrained_init_p_min,
                              w_config.from_pretrained_init_p_max)
    quotient = prob_0 + prob_p + prob_n
    prob_0 = tf.divide(prob_0, quotient)
    prob_p = tf.divide(prob_p, quotient)
    return prob_0, prob_p



def get_var_names(var_key, *var_descriptions):
    var_names = []
    for var_desc in var_descriptions:
        var_names.append(var_key + var_desc)
    return var_names


def get_var_name(var_key, var_description):
    return var_key + var_description


# TODO: exponentiation of log var
class Weights:
    def __init__(self, var_keys, layer_config, w_shape, b_shape, tau):
        self.var_keys = var_keys
        self.train_config = get_train_config()
        self.gauss = tf.distributions.Normal(loc=0., scale=1.)
        self.uniform = tf.distributions.Uniform(low=0.01, high=.99)
        self.w_shape = w_shape
        self.b_shape = b_shape

        self.var_dict = dict()
        self.logder_derivs = dict()
        self.arm_samples = dict()
        self.arm_weights = [dict(), dict()]
        self.c_arm_sample_op = tf.no_op()
        self.tensor_dict = dict()
        self.w_config = dict()
        self.layer_config = layer_config
        self.weight_summaries = None
        self.sample_op = None
        self.map_sample_op = None
        self.init_op = None
        self.tau = tau
        self.epsilon = .00001

        for var_key in var_keys:
            self.w_config[var_key] = layer_config.get_weight_config(var_key, do_copy=True)

        self.create_tf_vars()
        self.create_init_op()

        if self.train_config.algorithm in [AlgorithmC.AR, AlgorithmC.ARM]:
            self.c_arm_sample_op = self.c_arm_create_sample_op()

    # Adds initialized tensorflow variables for storing parametrization of weight distributions.
    # It further creates a sampling operation based on map estimate of the distribution
    # It also creates histogram summaries for tensorboard
    # Currently supported distributions: Gaussian, Categorical with two or three categories (binary and ternary)
    def create_tf_vars(self):
        sample_ops = list()
        map_sample_ops = list()
        weight_summaries = list()
        for var_key in self.var_keys:
            # var_key without any suffixes stores deterministic values for w and b (samples from the respective dist.)
            if var_key.startswith('w'):
                shape = self.w_shape
                self.var_dict[var_key] = tf.get_variable(name=var_key, shape=shape,
                                                         initializer=get_initializer((WeightC.XAVIER_INIT, None), shape))
            elif var_key.startswith('b'):
                shape = self.b_shape
                self.var_dict[var_key] = tf.get_variable(name=var_key, shape=shape,
                                                         initializer=tf.zeros_initializer())
            else:
                raise Exception('var_key {} does not start with w or b'.format(var_key))

            # Adds two variables for parametrizing a continuous Gaussian distribution: mean and variance
            if self.w_config[var_key].dist == WeightC.GAUSSIAN:
                mean_var_name, variance_var_name = get_var_names(var_key, VarNames.GAUSSIAN_MEAN, VarNames.GAUSSIAN_VAR)
                self.var_dict[mean_var_name] = tf.get_variable(name=mean_var_name,
                                                               shape=shape,
                                                               initializer=get_initializer(
                                                                   self.w_config[var_key].mean_initializer, shape),
                                                               dtype=tf.float32)
                self.var_dict[variance_var_name] = tf.exp(tf.get_variable(name=variance_var_name,
                                                                   shape=shape,
                                                                   initializer=get_initializer(
                                                                              self.w_config[var_key].logvar_initializer,
                                                                              shape),
                                                                   dtype=tf.float32))

                weight_summaries.append(tf.summary.histogram(mean_var_name, self.var_dict[mean_var_name]))
                weight_summaries.append(tf.summary.histogram(variance_var_name, self.var_dict[variance_var_name]))

            # Adds variables for parametrizing a binary or ternary categorical distribution
            # using sigmoid parametrization
            elif self.w_config[var_key].parametrization == WeightC.SIGMOID:
                sa_var_name, sb_var_name = get_var_names(var_key, VarNames.SIGMOID_A, VarNames.SIGMOID_B)

                if self.w_config[var_key].dist == WeightC.BINARY:
                    #  p(w=1) = sigmoid(sb) -> from paper 1710.07739
                    if self.w_config[var_key].sb_initializer[0] == WeightC.DIRICHLET_INIT:
                        sb_init = get_dirichlet_init(self.w_config[var_key].sb_initializer, shape, WeightC.SIGMOID)
                    else:
                        sb_init = get_initializer(self.w_config[var_key].sb_initializer, shape)
                    self.var_dict[sb_var_name] = tf.get_variable(name=sb_var_name,
                                                                 shape=shape,
                                                                 initializer=sb_init,
                                                                 dtype=tf.float32)

                    weight_summaries.append(tf.summary.histogram(sb_var_name, self.var_dict[sb_var_name]))
                elif self.w_config[var_key].dist == WeightC.TERNARY:
                    # p(w=0) = sigmoid(sa), p(w=1 | w !=0) = sigmoid(sb) -> from paper 1710.07739
                    if self.w_config[var_key].sa_initializer[0] == WeightC.DIRICHLET_INIT:
                        sa_init, sb_init = get_dirichlet_init(self.w_config[var_key].sa_initializer, shape, WeightC.SIGMOID)
                    else:
                        sa_init = get_initializer(self.w_config[var_key].sa_initializer, shape)
                        sb_init = get_initializer(self.w_config[var_key].sb_initializer, shape)

                    self.var_dict[sa_var_name] = tf.get_variable(name=sa_var_name,
                                                                 shape=shape,
                                                                 initializer=sa_init,
                                                                 dtype=tf.float32)
                    self.var_dict[sb_var_name] = tf.get_variable(name=sb_var_name,
                                                                 shape=shape,
                                                                 initializer=sb_init,
                                                                 dtype=tf.float32)

                    weight_summaries.append(tf.summary.histogram(sa_var_name, self.var_dict[sa_var_name]))
                    weight_summaries.append(tf.summary.histogram(sb_var_name, self.var_dict[sb_var_name]))
                else:
                    raise Exception("Sigmoid parametrization does not support the given weight type: " +
                                    self.w_config[var_key].type)

            elif self.w_config[var_key].parametrization == WeightC.LOGIT:
                log_neg_var_name, log_zer_var_name, log_pos_var_name = get_var_names(var_key,
                                                                                     VarNames.LOGITS_NEG,
                                                                                     VarNames.LOGITS_ZER,
                                                                                     VarNames.LOGITS_POS)

                if self.w_config[var_key].dist == WeightC.BINARY:
                    if self.w_config[var_key].log_pos_initializer[0] == WeightC.DIRICHLET_INIT:
                        log_neg_init, log_pos_init = get_dirichlet_init(self.w_config[var_key].log_pos_initializer, shape, WeightC.LOGIT)
                    else:
                        log_neg_init = get_initializer(self.w_config[var_key].log_neg_initializer, shape)
                        log_pos_init = get_initializer(self.w_config[var_key].log_pos_initializer, shape)

                    # p(w) = softmax(logits) -> Stored are the unscaled logits for possible weight values
                    self.var_dict[log_neg_var_name] = tf.get_variable(name=log_neg_var_name,
                                                                      shape=shape,
                                                                      initializer=log_neg_init,
                                                                      dtype=tf.float32)
                    self.var_dict[log_pos_var_name] = tf.get_variable(name=log_pos_var_name,
                                                                      shape=shape,
                                                                      initializer=log_pos_init,
                                                                      dtype=tf.float32)

                    weight_summaries.append(tf.summary.histogram(log_neg_var_name, self.var_dict[log_neg_var_name]))
                    weight_summaries.append(tf.summary.histogram(log_pos_var_name, self.var_dict[log_pos_var_name]))

                elif self.w_config[var_key].dist == WeightC.TERNARY:
                    if self.w_config[var_key].log_pos_initializer[0] == WeightC.DIRICHLET_INIT:
                        log_neg_init, log_zer_init, log_pos_init = get_dirichlet_init(self.w_config[var_key].log_pos_initializer, shape, WeightC.LOGIT)
                    else:
                        log_neg_init = get_initializer(self.w_config[var_key].log_neg_initializer, shape)
                        log_zer_init = get_initializer(self.w_config[var_key].log_zer_initializer, shape)
                        log_pos_init = get_initializer(self.w_config[var_key].log_pos_initializer, shape)

                    # p(w) = softmax(logits) -> Stored are the unscaled logits for possible weight values
                    self.var_dict[log_neg_var_name] = tf.get_variable(name=log_neg_var_name,
                                                                      shape=shape,
                                                                      initializer=log_neg_init,
                                                                      dtype=tf.float32)
                    self.var_dict[log_zer_var_name] = tf.get_variable(name=log_zer_var_name,
                                                                      shape=shape,
                                                                      initializer=log_zer_init,
                                                                      dtype=tf.float32)
                    self.var_dict[log_pos_var_name] = tf.get_variable(name=log_pos_var_name,
                                                                      shape=shape,
                                                                      initializer=log_pos_init,
                                                                      dtype=tf.float32)

                    weight_summaries.append(tf.summary.histogram(log_neg_var_name, self.var_dict[log_neg_var_name]))
                    weight_summaries.append(tf.summary.histogram(log_zer_var_name, self.var_dict[log_zer_var_name]))
                    weight_summaries.append(tf.summary.histogram(log_pos_var_name, self.var_dict[log_pos_var_name]))
                else:
                    raise Exception("Logits parametrization does not support the given weight type: " +
                                    self.w_config[var_key].type)
            else:
                raise Exception("Incompatible weight type (" + self.w_config[var_key].type + ") and " +
                                "parametrization (" + self.w_config[var_key].parametrization + ")")

            sample_ops.append(tf.assign(self.var_dict[var_key], self.get_weight_sample(var_key)))
            map_sample_ops.append(tf.assign(self.var_dict[var_key], self.get_map_estimate(var_key)))

        self.sample_op = tf.group(*sample_ops)
        self.map_sample_op = tf.group(*map_sample_ops)
        self.weight_summaries = tf.summary.merge(weight_summaries)

    def c_arm_create_sample_op(self):
        arm_sample_ops = []
        for var_key in self.var_keys:
            if self.w_config[var_key].dist == WeightC.BINARY:
                arm_var_name = get_var_name(var_key, VarNames.ARM)
                shape = self.var_dict[var_key].shape
                self.arm_samples[var_key] = tf.get_variable(name=arm_var_name, shape=shape, dtype=tf.float32,
                                                            trainable=False)
                arm_sample_ops.append(tf.assign(self.arm_samples[var_key], self.uniform.sample(shape)))
        return tf.group(arm_sample_ops)

    def get_map_estimate(self, var_key):
        if self.check_w_dist(var_key, dist=WeightC.GAUSSIAN):
            mean_var_name = get_var_name(var_key, VarNames.GAUSSIAN_MEAN)
            return self.var_dict[mean_var_name]

        elif self.check_w_dist(var_key, dist=WeightC.BINARY):
            if self.check_w_param(var_key, parametrization=WeightC.SIGMOID):
                sb_var_name = get_var_name(var_key, VarNames.SIGMOID_B)
                return -1. + 2. * tf.cast(tf.argmax([tf.zeros_like(self.var_dict[sb_var_name]),
                                                     self.var_dict[sb_var_name]]), tf.float32)

            elif self.check_w_param(var_key, parametrization=WeightC.LOGIT):
                log_neg_var_name, log_pos_var_name = get_var_names(var_key, VarNames.LOGITS_NEG, VarNames.LOGITS_POS)
                return -1. + 2 * tf.cast(tf.argmax([self.var_dict[log_neg_var_name],
                                         self.var_dict[log_pos_var_name]]), tf.float32)

        elif self.check_w_dist(var_key, dist=WeightC.TERNARY):
            if self.check_w_param(var_key, parametrization=WeightC.SIGMOID):
                probs = self.get_discrete_probs(var_key)
                return -1. + tf.cast(tf.argmax(probs), tf.float32)

            elif self.check_w_param(var_key, parametrization=WeightC.LOGIT):
                log_neg_var_name, log_zer_var_name, log_pos_var_name = get_var_names(var_key,
                                                                                     VarNames.LOGITS_NEG,
                                                                                     VarNames.LOGITS_ZER,
                                                                                     VarNames.LOGITS_POS)
                return -1. + tf.cast(tf.argmax([self.var_dict[log_neg_var_name],
                                                self.var_dict[log_zer_var_name],
                                                self.var_dict[log_pos_var_name]]), tf.float32)

    # In contrast to generate weight sample, this sampling method is not dependent on
    def get_weight_sample(self, var_key):
        shape = self.var_dict[var_key].shape
        if self.check_w_dist(var_key, dist=WeightC.GAUSSIAN):
            mean_var_name, variance_var_name = get_var_names(var_key, VarNames.GAUSSIAN_MEAN, VarNames.GAUSSIAN_VAR)
            return self.var_dict[mean_var_name] + self.gauss.sample(shape) * tf.sqrt(self.var_dict[variance_var_name])
        else:
            probs = self.get_discrete_probs(var_key)
            reparam_args = self.gumbel_reparam_args(probs, shape)

            if self.check_w_dist(var_key, dist=WeightC.BINARY):
                exact_weights = -1. + 2. * tf.cast(tf.argmax(reparam_args), dtype=tf.float32)
            elif self.check_w_dist(var_key, dist=WeightC.TERNARY):
                exact_weights = -1. + tf.cast(tf.argmax(reparam_args), dtype=tf.float32)
            else:
                raise Exception()
            return exact_weights

    def check_w_dist(self, var_key, dist):
        return self.w_config[var_key].dist == dist

    def check_w_param(self, var_key, parametrization):
        return self.w_config[var_key].parametrization == parametrization

    def update_arm(self, loss, lr):
        update_ops = []
        for var_key in self.var_keys:
            if self.check_w_dist(var_key, dist=WeightC.BINARY):
                sb_var_name = get_var_name(var_key, VarNames.SIGMOID_B)
                update_ops.append(tf.assign(self.var_dict[sb_var_name],
                                            self.var_dict[sb_var_name] -
                                            lr * tf.multiply(loss, self.arm_samples[var_key] - .5)))
        return tf.group(*update_ops)

    def generate_weight_sample(self, var_key, exact=False, second_arm_pass=False, data_key=None):
        shape = self.var_dict[var_key].shape

        if self.check_w_dist(var_key, dist=WeightC.GAUSSIAN):
            mean_var_name, variance_var_name = get_var_names(var_key, VarNames.GAUSSIAN_MEAN, VarNames.GAUSSIAN_VAR)

            if self.train_config.algorithm in [AlgorithmC.AR, AlgorithmC.ARM, AlgorithmC.LOG_DERIVATIVE]:
                return self.var_dict[mean_var_name]
            return self.var_dict[mean_var_name] + self.gauss.sample(shape) * tf.sqrt(self.var_dict[variance_var_name])

        elif self.check_w_dist(var_key, dist=WeightC.BINARY):
            probs = self.get_discrete_probs(var_key)

            if self.train_config.algorithm in [AlgorithmC.AR, AlgorithmC.ARM]:
                if data_key == 'tr':
                    if second_arm_pass is False:
                        return 2 * tf.cast(tf.math.greater(probs[1], self.arm_samples[var_key]), dtype=tf.float32) - 1
                    else:
                        return 2 * tf.cast(tf.math.greater(self.arm_samples[var_key], probs[0]), dtype=tf.float32) - 1
                else:
                    exact = True

            reparam_args = self.gumbel_reparam_args(probs, shape)
            exact_weights = -1. + 2. * tf.cast(tf.argmax(reparam_args), dtype=tf.float32)
            gumbel_weights = self.expectation_tau_softmax(reparam_args)

            if exact:
                return exact_weights

            if self.train_config.algorithm == AlgorithmC.LOG_DERIVATIVE:
                sb_var_name = get_var_name(var_key, VarNames.SIGMOID_B)
                sig = tf.sigmoid(self.var_dict[sb_var_name])
                #self.logder_derivs[var_key] = tf.multiply(exact_weights, tf.multiply(sig, 1 - sig))

                # This derivative of log probability function with respect to the sigmoid_b parameter
                # and evaluated at exact weights.
                # It implements d/dq log(p(x_i | q))
                self.logder_derivs[var_key] = 0.5 * exact_weights + (1 - sig) - .5
                return exact_weights

            if self.train_config.ste_type == AlgorithmC.CUSTOM_LOGIT_STE:
                if not self.check_w_param(var_key, WeightC.LOGIT):
                    raise Exception('Custom logit STE only possible with logits parametrization')

                log_neg_var_name, log_pos_var_name = get_var_names(var_key, VarNames.LOGITS_NEG, VarNames.LOGITS_POS)
                derivative_weights = -tf.multiply(self.var_dict[log_neg_var_name],
                                                  tf.multiply(exact_weights - 1)) * (-.5) + \
                                     tf.multiply(self.var_dict[log_pos_var_name],
                                                 tf.multiply(exact_weights + 1)) * .5

                return tf.stop_gradient(exact_weights - derivative_weights) + derivative_weights

            if self.train_config.algorithm == AlgorithmC.REPARAMETRIZATION:
                if self.train_config.ste_type == AlgorithmC.NO_STE:
                    return gumbel_weights
                elif self.train_config.ste_type == AlgorithmC.GUMBEL_STE:
                    return tf.stop_gradient(exact_weights - gumbel_weights) + gumbel_weights

            return gumbel_weights

        elif self.check_w_dist(var_key, dist=WeightC.TERNARY):
            probs = self.get_discrete_probs(var_key)

            reparam_args = self.gumbel_reparam_args(probs, shape)
            exact_weights = -1. + tf.cast(tf.argmax(reparam_args), dtype=tf.float32)
            gumbel_weights = self.expectation_tau_softmax(reparam_args)

            if exact:
                return exact_weights

            if self.train_config.ste_type == AlgorithmC.CUSTOM_LOGIT_STE:
                if not self.check_w_param(var_key, parametrization=WeightC.LOGIT):
                    raise Exception('Custom STE only possible with logits parametrization')

                log_neg_var_name, log_zer_var_name, log_pos_var_name = get_var_names(var_key,
                                                                                     VarNames.LOGITS_NEG,
                                                                                     VarNames.LOGITS_ZER,
                                                                                     VarNames.LOGITS_POS)
                derivative_weights = -tf.multiply(self.var_dict[log_neg_var_name],
                                                  tf.multiply(exact_weights - 1,  exact_weights)) * 0.5 + \
                                     tf.multiply(self.var_dict[log_zer_var_name],
                                                 tf.multiply(exact_weights - 1,  exact_weights + 1)) * (-1) + \
                                     tf.multiply(self.var_dict[log_pos_var_name],
                                                 tf.multiply(exact_weights + 1,  exact_weights)) * 0.5
                return tf.stop_gradient(exact_weights - derivative_weights) + derivative_weights

            if self.train_config.ste_type == AlgorithmC.GUMBEL_STE:
                return tf.stop_gradient(exact_weights - gumbel_weights) + gumbel_weights

            return gumbel_weights
        else:
            raise Exception('weight type {} not understood'.format(self.w_config[var_key].type))

    def create_tensor_samples(self, suffix='', second_arm_pass=False, data_key=None):
        for var_key in self.var_keys:
            self.tensor_dict[var_key+suffix] = self.generate_weight_sample(var_key, exact=False,
                                                                           second_arm_pass=second_arm_pass,
                                                                           data_key=data_key)

    def normalize_weights(self, var_key):
        mean, var = tf.nn.moments(self.var_dict[var_key], axes=[0, 1])
        return tf.divide(self.var_dict[var_key], tf.sqrt(var))

    # Creates an operation that allows to initialize distribution parameters (Gaussian, binary and ternary categorical)
    # from continuous values. This is used when a NN with continuous deterministic weights is used for pretraining.
    # In case of a Gaussian distribution, just the mean is initialized.
    # In case of a discrete distribution, a discretization method is applied to infer weight probabilities
    # The continuous values are stored in self.var_dict[var_key], the respective distribution parameters will be stored
    # in self.var_dict[var_key + parameter_suffix]
    def create_init_op(self):
        init_ops = []
        for var_key in self.var_keys:
            if self.check_w_dist(var_key, dist=WeightC.GAUSSIAN):
                # Initialize Gaussian mean parameter
                mean_var_name = get_var_name(var_key, VarNames.GAUSSIAN_MEAN)
                init_ops.append(tf.assign(self.var_dict[mean_var_name], self.var_dict[var_key]))

            elif self.check_w_dist(var_key, dist=WeightC.BINARY):
                if not var_key.startswith('w'):
                    raise Exception()

                init_weights = self.normalize_weights(var_key)
                prob_1 = get_bin_prob_from_pretrained(init_weights, self.w_config[var_key])

                if self.check_w_param(var_key, parametrization=WeightC.SIGMOID):
                    sb_var_name = get_var_name(var_key, VarNames.SIGMOID_B)
                    init_ops.append(tf.assign(self.var_dict[sb_var_name], -tf.log(tf.divide(1. - prob_1, prob_1))))

                elif self.check_w_param(var_key, parametrization=WeightC.LOGIT):
                    log_neg_var_name, log_pos_var_name = get_var_names(var_key, VarNames.LOGITS_NEG, VarNames.LOGITS_POS)

                    init_ops.append(tf.assign(self.var_dict[log_neg_var_name], tf.zeros_like(self.var_dict[var_key])))
                    init_ops.append(tf.assign(self.var_dict[log_pos_var_name], tf.log(tf.divide(prob_1, (1 - prob_1)))))

            elif self.check_w_dist(var_key, dist=WeightC.TERNARY):
                if not var_key.startswith('w'):
                    raise Exception()

                init_weights = self.normalize_weights(var_key)
                prob_0, prob_1 = get_ter_prob_from_pretrained(init_weights, self.w_config[var_key])

                if self.check_w_param(var_key, parametrization=WeightC.SIGMOID):
                    sa_var_name, sb_var_name = get_var_names(var_key, VarNames.SIGMOID_A, VarNames.SIGMOID_B)

                    cond_prob = tf.divide(prob_1, 1 - prob_0)
                    init_ops.append(tf.assign(self.var_dict[sa_var_name], -tf.log(tf.divide(1. - prob_0, prob_0))))
                    init_ops.append(tf.assign(self.var_dict[sb_var_name],
                                              -tf.log(tf.divide(1. - cond_prob, cond_prob))))

                elif self.check_w_param(var_key, parametrization=WeightC.LOGIT):
                    log_neg_var_name, log_zer_var_name, log_pos_var_name = get_var_names(var_key,
                                                                                         VarNames.LOGITS_NEG,
                                                                                         VarNames.LOGITS_ZER,
                                                                                         VarNames.LOGITS_POS)

                    init_ops.append(tf.assign(self.var_dict[log_neg_var_name],
                                              tf.log(tf.divide((1 - prob_1 - prob_0), prob_0))))
                    init_ops.append(tf.assign(self.var_dict[log_zer_var_name], tf.zeros_like(self.var_dict[var_key])))
                    init_ops.append(tf.assign(self.var_dict[log_pos_var_name], tf.log(tf.divide(prob_1, prob_0))))
            else:
                raise Exception('weight type {} not understood'.format(self.w_config[var_key].type))
        self.init_op = tf.group(*init_ops)

    # Adds a Dirichlet regularizer for discrete variables.
    def get_dir_reg(self):
        dir_reg = 0.
        count = 0.
        for var_key in self.var_keys:
            if self.check_w_dist(var_key, dist=WeightC.BINARY) or self.check_w_dist(var_key, dist=WeightC.TERNARY):
                probs = self.get_discrete_probs(var_key, stacked=True)
                dir_reg += tf.reduce_mean(tf.reduce_prod(probs, axis=0))
                count += 100.

        if count == 0.:
            return 0.
        return dir_reg / count

    # That was changed to actual entropy
    # Adds a L2 regularizer on the parameters sa and sb (probability decay) to penalize low entropy
    def get_entropy_reg(self):
        ent_reg = 0.
        count = 0.
        for var_key in self.var_keys:
            sa_var_name, sb_var_name = get_var_names(var_key, VarNames.SIGMOID_A, VarNames.SIGMOID_B)
            if self.check_w_dist(var_key, dist=WeightC.BINARY) or self.check_w_dist(var_key, dist=WeightC.TERNARY):
                probs = self.get_discrete_probs(var_key, stacked=True)
                ent_reg += -tf.reduce_sum(tf.log(probs + .000000001) * probs)
                count += tf.cast(tf.size(self.var_dict[sb_var_name]), dtype=tf.float32)
        if count == 0.:
            return 0.

        return ent_reg / count

    # Adds a L2 regularizer for pretraining a deterministic network_api (non-bayesian)
    def get_pretraining_reg(self):
        l1_regu = tf.contrib.layers.l1_regularizer(scale=4.0)
        reg_term = 0.
        for var_key in self.var_keys:
            if self.check_w_dist(var_key, dist=WeightC.BINARY):
                reg_term += tf.nn.l2_loss(self.var_dict[var_key] - 1) + \
                           tf.nn.l2_loss(self.var_dict[var_key] + 1) - l1_regu(self.var_dict[var_key])
        return reg_term

    # Adds a regularization term on the posterior variance
    def get_var_reg(self):
        var_reg = 0
        count = 0.
        for var_key in self.var_keys:
            m, v = self.get_stats(var_key)
            var_reg += tf.reduce_mean(v)
            count += 1
        return var_reg / count

    # Returns probabilities p(w=-1), p(w=1) in binary case, and p(w=-1), p(w=0), p(w=1) in ternary case
    def get_discrete_probs(self, var_key, stacked=False):
        if self.check_w_param(var_key, WeightC.SIGMOID):
            if self.check_w_dist(var_key, WeightC.BINARY):
                sb_var_name = get_var_name(var_key, VarNames.SIGMOID_B)
                prob_1 = tf.nn.sigmoid(self.var_dict[sb_var_name])
                prob_n1 = 1. - prob_1

                if stacked:
                    return tf.stack([prob_n1, prob_1], axis=0)
                else:
                    return [prob_n1, prob_1]

            elif self.check_w_dist(var_key, WeightC.TERNARY):
                sa_var_name, sb_var_name = get_var_names(var_key, VarNames.SIGMOID_A, VarNames.SIGMOID_B)
                prob_0 = tf.nn.sigmoid(self.var_dict[sa_var_name])
                prob_1 = tf.nn.sigmoid(self.var_dict[sb_var_name]) * (1 - prob_0)
                prob_n1 = 1. - prob_0 - prob_1

                if stacked:
                    return tf.stack([prob_n1, prob_0, prob_1], axis=0)
                else:
                    return [prob_n1, prob_0, prob_1]

        elif self.check_w_param(var_key, WeightC.LOGIT):
            if self.check_w_dist(var_key, WeightC.BINARY):
                log_neg_var_name, log_pos_var_name = get_var_names(var_key, VarNames.LOGITS_NEG, VarNames.LOGITS_POS)
                stacked_logits = tf.stack([self.var_dict[log_neg_var_name], self.var_dict[log_pos_var_name]], axis=0)

            elif self.check_w_dist(var_key, WeightC.TERNARY):
                log_neg_var_name, log_zer_var_name, log_pos_var_name = get_var_names(var_key,
                                                                                     VarNames.LOGITS_NEG,
                                                                                     VarNames.LOGITS_ZER,
                                                                                     VarNames.LOGITS_POS)
                stacked_logits = tf.stack([self.var_dict[log_neg_var_name], self.var_dict[log_zer_var_name],
                                           self.var_dict[log_pos_var_name]], axis=0)

            stacked_softmax = tf.nn.softmax(stacked_logits, axis=0)

            if stacked:
                return stacked_softmax
            else:
                return tf.unstack(stacked_softmax, axis=0)

    # Returns mean and variance of a specified weight / bias
    def get_stats(self, var_key):
        if self.check_w_dist(var_key, dist=WeightC.GAUSSIAN):
            mean_var_name, variance_var_name = get_var_names(var_key, VarNames.GAUSSIAN_MEAN, VarNames.GAUSSIAN_VAR)

            m = self.var_dict[mean_var_name]
            v = self.var_dict[variance_var_name]

        elif self.check_w_dist(var_key, dist=WeightC.BINARY):
            if self.check_w_param(var_key, parametrization=WeightC.SIGMOID):
                sb_var_name = get_var_name(var_key, VarNames.SIGMOID_B)
                m = tf.nn.tanh(self.var_dict[sb_var_name] / 2)

            elif self.check_w_param(var_key, parametrization=WeightC.LOGIT):
                log_neg_var_name, log_pos_var_name = get_var_names(var_key, VarNames.LOGITS_NEG, VarNames.LOGITS_POS)
                m = tf.nn.tanh((self.var_dict[log_pos_var_name] - self.var_dict[log_neg_var_name]) / 2)

            v = 1 - tf.square(m) + .0001

        elif self.check_w_dist(var_key, dist=WeightC.TERNARY):
            if self.check_w_param(var_key, parametrization=WeightC.SIGMOID):
                sa_var_name, sb_var_name = get_var_names(var_key, VarNames.SIGMOID_A, VarNames.SIGMOID_B)
                prob_not_zero = 1. - tf.nn.sigmoid(self.var_dict[sa_var_name])
                m = tf.nn.tanh(self.var_dict[sb_var_name] / 2.) * prob_not_zero

            elif self.check_w_param(var_key, parametrization=WeightC.LOGIT):
                log_neg_var_name, log_zer_var_name, log_pos_var_name = get_var_names(var_key,
                                                                                     VarNames.LOGITS_NEG,
                                                                                     VarNames.LOGITS_ZER,
                                                                                     VarNames.LOGITS_POS)
                probs = tf.nn.softmax([self.var_dict[log_neg_var_name], self.var_dict[log_zer_var_name],
                                       self.var_dict[log_pos_var_name]])
                prob_not_zero = probs[0] + probs[2]
                m = probs[2] - probs[0]

            v = prob_not_zero - tf.square(m) + .0001
        else:
            raise Exception()
        return m, v

    # Calculates the KL loss over all weights
    def get_kl_loss(self):
        kl_loss = 0
        for var_key in self.var_keys:
            if self.check_w_dist(var_key, dist=WeightC.GAUSSIAN):
                mean_var_name, variance_var_name = get_var_names(var_key, VarNames.GAUSSIAN_MEAN, VarNames.GAUSSIAN_VAR)

                prior_v = np.exp(self.w_config[var_key].logvar_prior)
                prior_m = self.w_config[var_key].mean_prior
                q_v = self.var_dict[variance_var_name]
                q_m = self.var_dict[mean_var_name]
                kl_loss += tf.reduce_sum(0.5 * tf.log(tf.divide(prior_v, q_v)) +
                                         tf.divide(q_v + tf.square(q_m - prior_m),
                                                   2 * prior_v)
                                         - 0.5)
            else:
                priors = self.w_config[var_key].priors
                probs = self.get_discrete_probs(var_key, stacked=False)
                epsilon = 1e-20

                if self.check_w_dist(var_key, dist=WeightC.BINARY):
                    kl_loss += tf.reduce_sum(probs[0] * tf.log(epsilon + tf.divide(probs[0], priors[0])) +
                                             probs[1] * tf.log(epsilon + tf.divide(probs[1], priors[1])))

                elif self.check_w_dist(var_key, dist=WeightC.TERNARY):
                    kl_loss += tf.reduce_sum(probs[0] * tf.log(epsilon + tf.divide(probs[0], priors[0])) +
                                             probs[1] * tf.log(epsilon + tf.divide(probs[1], priors[1])) +
                                             probs[2] * tf.log(epsilon + tf.divide(probs[2], priors[2])))
                else:
                    raise Exception('weight type {} not understood'.format(self.w_config[var_key].type))
        return kl_loss

    def adapt_weights(self, x_m, w_var_key, b_var_key, a):
        w_m_new = []
        w_v_new = []
        b_m_new = []
        b_v_new = []
        w_var_key = w_var_key + '_adapt'
        b_var_key = b_var_key + '_adapt'
        x_m = tf.unstack(tf.expand_dims(x_m, axis=2), axis=0)
        acts = tf.unstack(tf.expand_dims(a, axis=1), axis=0)
        for x, a, w_m, w_v, b_m, b_v in zip(x_m, acts, self.tensor_dict[w_var_key + '_m'], self.tensor_dict[w_var_key + '_v'],
                                         self.tensor_dict[b_var_key + '_v'], self.tensor_dict[b_var_key + '_v']):
            x_square = tf.square(x)
            r_means = tf.concat([b_m, tf.multiply(x, w_m)], axis=0)
            r_vars = tf.concat([b_v, tf.multiply(x_square, w_v)], axis=0)
            lambda_m = tf.reduce_mean(r_means, axis=0, keepdims=True)
            lambda_v = tf.reduce_mean(r_vars, axis=0, keepdims=True)
            r_means_new = tf.split(r_means + tf.multiply(tf.divide(a-lambda_m, lambda_v), r_vars), [1, -1], axis=0)
            r_vars_new = tf.split(r_vars - tf.divide(tf.square(r_vars), lambda_v), [1,-1], axis=0)
            w_m_new.append(tf.divide(r_means_new[1], x))
            w_v_new.append(tf.divide(r_vars_new[1], x_square))
            b_m_new.append(r_means_new[0])
            b_v_new.append(r_vars_new[0])

        self.tensor_dict[w_var_key + '_m'] = tf.stack(w_m_new, axis=0)
        self.tensor_dict[w_var_key + '_v'] = tf.stack(w_v_new, axis=0)
        self.tensor_dict[b_var_key + '_m'] = tf.stack(b_m_new, axis=0)
        self.tensor_dict[b_var_key + '_v'] = tf.stack(b_v_new, axis=0)

    def create_adapted_stats(self, var_key, m, v):
        self.tensor_dict[var_key + '_adapt_m'] = m
        self.tensor_dict[var_key + '_adapt_v'] = v

    def get_adapted_stats(self, var_key, m, v):
        return self.tensor_dict[var_key + '_adapt_m'], self.tensor_dict[var_key + '_adapt_v']

    # Used by local reparametrization for both sampling continuous and discrete activations
    # Supports continuous, binary and ternary weights
    # If act_func == None: Returns sample of activation
    #                Else: Returns sample of discretized tanh or sig
    def sample_activation(self, w_var_key, b_var_key, x_m, act_func, init, layer_norm=False):
        #if not self.layer_config.lr_adapt or init:
        w_m, w_v = self.get_stats(w_var_key)
        b_m, b_v = self.get_stats(b_var_key)
        #else:
            #w_m, w_v = self.get_adapted_stats(w_var_key)
            #b_m, b_v = self.get_adapted_stats(w_var_key)

        #if self.layer_config.lr_adapt is False:
        if self.layer_config.get_gate_config(b_var_key).bias_enabled:
            mean = tf.matmul(x_m, w_m) + b_m
            std = tf.sqrt(tf.matmul(tf.square(x_m), w_v) + b_v + self.epsilon)
        else:
            mean = tf.matmul(x_m, w_m)
            std = tf.sqrt(tf.matmul(tf.square(x_m,), w_v) + self.epsilon)
        if layer_norm:
            m, v = tf.nn.moments(mean, axes=1)
            std = tf.divide(std, tf.expand_dims(tf.sqrt(v), axis=1) + .005)
            mean = tf.contrib.layers.layer_norm(mean)
        #else:
            #layer_inputs = tf.unstack(tf.expand_dims(x_m, axis=1), axis=0)
            #means = []
            #stds = []
            #if init:
                #w_m = [w_m] * len(layer_inputs)
                #w_v = [w_v] * len(layer_inputs)
                #b_m = [b_m] * len(layer_inputs)
                #b_v = [b_v] * len(layer_inputs)
                #self.create_adapted_weights(w_var_key, w_m, w_v)
                #self.create_adapted_weights(b_var_key, b_m, b_v)
            #for sample_w_m, sample_w_v, sample_b_m, sample_b_v, layer_input in zip(w_m, w_v, b_m, b_v, layer_inputs):
                #means.append(tf.matmul(layer_input, sample_w_m) + sample_b_m)
                #stds.append(tf.sqrt(tf.matmul(tf.square(layer_input), sample_w_v) + sample_b_v))
            #mean = tf.squeeze(tf.stack(means, axis=0))
            #std = tf.squeeze(tf.stack(stds, axis=0))

        shape = (tf.shape(x_m)[0], tf.shape(b_m)[1])

        if act_func is None:
            a = mean + tf.multiply(self.gauss.sample(sample_shape=shape), std)
            #if self.layer_config.lr_adapt:
                #self.adapt_weights(x_m, w_var_key, b_var_key, a)
            return a
        else:
            prob_1 = 0.5 + 0.5 * tf.erf(tf.divide(mean, std * np.sqrt(2) + .000001))
            prob_1 = 0.0001 + (0.9999 - 0.0001) * prob_1
            reparam_args = [tf.log(1-prob_1) + self.sample_gumbel(shape), tf.log(prob_1) + self.sample_gumbel(shape)]
            gumbel_output = tf.nn.tanh((reparam_args[1] - reparam_args[0]) / (self.tau * 2))

            if self.train_config.ste_type == AlgorithmC.GUMBEL_STE:
                exact_output = -1 + 2 * tf.cast(tf.argmax(reparam_args), dtype=tf.float32)
                output = tf.stop_gradient(exact_output - gumbel_output) + gumbel_output
            elif self.train_config.ste_type == AlgorithmC.NO_STE:
                output = gumbel_output
            else:
                raise Exception()

            if act_func == ActivationFunctionsC.TANH:
                return output
            elif act_func == ActivationFunctionsC.SIGMOID:
                return (output + 1.) / 2.
            else:
                raise Exception('activation function not understood')

    def get_weight_probs(self):
        prob_dict = {}
        for var_key in self.var_keys:
            prob_dict[var_key] = {}

            if self.check_w_dist(var_key, dist=WeightC.GAUSSIAN):
                mean_var_name, variance_var_name = get_var_names(var_key, VarNames.GAUSSIAN_MEAN, VarNames.GAUSSIAN_VAR)

                prob_dict[var_key]['m'] = self.var_dict[mean_var_name]
                prob_dict[var_key]['v'] = self.var_dict[variance_var_name]
            else:
                prob_dict[var_key]['probs'] = self.get_discrete_probs(var_key)

        return prob_dict

    def sample_gumbel(self, shape):
        return -tf.log(-tf.log(self.uniform.sample(shape)))

    # Computes expectation of the tau-softmax relaxation given the reparametrization arguments
    def expectation_tau_softmax(self, reparam_args):
        args = []
        for reparam_arg in reparam_args:
            args.append(reparam_arg / self.tau)

        softmax = tf.nn.softmax(tf.stack(args, axis=0), axis=0)
        return softmax[-1] - softmax[0]

    # Computes the reparametrization arguments for the Gumbel-reparametrization trick
    def gumbel_reparam_args(self, probs, shape):
        reparam_args = []
        if type(probs) is list:
            for prob in probs:
                reparam_args.append(tf.log(prob + .00001) + self.sample_gumbel(shape))
        else:
            for idx in range(probs.shape[0]):
                reparam_args.append(tf.log(probs[idx] + .00001) + self.sample_gumbel(shape))
        return reparam_args



