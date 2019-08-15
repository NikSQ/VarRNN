import tensorflow as tf
#import tensorflow_probability as tfp
import numpy as np
import copy


def get_mean_initializer(w_config, shape):
    if w_config['init_m'] == 'xavier':
        init_vals = np.random.randn(shape[0], shape[1]) * np.sqrt(2/sum(shape))
    elif w_config['init_m'] == 'zeros':
        init_vals = np.zeros(shape)
    elif w_config['init_m'] == 'ones':
        init_vals = np.ones(shape)
    else:
        raise Exception("{} is not a valid weight initialization".format(w_config['init_m']))
    return tf.constant_initializer(init_vals)


def get_var_initializer(w_config, shape):
    if w_config['init_v'] == 'xavier':
        w_config['prior_v'] = np.sqrt(2/sum(shape))
        init_vals = np.ones(shape) * w_config['prior_v']
    else:
        init_vals = np.ones(shape) * w_config['init_v']
        #init_vals = np.ones(shape) * 0.0001
    return tf.constant_initializer(init_vals)


def get_binary_initializer(shape):
    init_vals = 2 * np.random.binomial(n=1, p=0.5, size=shape) - 1
    return tf.constant_initializer(init_vals)


def get_xavier_initializer(shape):
    init_vals = np.random.randn(shape[0], shape[1]) * np.sqrt(2/sum(shape))
    return tf.constant_initializer(init_vals)


# Return p(w=0) for initializing ternary weights, given a continuous weight
def get_init_zero_prob(var, w_config):
    weight = tf.maximum(tf.minimum(var, 1.), -1.)
    prob_0 = w_config['p0max'] - \
             (w_config['p0max'] - w_config['p0min']) * tf.abs(weight)
    return weight, prob_0


# Return p(w=1) for initializing ternary weights, given a continuous weight
def get_init_one_prob(w_config, prob_0, weight):
    prob_1 = 0.5 * (1. + tf.divide(weight, 1. - prob_0))
    prob_1 = tf.minimum(tf.maximum(prob_1, w_config['pmin']), w_config['pmax'])
    return prob_1


# Returns the probabilities p(w=-1), p(w=0), p(w=1) for sigmoid parametrization
def get_ternary_probs(sa, sb):
    prob_0 = tf.nn.sigmoid(sa)
    prob_1 = tf.nn.sigmoid(sb)*(1 - prob_0)
    prob_2 = 1. - prob_0 - prob_1
    #prob_0 = (.99 - .05) * prob_0 + .05
    #prob_1 = (.95 - .05) * prob_1 + .05
    #prob_2 = (.95 - .05) * prob_2 + .05
    return [prob_2, prob_0, prob_1]


class Weights:
    def __init__(self, var_keys, layer_config, w_shape, b_shape, batchnorm):
        self.var_keys = var_keys
        self.gauss = tf.distributions.Normal(loc=0., scale=1.)
        self.uniform = tf.distributions.Uniform(low=0.01, high=.99)
        self.w_shape = w_shape
        self.b_shape = b_shape

        self.var_dict = dict()
        self.tensor_dict = dict()
        self.w_config = dict()
        self.layer_config = layer_config
        self.weight_summaries = None
        self.sample_op = None
        self.init_op = None

        for var_key in var_keys:
            self.w_config[var_key] = copy.deepcopy(layer_config[var_key])

        self.create_vars()
        self.create_init_op()

    # Returns unnormalized probabilities of the concrete distribution
    def get_exp_gumbel(self, probs, shape):
        return tf.exp((tf.log(probs) - tf.log(-tf.log(self.uniform.sample(shape))))/self.layer_config['tau'])

    def create_vars(self):
        sample_ops = list()
        weight_summaries = list()
        for var_key in self.var_keys:
            # var_key without any suffixes stores deterministic values for w and b (samples from the respective dist.)
            if var_key.startswith('w'):
                shape = self.w_shape
                self.var_dict[var_key] = tf.get_variable(name=var_key, shape=shape,
                                                         initializer=get_xavier_initializer(shape))
            elif var_key.startswith('b'):
                shape = self.b_shape
                self.var_dict[var_key] = tf.get_variable(name=var_key, shape=shape,
                                                         initializer=tf.zeros_initializer())
            else:
                raise Exception('var_key {} does not start with w or b'.format(var_key))

            # Continuous distributions are parametrized with mean and variance with respective suffixes _m and _v
            if self.w_config[var_key]['type'] == 'continuous':
                self.var_dict[var_key + '_m'] = tf.get_variable(name=var_key + '_m', shape=shape,
                                                                       initializer=get_mean_initializer(
                                                                           self.w_config[var_key], shape))
                self.var_dict[var_key + '_v'] = tf.exp(tf.get_variable(name=var_key + '_v', shape=shape,
                                                                initializer=get_var_initializer(
                                                                    self.w_config[var_key], shape)))

                weight_summaries.append(tf.summary.histogram(var_key + '_m', self.var_dict[var_key + '_m']))
                weight_summaries.append(tf.summary.histogram(var_key + '_v', self.var_dict[var_key + '_v']))
            # binary distribution is represented with a bernoulli parameter p(w=1) = sb
            elif self.w_config[var_key]['type'] == 'binary':
                if self.layer_config['parametrization'] == 'sigmoid':
                    #  p(w=1) = sigm(sb) -> from paper 1710.07739
                    self.var_dict[var_key + '_sb'] = tf.get_variable(name=var_key + '_sb', shape=shape,
                                                                     initializer=get_xavier_initializer(shape), dtype=tf.float32)
                    weight_summaries.append(tf.summary.histogram(var_key + '_sb', self.var_dict[var_key + '_sb']))
                elif self.layer_config['parametrization'] == 'logits':
                    # p(w) = softmax(logits) -> Stored are the unscaled logits for negative and positive weight value
                    self.var_dict[var_key + '_log_neg'] = tf.get_variable(name=var_key + '_log_neg', shape=shape,
                                                                          initializer=get_xavier_initializer(shape), dtype=tf.float32)
                    self.var_dict[var_key + '_log_pos'] = tf.get_variable(name=var_key + '_log_pos', shape=shape,
                                                                          initializer=get_xavier_initializer(shape), dtype=tf.float32)
                    weight_summaries.append(tf.summary.histogram(var_key + '_log_neg', self.var_dict[var_key + '_log_neg']))
                    weight_summaries.append(tf.summary.histogram(var_key + '_log_pos', self.var_dict[var_key + '_log_pos']))
                else:
                    raise Exception("Weight parametrization not understood")
            elif self.w_config[var_key]['type'] == 'ternary':
                if self.layer_config['parametrization'] == 'sigmoid':
                    # p(w=0) = sigm(sa), p(w=1 | w !=0) = sigm(sb) -> from paper 1710.07739
                    self.var_dict[var_key + '_sa'] = tf.get_variable(name=var_key + '_sa', shape=shape,
                                                                     initializer=get_xavier_initializer(shape), dtype=tf.float32)
                    self.var_dict[var_key + '_sb'] = tf.get_variable(name=var_key + '_sb', shape=shape,
                                                                     initializer=get_xavier_initializer(shape), dtype=tf.float32)
                    weight_summaries.append(tf.summary.histogram(var_key + '_sa', self.var_dict[var_key + '_sa']))
                    weight_summaries.append(tf.summary.histogram(var_key + '_sb', self.var_dict[var_key + '_sb']))
                elif self.layer_config['parametrization'] == 'logits':
                    # p(w) = softmax(logits) -> Stored are the unscaled logits for negative, zero and positive weight value
                    self.var_dict[var_key + '_log_neg'] = tf.get_variable(name=var_key + '_log_neg', shape=shape,
                                                                          initializer=get_xavier_initializer(shape), dtype=tf.float32)
                    self.var_dict[var_key + '_log_zer'] = tf.get_variable(name=var_key + '_log_zer', shape=shape,
                                                                          initializer=get_xavier_initializer(shape), dtype=tf.float32)
                    self.var_dict[var_key + '_log_pos'] = tf.get_variable(name=var_key + '_log_pos', shape=shape,
                                                                          initializer=get_xavier_initializer(shape), dtype=tf.float32)
                    weight_summaries.append(tf.summary.histogram(var_key + '_log_neg', self.var_dict[var_key + '_log_neg']))
                    weight_summaries.append(tf.summary.histogram(var_key + '_log_zer', self.var_dict[var_key + '_log_zer']))
                    weight_summaries.append(tf.summary.histogram(var_key + '_log_pos', self.var_dict[var_key + '_log_pos']))
                else:
                    raise Exception("Weight parametrization not understood")

            else:
                raise Exception('weight type {} not understood'.format(self.w_config[var_key]['type']))

            sample_ops.append(tf.assign(self.var_dict[var_key], self.get_map_estimate(var_key)))

        self.sample_op = tf.group(*sample_ops)
        self.weight_summaries = tf.summary.merge(weight_summaries)

    def get_map_estimate(self, var_key):
        if self.w_config[var_key]['type'] == 'continuous':
            return self.var_dict[var_key + '_m']
        elif self.w_config[var_key]['type'] == 'binary':
            if self.layer_config['parametrization'] == 'sigmoid':
                return -1. + 2. * tf.cast(tf.argmax[tf.zeros_like(self.var_dict[var_key + '_sb']),
                                                  self.var_dict[var_key + '_sb']], tf.float32)
            elif self.layer_config['parametrization'] == 'logits':
                return -1. + 2 * tf.cast(tf.argmax([self.var_dict[var_key + '_log_neg'],
                                          self.var_dict[var_key + '_log_pos']], tf.float32))
        elif self.w_config[var_key]['type'] == 'ternary':
            if self.layer_config['parametrization'] == 'sigmoid':
                probs = get_ternary_probs(self.var_dict[var_key + '_sa'], self.var_dict[var_key + '_sb'])
                return -1. + tf.cast(tf.argmax(probs), tf.float32)
            elif self.layer_config['parametrization'] == 'logits':
                return -1. + tf.cast(tf.argmax([self.var_dict[var_key + '_log_neg'],
                                          self.var_dict[var_key + '_log_zer'],
                                          self.var_dict[var_key + '_log_pos']]))

    def generate_weight_sample(self, var_key, exact=False):
        shape = self.var_dict[var_key].shape
        if self.w_config[var_key]['type'] == 'continuous':
            return self.var_dict[var_key + '_m'] + self.gauss.sample(shape) * tf.sqrt(self.var_dict[var_key + '_v'])
        elif self.w_config[var_key]['type'] == 'binary':
            if self.layer_config['parametrization'] == 'sigmoid':
                if exact:
                    return -1. + 2. * tf.cast(tf.argmax([-self.var_dict[var_key + '_sb']
                                                         - tf.log(-tf.log(self.uniform.sample(shape))),
                                                         -tf.log(-tf.log(self.uniform.sample(shape)))]), dtype=tf.float32)
                else:
                    return tf.nn.tanh((self.var_dict[var_key + '_sb']
                                       - tf.log(-tf.log(self.uniform.sample(shape)))
                                       + tf.log(-tf.log(self.uniform.sample(shape))))
                                      / (self.layer_config['tau'] * 2))
            elif self.layer_config['parametrization'] == 'logits':
                if exact:
                    return -1. + 2. * tf.cast(tf.argmax([self.var_dict[var_key + '_log_neg']
                                                         - tf.log(-tf.log(self.uniform.sample(shape))),
                                                         self.var_dict[var_key + '_log_pos']
                                                         - tf.log(-tf.log(self.uniform.sample(shape)))]),
                                              dtype=tf.float32)
                else:
                    return tf.nn.tanh((self.var_dict[var_key + '_log_pos'] - self.var_dict[var_key + ['_log_neg']]
                                       - tf.log(-tf.log(self.uniform.sample(shape)))
                                       + tf.log(-tf.log(self.uniform.sample(shape))))
                                      / (self.layer_config['tau'] * 2))
            else:
                raise Exception('parametrization not understood')

        elif self.w_config[var_key]['type'] == 'ternary':
            if self.layer_config['parametrization'] == 'sigmoid':
                probs = get_ternary_probs(self.var_dict[var_key + '_sa'], self.var_dict[var_key + '_sb'])
                if exact:
                    return -1. + tf.cast(tf.argmax([tf.divide(probs[0], -tf.log(self.uniform.sample(shape))),
                                                     tf.divide(probs[1], -tf.log(self.uniform.sample(shape))),
                                                     tf.divide(probs[2], -tf.log(self.uniform.sample(shape)))]),
                                                    dtype=tf.float32)
                else:
                    exp0 = self.get_exp_gumbel(probs[0], self.var_dict[var_key].shape)
                    exp1 = self.get_exp_gumbel(probs[1], self.var_dict[var_key].shape)
                    exp2 = self.get_exp_gumbel(probs[2], self.var_dict[var_key].shape)
                    return tf.divide(exp2 - exp0, exp0 + exp1 + exp2)
            elif self.layer_config['parametrization'] == 'logits':
                probs = tf.nn.softmax([self.var_dict[var_key + '_log_neg'], self.var_dict[var_key + '_log_zer'],
                                       self.var_dict[var_key + '_log_pos']])
                if exact:
                    return -1. + tf.cast(tf.argmax([self.var_dict[var_key + '_log_neg'] -
                                                    tf.log(-tf.log(self.uniform.sample(shape))),
                                                    self.var_dict[var_key + '_log_zer'] -
                                                    tf.log(-tf.log(self.uniform.sample(shape))),
                                                    self.var_dict[var_key + '_log_pos'] -
                                                    tf.log(-tf.log(self.uniform.sample(shape)))]),
                                         dtype=tf.float32)
                else:
                    exp0 = self.get_exp_gumbel(probs[0], self.var_dict[var_key].shape)
                    exp1 = self.get_exp_gumbel(probs[1], self.var_dict[var_key].shape)
                    exp2 = self.get_exp_gumbel(probs[2], self.var_dict[var_key].shape)
                    return tf.divide(exp2 - exp0, exp0 + exp1 + exp2)
            else:
                raise Exception('parametrization not understood')

        else:
            raise Exception('weight type {} not understood'.format(self.w_config[var_key]['type']))

    def create_tensor_samples(self, suffix=''):
        for var_key in self.var_keys:
            self.tensor_dict[var_key+suffix] = self.generate_weight_sample(var_key, False)

    def normalize_weights(self, var_key):
        mean, var = tf.nn.moments(self.var_dict[var_key], axes=[0,1])
        tf.assign(self.var_dict[var_key], tf.divide(self.var_dict[var_key], tf.sqrt(var)))

    def create_init_op(self):
        init_ops = []
        for var_key in self.var_keys:
            if self.w_config[var_key]['type'] == 'continuous':
                init_ops.append(tf.assign(self.var_dict[var_key + '_m'], self.var_dict[var_key]))
            elif self.w_config[var_key]['type'] == 'binary':
                if var_key.startswith('w'):
                    self.normalize_weights(var_key)
                prob_1 = get_init_one_prob(self.w_config[var_key], 0., self.var_dict[var_key])
                if self.layer_config['parametrization'] == 'sigmoid':
                    init_ops.append(tf.assign(self.var_dict[var_key + '_sb'], -tf.log(tf.divide(1. - prob_1, prob_1))))
                elif self.layer_config['parametrization'] == 'logits':
                    init_ops.append(tf.assign(self.var_dict[var_key + '_log_neg'], tf.zeros_like(self.var_dict[var_key])))
                    init_ops.append(tf.assign(self.var_dict[var_key + '_log_pos'], tf.log(tf.divide(prob_1, (1 - prob_1)))))
            elif self.w_config[var_key]['type'] == 'ternary':
                if var_key.startswith('w'):
                    self.normalize_weights(var_key)
                weight, prob_0 = get_init_zero_prob(self.var_dict[var_key], self.w_config[var_key])
                prob_1 = get_init_one_prob(self.w_config[var_key], prob_0, weight)
                if self.layer_config['parametrization'] == 'sigmoid':
                    init_ops.append(tf.assign(self.var_dict[var_key + '_sa'], -tf.log(tf.divide(1. - prob_0, prob_0))))
                    init_ops.append(tf.assign(self.var_dict[var_key + '_sb'], -tf.log(tf.divide(1. - prob_1, prob_1))))
                elif self.layer_config['parametrization'] == 'logits':
                    init_ops.append(tf.assign(self.var_dict[var_key + '_log_zer'], tf.zeros_like(self.var_dict[var_key])))
                    init_ops.append(tf.assign(self.var_dict[var_key + '_log_pos'], tf.log(tf.divide(prob_1, prob_0))))
                    init_ops.append(tf.assign(self.var_dict[var_key + '_log_neg'], tf.log(tf.divide((1 - prob_1), prob_0))))
            else:
                raise Exception('weight type {} not understood'.format(self.w_config[var_key]['type']))
        self.init_op = tf.group(*init_ops)

    # Adds a Dirichlet regularizer for discrete variables.
    # TODO: Support for logit parametrization
    def get_dir_reg(self):
        dir_reg = 0.
        count = 0.
        for var_key in self.var_keys:
            if self.w_config[var_key]['type'] == 'binary':
                exp = tf.exp(-self.var_dict[var_key + '_sb'])
                dir_reg += tf.reduce_mean(tf.divide(exp, tf.square(1 + exp)))
                count += 1.
            elif self.w_config[var_key]['type'] == 'ternary':
                probs = tf.stack(get_ternary_probs(self.var_dict[var_key + '_sa'], self.var_dict[var_key + '_sb']), axis=0)
                dir_reg += tf.reduce_mean(tf.reduce_prod(probs, axis=0))
                count += 1.
        if count == 0.:
            return 0.
        return dir_reg / count

    # Adds a L2 regularizer on the parameters sa and sb (probability decay) to penalize low entropy
    # TODO: Support for logit parametrization
    def get_entropy_reg(self):
        ent_reg = 0.
        count = 0.
        for var_key in self.var_keys:
            if self.w_config[var_key]['type'] == 'binary':
                ent_reg += tf.nn.l2_loss(self.var_dict[var_key + '_sb'])
                count += tf.cast(tf.size(self.var_dict[var_key + '_sb']), dtype=tf.float32)
            if self.w_config[var_key]['type'] == 'ternary':
                ent_reg += tf.nn.l2_loss(self.var_dict[var_key + '_sa']) \
                            + tf.nn.l2_loss(self.var_dict[var_key + '_sb'])
                count += tf.cast(tf.size(self.var_dict[var_key + '_sa']) + tf.size(self.var_dict[var_key + '_sb']),
                                 dtype=tf.float32)
        if count == 0.:
            return 0.

        return ent_reg / count

    # Adds a L2 regularizer for pretraining a deterministic network (non-bayesian)
    def get_pretraining_reg(self):
        l1_regu = tf.contrib.layers.l1_regularizer(scale=4.0)
        reg_term = 0.
        for var_key in self.var_keys:
            if self.w_config[var_key]['type'] == 'binary':
                reg_term += tf.nn.l2_loss(self.var_dict[var_key] - 1) + \
                           tf.nn.l2_loss(self.var_dict[var_key] + 1) - l1_regu(self.var_dict[var_key])
        return reg_term

    # Adds a regularization term on the posterior variance
    def get_var_reg(self):
        var_reg = 0
        count = 0.
        for var_key in self.var_keys:
            if self.w_config[var_key]['type'] == 'continuous' or self.w_config[var_key]['type'] == 'ternary':
                m, v = self.get_stats(var_key)
                var_reg += tf.reduce_mean(v)
                count += 1
        return var_reg / count

    # Returns mean and variance of a specified weight / bias
    def get_stats(self, var_key):
        if self.w_config[var_key]['type'] == 'continuous':
            m = self.var_dict[var_key + '_m']
            v = self.var_dict[var_key + '_v']
        elif self.w_config[var_key]['type'] == 'binary':
            if self.layer_config['parametrization'] == 'sigmoid':
                m = tf.nn.tanh(self.var_dict[var_key + '_sb'] / 2)
            elif self.layer_config['parametrization'] == 'logits':
                m = tf.nn.tanh((self.var_dict[var_key + '_log_pos'] - self.var_dict[var_key + '_log_neg'])/2)
            v = 1 - tf.square(m)
        elif self.w_config[var_key]['type'] == 'ternary':
            if self.layer_config['parametrization'] == 'sigmoid':
                prob_not_zero = 1. - tf.nn.sigmoid(self.var_dict[var_key + '_sa'])
                m = tf.nn.tanh(self.var_dict[var_key + '_sb'] / 2.) * prob_not_zero
            elif self.layer_config['parametrization'] == 'logits':
                probs = tf.nn.softmax([self.var_dict[var_key + '_log_neg'], self.var_dict[var_key + '_log_zer'],
                                       self.var_dict[var_key + '_log_pos']])
                prob_not_zero = probs[0] + probs[2]
                m = (probs[2] - probs[0]) * prob_not_zero
            v = prob_not_zero - tf.square(m)
        else:
            raise Exception()
        return m, v

    # Calculates the KL loss over all weights
    def get_kl_loss(self):
        kl_loss = 0
        for var_key in self.var_keys:
            if self.w_config[var_key]['type'] == 'continuous':
                prior_v = np.exp(self.w_config[var_key]['prior_v'])
                prior_m = self.w_config[var_key]['prior_m']
                q_v = self.var_dict[var_key + '_v']
                q_m = self.var_dict[var_key + '_m']
                kl_loss += tf.reduce_sum(0.5 * tf.log(tf.divide(prior_v, q_v)) + tf.divide(q_v + tf.square(q_m - prior_m),
                                     2 * prior_v) - 0.5)
            elif self.w_config[var_key]['type'] == 'binary':
                priors = self.w_config[var_key]['priors']
                if sum(priors) != 1:
                    raise Exception('prior probabilities are not normalized')

                if self.layer_config['parametrization'] == 'sigmoid':
                    prob_1 = tf.nn.sigmoid(self.var_dict[var_key + '_sb'])
                    probs = [1. - prob_1, prob_1]
                elif self.layer_config['parametrization'] == 'logits':
                    probs = tf.nn.softmax([self.var_dict[var_key + '_log_neg'], self.var_dict[var_key + '_log_pos']])

                epsilon = 1e-20
                kl_loss += tf.reduce_sum(probs[0] * tf.log(epsilon + tf.divide(probs[0], priors[0])) +
                                         probs[1] * tf.log(epsilon + tf.divide(probs[1], priors[1])))
            elif self.w_config[var_key]['type'] == 'ternary':
                priors = self.w_config[var_key]['priors']
                if sum(priors) != 1:
                    raise Exception('prior probabilities are not normalized')

                if self.layer_config['parametrization'] == 'sigmoid':
                    probs = get_ternary_probs(self.var_dict[var_key + '_sa'], self.var_dict[var_key + '_sb'])
                elif self.layer_config['parametrization'] == 'logits':
                    probs = tf.nn.softmax([self.var_dict[var_key + '_log_neg'], self.var_dict[var_key + '_log_zer'],
                                           self.var_dict[var_key + '_log_pos']])

                epsilon = 1e-20
                kl_loss += tf.reduce_sum(probs[0] * tf.log(epsilon + tf.divide(probs[0], priors[0])) +
                                         probs[1] * tf.log(epsilon + tf.divide(probs[1], priors[1])) +
                                         probs[2] * tf.log(epsilon + tf.divide(probs[2], priors[2])))
            else:
                raise Exception('weight type {} not understood'.format(self.w_config[var_key]['type']))
        return tf.cast(kl_loss, dtype=tf.float64)

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
