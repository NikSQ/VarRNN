import tensorflow as tf
import numpy as np
import copy


def get_mean_initializer(w_config, shape):
    if w_config['init_m'] == 'xavier':
        init_vals = np.random.randn(shape[0], shape[1]) * np.sqrt(2/sum(shape))
    elif w_config['init_m'] == 'same':
        init_vals = np.ones(shape) * w_config['prior_m']
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


class Weights:
    def __init__(self, var_keys, layer_config, w_shape, b_shape):
        self.var_keys = var_keys
        self.gauss = tf.distributions.Normal(loc=0., scale=1.)
        self.uniform = tf.distributions.Uniform(low=0., high=1.)
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

    def create_vars(self):
        kl_loss = 0
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
                self.var_dict[var_key + '_sb'] = tf.get_variable(name=var_key + '_sb', shape=shape,
                                                                 initializer=tf.zeros_initializer(), dtype=tf.float32)
                weight_summaries.append(tf.summary.histogram(var_key + '_sb', self.var_dict[var_key + '_sb']))
            # p(w=0) = sa, p(w=1 | w !=0) = sb -> from paper 1710.07739
            elif self.w_config[var_key]['type'] == 'ternary':
                self.var_dict[var_key + '_sa'] = tf.get_variable(name=var_key + '_sa', shape=shape,
                                                                 initializer=tf.zeros_initializer(), dtype=tf.float32)
                self.var_dict[var_key + '_sb'] = tf.get_variable(name=var_key + '_sb', shape=shape,
                                                                 initializer=tf.zeros_initializer(), dtype=tf.float32)
                weight_summaries.append(tf.summary.histogram(var_key + '_sa', self.var_dict[var_key + '_sa']))
                weight_summaries.append(tf.summary.histogram(var_key + '_sb', self.var_dict[var_key + '_sb']))
            else:
                raise Exception('weight type {} not understood'.format(self.w_config[var_key]['type']))

            sample_ops.append(tf.assign(self.var_dict[var_key], self.generate_sample(var_key, True)))

        self.sample_op = tf.group(*sample_ops)
        self.weight_summaries = tf.summary.merge(weight_summaries)

    def generate_sample(self, var_key, exact=False):
        shape = self.var_dict[var_key].shape
        if self.w_config[var_key]['type'] == 'continuous':
            return self.var_dict[var_key + '_m'] + self.gauss.sample(shape) * tf.square(self.var_dict[var_key + '_v'])
        elif self.w_config[var_key]['type'] == 'binary':
            if exact:
                return -1. + 2. * tf.cast(tf.argmax([-self.var_dict[var_key + '_sb']
                                                     - tf.log(-tf.log(self.uniform.sample(shape))),
                                                     -tf.log(-tf.log(self.uniform.sample(shape)))]), dtype=tf.float32)
            else:
                return tf.nn.tanh((self.var_dict[var_key + '_sb']
                                   - tf.log(-tf.log(self.uniform.sample(shape)))
                                   + tf.log(-tf.log(self.uniform.sample(shape))))
                                  / self.layer_config['tau'])
        elif self.w_config[var_key]['type'] == 'ternary':
            # TODO: Implement ternary sampling
            return
        else:
            raise Exception('weight type {} not understood'.format(self.w_config[var_key]['type']))

    def create_tensor_samples(self, suffix=''):
        for var_key in self.var_keys:
            self.tensor_dict[var_key+suffix] = self.generate_sample(var_key, False)

    def create_init_op(self):
        init_ops = []
        for var_key in self.var_keys:
            if self.w_config[var_key]['type'] == 'continuous':
                init_ops.append(tf.assign(self.var_dict[var_key + '_m'], self.var_dict[var_key]))
            elif self.w_config[var_key]['type'] == 'binary':
                expectation = tf.maximum(self.var_dict[var_key], self.w_config[var_key]['pmin'])
                expectation = tf.minimum(expectation, self.w_config[var_key]['pmax'])
                init_ops.append(tf.assign(self.var_dict[var_key + '_sb'], -tf.log((1 - expectation)/(1 + expectation))))
            elif self.w_config[var_key]['type'] == 'ternary':
                raise Exception()
            else:
                raise Exception('weight type {} not understood'.format(self.w_config[var_key]['type']))
        self.init_op = tf.group(*init_ops)

    def get_beta_reg(self):
        beta_reg = 0
        count = 0
        for var_key in self.var_keys:
            if self.w_config[var_key]['type'] == 'binary':
                exp = tf.exp(-self.var_dict[var_key + '_sb'])
                beta_reg += tf.reduce_mean(tf.divide(exp, tf.square(1 + exp)))
                count += 1
        return tf.cast(beta_reg / count, dtype=tf.float64)

    def get_pretraining_reg(self):
        l1_regu = tf.contrib.layers.l1_regularizer(scale=4.0)
        reg_term = 0
        count = 0
        for var_key in self.var_keys:
            if self.w_config[var_key]['type'] == 'binary':
                reg_term += tf.nn.l2_loss(self.var_dict[var_key] - 1) + \
                           tf.nn.l2_loss(self.var_dict[var_key] + 1) - l1_regu(self.var_dict[var_key])
                count += 1
        return tf.cast(reg_term / count, dtype=tf.float64)

    # Regularizing posterior variance
    def get_var_reg(self):
        var_reg = 0
        var_counter = 0
        for var_key in self.var_keys:
            if self.w_config[var_key]['type'] == 'continuous':
                m, v = self.get_stats(var_key)
                var_reg += tf.reduce_mean(v)
                var_counter += 1
        return tf.cast(var_reg / var_counter, dtype=tf.float64)

    def get_stats(self, var_key):
        if self.w_config[var_key]['type'] == 'continuous':
            m = self.var_dict[var_key + '_m']
            v = self.var_dict[var_key + '_v']
        elif self.w_config[var_key]['type'] == 'binary':
            m = tf.nn.tanh(self.var_dict[var_key + '_sb'] / 2)
            v = tf.divide(1, tf.cosh(self.var_dict[var_key + '_sb']))
        else:
            raise Exception()
        return m, v

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
                prob = tf.nn.sigmoid(self.var_dict[var_key + '_sb'])
                probs = [1 - prob, prob]
                kl_loss += tf.reduce_sum(priors[0] * tf.log(tf.divide(priors[0], probs[0])) +
                                     priors[1] * tf.log(tf.divide(priors[1], probs[1])))
            elif self.w_config[var_key]['type'] == 'ternary':
                raise Exception('weight type {} not understood'.format(self.w_config[var_key]['type']))
            else:
                raise Exception('weight type {} not understood'.format(self.w_config[var_key]['type']))
        return tf.cast(kl_loss, dtype=tf.float64)

    def sample_activation(self, w_var_key, b_var_key, x_m):
        w_m, w_v = self.get_stats(w_var_key)
        b_m, b_v = self.get_stats(b_var_key)

        mean = tf.matmul(x_m, w_m) + b_m
        std = tf.sqrt(tf.matmul(tf.square(x_m), w_v) + b_v)
        shape = (tf.shape(x_m)[0], tf.shape(b_m)[1])
        return mean + tf.multiply(self.gauss.sample(sample_shape=shape), std)


