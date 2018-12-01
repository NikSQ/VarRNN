import tensorflow as tf
from src.fp_tools import approx_activation, sample_activation
from src.weights import Weights


# FC LAYER..
# Can only be used as output
class FCLayer:
    def __init__(self, rnn_config, layer_idx):
        self.rnn_config = rnn_config
        self.layer_config = rnn_config['layer_configs'][layer_idx]
        self.w_shape = (rnn_config['layout'][layer_idx-1], rnn_config['layout'][layer_idx])
        self.b_shape = (1, self.w_shape[1])
        self.acts = {'n': None}

        with tf.variable_scope(self.layer_config['var_scope']):
            var_keys = ['w', 'b']
            self.weights = Weights(var_keys, self.layer_config, self.w_shape, self.b_shape)

            #self.v_loss = tf.nn.l2_loss(self.weights.var_dict['b_v']) + tf.nn.l2_loss(self.weights.var_dict['w_v'])
            self.v_loss = 0

    # Returns the output of the layer. If its the output layer, this only returns the activation
    # TODO: Implement it for non-ouput case
    def create_pfp(self, x_m, x_v, mod_layer_config, init):
        a_m, a_v = approx_activation(self.weights.var_dict['w_m'], self.weights.var_dict['w_v'], self.weights.var_dict['b_m'], self.weights.var_dict['b_v'], x_m, x_v)
        if self.layer_config['is_output']:
            return a_m, a_v
        else:
            raise Exception('fc layer can only be used as output')

    def create_l_sampling_pass(self, x, mod_layer_config, init):
        return sample_activation(self.weights.var_dict['w_m'], self.weights.var_dict['w_v'], self.weights.var_dict['b_m'], self.weights.var_dict['b_v'], x)

    def create_g_sampling_pass(self, x, mod_layer_config, init):
        if init:
            self.weights.create_tensor_samples()
        if self.layer_config['is_output']:
            return tf.matmul(x, self.weights.tensor_dict['w']) + self.weights.tensor_dict['b']

    def create_fp(self, x, init):
        act = tf.matmul(x, self.weights.var_dict['w']) + self.weights.var_dict['b']
        if init:
            self.acts['n'] = act
        else:
            self.acts['n'] = tf.concat([self.acts['n'], act], 0)

        if self.layer_config['is_output']:
            return act



