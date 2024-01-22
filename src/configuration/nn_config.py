from copy import deepcopy

from src.configuration.constants import NetworkC

DEFAULT_BIAS_ENABLED = True
DEFAULT_IS_ACT_FUNC_DISCRETE = False
DEFAULT_N_ACT_BINS = 2


class NNConfig:
    def __init__(self):
        self.layout = []
        self.layer_configs = []

    def add_layer_config(self, n_neurons, layer_config, var_scope="", copy_config=True):
        if copy_config:
            layer_config = deepcopy(layer_config)
        layer_config.set_var_scope(var_scope)
        layer_config.set_n_neurons(n_neurons)
        self.layer_configs.append(layer_config)
        self.layout.append(layer_config.n_neurons)

    def print_config(self):
        print("====================================")
        print("RNN configuration")

        for idx, layer in enumerate(self.layer_configs):
            print("")
            print("Layer #" + idx + ", \tType: " + layer.layer_type + ", \tScope: " + layer.var_scope)
            print("Units: " + layer.n_neurons + ", \tN activation bins: " + layer.n_activation_bins)
            layer.print_gate_configs()


class LayerConfig:
    def __init__(self, layer_type):
        self.n_activation_bins = None
        self.layer_type = layer_type
        self.var_scope = None
        self.n_neurons = None
        self.gate_configs_map = {}

    def set_var_scope(self, var_scope):
        self.var_scope = var_scope

    def set_n_neurons(self, n_neurons):
        self.n_neurons = n_neurons

    def get_gate_config(self, var_key):
        if len(var_key) == 2:
            return self.gate_configs_map[var_key[1]]
        else:
            return self.gate_config

    def print_gate_configs(self):
        for key in self.gate_configs_map.keys():
            print(self.gate_configs_map[key].print_config())



class InputLayerConfig(LayerConfig):
    def __init__(self):
        super().__init__(layer_type=NetworkC.INPUT_LAYER)


class LSTMLayerConfig(LayerConfig):
    def __init__(self):
        super().__init__(layer_type=NetworkC.LSTM_LAYER)

        self.i_gate_config = FFGateConfig()
        self.c_gate_config = FFGateConfig()
        self.o_gate_config = FFGateConfig()

        self.gate_map = {NetworkC.INPUT_GATE: self.i_gate_config,
                         NetworkC.CANDIDATE_GATE: self.c_gate_config,
                         NetworkC.OUTPUT_GATE: self.o_gate_config}

        self.gate_configs_map = {"i": self.i_gate_config,
                                 "o": self.o_gate_config,
                                 "c": self.c_gate_config}

    def get_weight_config(self, var_key, do_copy):
        if var_key == "wi":
            config = self.i_gate_config.w_config
        elif var_key == "wc":
            config = self.c_gate_config.w_config
        elif var_key == "wo":
            config = self.o_gate_config.w_config
        elif var_key == "bi":
            config = self.i_gate_config.b_config
        elif var_key == "bc":
            config = self.c_gate_config.b_config
        elif var_key == "bo":
            config = self.o_gate_config.b_config
        else:
            raise Exception()
        if do_copy:
            config = deepcopy(config)
        return config

    def set_all_weight_configs(self, weight_config=None, bias_config=None):
        self.set_gates_config(list(self.gate_map.keys()), weight_config, bias_config)
        return self

    def set_gates_config(self, gates, weight_config=None, bias_config=None):
        if type(gates) is not list:
            gates = [gates]

        for gate in gates:
            self.gate_map[gate].set_w_config(weight_config)
            self.gate_map[gate].set_b_config(bias_config)
        return self

    def set_act_funcs_codomain(self, discrete_gates=[], continuous_gates=[]):
        if not set(discrete_gates).isdisjoint(continuous_gates):
            raise Exception("Gate was set to be discrete and continuous at the same time")

        for gate in discrete_gates:
            self.gate_map[gate].set_act_func_codomain(True)

        for gate in continuous_gates:
            self.gate_map[gate].set_act_func_codomain(False)

        return self


class FFLayerConfig(LayerConfig):
    def __init__(self):
        super().__init__(layer_type=NetworkC.FEED_FORWARD_LAYER)

        self.gate_config = FFGateConfig()

    def get_weight_config(self, var_key, do_copy):
        if var_key == "w":
            config = self.gate_config.w_config
        elif var_key == "b":
            config = self.gate_config.b_config
        else:
            raise Exception()

        if do_copy:
            config = deepcopy(config)
        return config

    def set_weight_config(self, weight_config=None, bias_config=None):
        self.gate_config.set_w_config(weight_config)
        self.gate_config.set_b_config(bias_config)
        return self


# Stores relevant configurations for a feed-forward components. This can be a gate in
# an LSTM layer or all neurons in a feed-forward layer
class FFGateConfig:
    def __init__(self):
        self.w_config = None
        self.b_config = None
        self.is_act_func_discrete = DEFAULT_IS_ACT_FUNC_DISCRETE
        self.n_act_bins = DEFAULT_N_ACT_BINS
        self.bias_enabled = DEFAULT_BIAS_ENABLED

    def set_w_config(self, weight_config=None):
        if weight_config is not None:
            self.w_config = deepcopy(weight_config)

    def set_b_config(self, bias_config=None):
        if bias_config is not None:
            self.b_config = deepcopy(bias_config)

    def set_act_func_codomain(self, is_discrete):
        self.is_act_func_discrete = is_discrete

    def set_n_act_bins(self, n_act_bins):
        self.n_act_bins = n_act_bins

    def set_bias_enabled(self, bias_enabled):
        self.bias_enabled = bias_enabled

    def print_config(self):
        print("Bias: " + self.bias_enabled + ", \tN activation bins: " + self.n_act_bins + ", \tDiscrete activation: " + self.is_act_func_discrete)
        if self.w_config is not None:
            self.w_config.print_config()
        if self.b_config is not None:
            self.b_config.print_config()