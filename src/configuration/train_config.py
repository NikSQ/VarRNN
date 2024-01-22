import numpy as np
from src.configuration.constants import AlgorithmC


DEFAULT_LEARNING_RATE = .2
DEFAULT_LEARNING_RATE_ANNEAL_PERIOD = np.inf
DEFAULT_GUMBEL_TAU = 1.5

DEFAULT_STE_TYPE = AlgorithmC.NO_STE
DEFAULT_N_FORWARD_PASSES = 1

DEFAULT_VARIANCE_REGULARIZATION = 0.
DEFAULT_ENTROPY_REGULARIZATION = 0.
DEFAULT_DIRICHLET_REGULARIZATION = 0.

DEFAULT_GRADIENT_CLIP_VALUE = 1.0
DEFAULT_GRADIENT_CLIPPING_ENABLED = False

DEFAULT_LAYER_NORM_ENABLED = False
DEFAULT_DROPOUT_ENABLED = False
DEFAULT_P_DROPOUT = .05


class TrainConfig:
    def __init__(self, task_id, data_multiplier=None):
        self.learning_rate = DEFAULT_LEARNING_RATE
        self.learning_rate_anneal_period = DEFAULT_LEARNING_RATE_ANNEAL_PERIOD
        self.gumbel_tau = DEFAULT_GUMBEL_TAU

        self.algorithm = None
        self.ste_type = DEFAULT_STE_TYPE
        self.n_forward_passes = DEFAULT_N_FORWARD_PASSES

        self.activation_normalization = None
        self.variance_regularization = DEFAULT_VARIANCE_REGULARIZATION
        self.entropy_regularization = DEFAULT_ENTROPY_REGULARIZATION
        self.dirichlet_regularization = DEFAULT_DIRICHLET_REGULARIZATION
        self.data_multiplier = data_multiplier

        self.gradient_clipping_enabled = DEFAULT_GRADIENT_CLIPPING_ENABLED
        self.gradient_clip_value = DEFAULT_GRADIENT_CLIP_VALUE

        self.training_mode = None
        self.task_id = task_id

        self.layer_train_configs = {}
        self.pretraining_reg = 0.

    def set_load(self, path):
        self.load_model = True
        self.load_from_path = path

    def set_learning_rate(self,
                          learning_rate,
                          learning_rate_anneal_period=DEFAULT_LEARNING_RATE_ANNEAL_PERIOD):
        self.learning_rate = learning_rate
        self.learning_rate_anneal_period = learning_rate_anneal_period
        return self

    def set_algorithm(self,
                      algorithm,
                      n_forward_passes=DEFAULT_N_FORWARD_PASSES,
                      ste_type=DEFAULT_STE_TYPE):
        self.algorithm = algorithm
        self.n_forward_passes = n_forward_passes
        self.ste_type = ste_type
        return self

    def set_variance_regulariztion(self, lambda_reg):
        self.variance_regularization = lambda_reg
        return self

    def set_entropy_regularization(self, lambda_reg):
        self.entropy_regularization = lambda_reg
        return self

    def set_dirichlet_regularization(self, lambda_reg):
        self.dirichlet_regularization = lambda_reg
        return self

    # Returns annealed learning rate
    def get_learning_rate(self, current_epoch=None):
        if current_epoch is None or self.learning_rate_anneal_period is None:
            return self.learning_rate

        n_anneal_steps = int(current_epoch / self.learning_rate_anneal_period)
        return self.learning_rate / float(2 ** n_anneal_steps)

    def set_gradient_clipping(self, gradient_clip_value=None):
        if gradient_clip_value is None:
            self.gradient_clipping_enabled = False
        else:
            self.gradient_clipping_enabled = False
            self.gradient_clip_value = gradient_clip_value
        return self

    def add_layer_train_config(self, var_scope, layer_train_config):
        if type(var_scope) == list:
            for v_scope in var_scope:
                self.layer_train_configs[v_scope] = layer_train_config
        else:
            self.layer_train_configs[var_scope] = layer_train_config
        return self

    def print_config(self):
        print("====================================")
        print("Training configuration")
        print("")
        print("Learning rate: " + self.learning_rate + ", \tLearning rate anneal: " + self.learning_rate_anneal_period + ", \tTau: " + self.gumbel_tau)
        print("Algorithm: " + self.algorithm + ", \tSTE: " + self.ste_type + ", \tn forward passes: " + self.n_forward_passes)
        print("Var reg: " + self.variance_regularization + ", \tDir reg: " + self.dirichlet_regularization + ", \tEnt reg: " + self.entropy_regularization)
        print("Data multiplier: " + self.data_multiplier)
        print("Gradient clipping: " + self.gradient_clipping_enabled + ", \tGradient clip value: " + self.gradient_clip_value)
        for idx, key in enumerate(self.layer_train_configs.keys()):
            print("")
            print("Layer #" + idx)
            print(key)
            self.layer_train_configs[key].print_config()


class LayerTrainConfig:
    def __init__(self):
        self.layer_norm_enabled = DEFAULT_LAYER_NORM_ENABLED
        self.dropout_enabled = DEFAULT_DROPOUT_ENABLED
        self.p_dropout = DEFAULT_P_DROPOUT
        self.lr_adapt = False

    def set_config(self,
                   layer_norm_enabled=DEFAULT_LAYER_NORM_ENABLED,
                   dropout_enabled=DEFAULT_DROPOUT_ENABLED,
                   p_dropout=DEFAULT_P_DROPOUT):
        self.layer_norm_enabled = layer_norm_enabled
        self.dropout_enabled = dropout_enabled
        self.p_dropout = p_dropout
        self.lr_adapt = False
        return self

    def print_config(self):
        print("Layer norm: " + self.layer_norm_enabled + ", \tLR Adapt: " + self.lr_adapt)
        print("Dropout: " + self.dropout_enabled + ", \tp_dropout: " + self.p_dropout)


# TODO these configs to NN
class LayerwiseBatchnormConfig:
    def __init__(self):
        self.enabled = False
        self.components = []
        self.tau = 1
