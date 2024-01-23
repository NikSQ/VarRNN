import numpy as np
from src.configuration.constants import WeightC

DEFAULT_MEAN_INITIALIZER = WeightC.XAVIER_INIT
DEFAULT_LOGVAR_INITIALIZER = WeightC.XAVIER_INIT
DEFAULT_MEAN_PRIOR = 0.
DEFAULT_LOGVAR_PRIOR = 0.

DEFAULT_DISCRETE_WEIGHT_DIST = WeightC.TERNARY
DEFAULT_DISCRETE_INITIALIZER = WeightC.XAVIER_INIT
DEFAULT_DISCRETE_WEIGHT_PARAMETRIZATION = WeightC.SIGMOID
DEFAULT_BINARY_PRIORS = [1/2] * 2
DEFAULT_TERNARY_PRIORS = [1/3] * 3

DEFAULT_FROM_PRETRAINED_INIT_P_MIN = 0.05
DEFAULT_FROM_PRETRAINED_INIT_P_MAX = 0.95


class WeightConfig:
    def __init__(self, dist):
        self.dist = dist


class GaussianWeightConfig(WeightConfig):
    def __init__(self):
        super().__init__(dist=WeightC.GAUSSIAN)

        self.mean_initializer = DEFAULT_MEAN_INITIALIZER
        self.logvar_initializer = DEFAULT_LOGVAR_INITIALIZER

        self.mean_prior = DEFAULT_MEAN_PRIOR
        self.logvar_prior = DEFAULT_LOGVAR_PRIOR

    def set_priors(self, mean_prior=None, logvar_prior=None):
        if mean_prior is not None:
            self.mean_prior = mean_prior
        if logvar_prior is not None:
            self.logvar_prior = logvar_prior
        return self

    def set_initializers(self, mean_initializer=None, logvar_initializer=None):
        if mean_initializer is not None:
            self.mean_initializer = mean_initializer
        if logvar_initializer is not None:
            self.logvar_initializer = logvar_initializer
        return self

    def print_config(self):
        print("Type: " + str(self.dist) + ", \tM init: " + str(self.mean_initializer) + ", \tLog V init: " + str(self.logvar_initializer))
        print("M prior: " + str(self.mean_prior) + ", \t Log V prior: " + str(self.logvar_prior))


class DiscreteWeightConfig(WeightConfig):
    def __init__(self,
                 dist=DEFAULT_DISCRETE_WEIGHT_DIST,
                 parametrization=DEFAULT_DISCRETE_WEIGHT_PARAMETRIZATION):
        super().__init__(dist=dist)

        if dist == WeightC.BINARY:
            self.priors = DEFAULT_BINARY_PRIORS
        elif dist == WeightC.TERNARY:
            self.priors = DEFAULT_TERNARY_PRIORS
        else:
            raise Exception("DiscreteWeightConfig does not support weight distribution " + str(dist))

        self.parametrization = parametrization
        self.from_pretrained_init_p_min = DEFAULT_FROM_PRETRAINED_INIT_P_MIN
        self.from_pretrained_init_p_max = DEFAULT_FROM_PRETRAINED_INIT_P_MAX

        self.sa_initializer = DEFAULT_DISCRETE_INITIALIZER
        self.sb_initializer = DEFAULT_DISCRETE_INITIALIZER
        self.log_neg_initializer = DEFAULT_DISCRETE_INITIALIZER
        self.log_zer_initializer = DEFAULT_DISCRETE_INITIALIZER
        self.log_pos_initializer = DEFAULT_DISCRETE_INITIALIZER

    def print_config(self):
        print("Type: " + str(self.dist) + ", \tParametetrization: " + str(self.parametrization) + ", \tPriors: " + str(self.priors))
        print("p_init_min: " + str(self.from_pretrained_init_p_min) + ", \tp_init_max: " + str(self.from_pretrained_init_p_max))
        if self.parametrization == WeightC.SIGMOID:
            print("SB init: " + str(self.sb_initializer) + ", \tSA init: " + str(self.sa_initializer))
        else:
            print("Neg init: " + str(self.log_neg_initializer) + ", \tZer init: " + str(self.log_zer_initializer) + ", \tPos init: " + str(self.log_pos_initializer))

    # Sets a list of priors for the possible weight values.
    # The argument can also be a positive scalar, which defines how much larger the prior
    # of one weight value will be over the other. In the ternary case, this weight value is the
    # zero weight.
    def set_priors(self, priors):
        if type(priors) == list:
            if (self.dist == WeightC.BINARY and len(priors) != 2) or \
                    (self.dist == WeightC.TERNARY and len(priors) != 3):
                raise Exception("Weight type was set to " + str(self.dist) +
                                "which is incompatible with given priors " + str(priors))
            if True in [p < 0 for p in priors]:
                raise Exception("Priors need to be positive, but were " + str(priors))

            self.priors = [p / sum(priors) for p in priors]
        elif type(priors) in [int, float]:
            priors = float(priors)
            if priors < 0:
                raise Exception("Priors can be set with either a list or positive scalar. Got " + str(priors) + " instead.")

            if self.dist == WeightC.BINARY:
                unscaled_priors = [priors, 1]
            elif self.dist == WeightC.TERNARY:
                unscaled_priors = [1, priors, 1]

            self.priors = [p / sum(unscaled_priors) for p in unscaled_priors]
        else:
            raise Exception("Argument priors has invalid type " + str(type(priors)))
        return self

    def set_logit_initializers(self, log_neg_initializer=None, log_zer_initializer=None, log_pos_initializer=None):
        if log_neg_initializer is not None:
            self.log_neg_initializer = log_neg_initializer
        if log_zer_initializer is not None:
            self.log_zer_initializer = log_zer_initializer
        if log_pos_initializer is not None:
            self.log_pos_initializer = log_pos_initializer
        return self

    def set_sigmoid_initializers(self, sa_initializer=None, sb_initializer=None):
        if sa_initializer is not None:
            self.sa_initializer = sa_initializer
        if sb_initializer is not None:
            self.sb_initializer = sb_initializer
        return self

    # This sets parameters for the initialization process for discrete weights given a continuous weight
    # This initialization method is used when using a pretrained deterministic continuous network_api
    def set_from_pretrained_init_parameters(self, p_min, p_max):
        self.from_pretrained_init_p_min = p_min
        self.from_pretrained_init_p_max = p_max
        return self
