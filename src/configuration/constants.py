
class WeightC:
    CONFIG = "weight config"

    TYPE = "type"
    CONTINUOUS = "continuous"
    BINARY = "binary"
    TERNARY = "ternary"

    PARAMTERIZATION = "parametrization"
    SIGMOID = "sigmoid"
    LOGIT = "logit"

    MEAN_PRIOR = "mean prior"
    VARIANCE_PRIOR = "variance prior"

    MEAN_INIT = "mean initializer"
    VAR_INIT = "variance initializer"
    XAVIER = "xavier"
    PMAX_INIT = "pmax init"
    PMIN_INIT = "pmin init"

class AlgorithmC:
    CONFIG = "algorithm"
    AR = "ar"
    ARM = "arm"
    LOG_DERIVATIVE = "log derivative"
    GUMBEL_STE = "gumbel STE"
    STE = "STE"
    PFP = "probabilistic forward pass"
    LOCAL_REPARAMETRIZATION = "local reparametrization"
    REPARAMETRIZATION = "reparametrization"


class DiscreteActivationsC:
    I = "input"
    C = "candidate"
    O = "output"


