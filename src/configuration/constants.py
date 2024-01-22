
class WeightC:
    GAUSSIAN = "gaussian"
    BINARY = "binary"
    TERNARY = "ternary"

    SIGMOID = "sigmoid"
    LOGIT = "logit"

    XAVIER_INIT = "xavier"
    BINARY_INIT = "binary"


class VarNames:
    LOGITS_NEG = "_log_neg"
    LOGITS_ZER = "_log_zer"
    LOGITS_POS = "_log_pos"

    SIGMOID_A = "_sa"
    SIGMOID_B = "_sb"

    GAUSSIAN_MEAN = "_m"
    GAUSSIAN_VAR = "_v"

    ARM = '_arm'


class AlgorithmC:
    AR = "ar"
    ARM = "arm"
    LOG_DERIVATIVE = "log derivative"
    PFP = "probabilistic forward pass"
    LOCAL_REPARAMETRIZATION = "local reparametrization"
    REPARAMETRIZATION = "reparametrization"

    NO_STE = "disabled"
    GUMBEL_STE = "gumbel"
    CUSTOM_LOGIT_STE = "custom logit"


class DiscreteActivationsC:
    I = "input"
    C = "candidate"
    O = "output"


class ActivationFunctionsC:
    SIGMOID = "sigmoid"
    TANH = "tanh"


class NetworkC:
    INPUT_LAYER = "input_layer"
    LSTM_LAYER = "lstm layer"
    FEED_FORWARD_LAYER = "ff layer"

    INPUT_GATE = "input gate"
    CANDIDATE_GATE = "candidate gate"
    OUTPUT_GATE = "output gate"
    CELL_STATE = "cell state"
    CELL_INPUT = "cell input"
    CELL_OUTPUT = "cell output"


class GraphCreationKeys:
    CELL_INITIALIZATION = "cell_initialization"
    MODIFICATED_LAYER_CONFIG = "modificated_layer_config"
    SECOND_ARM_PASS = "second_arm_pass"
    DATA_KEY = "data_key"


class DatasetKeys:
    X = "x"
    Y = "y"
    SEQLEN = "seqlen"

    X_SHAPE = "x_shape"
    Y_SHAPE = "y_shape"
    X_PLACEHOLDER = "x_ph"
    Y_PLACEHOLDER = "y_ph"

    SAMPLE_LIST = "sample_list"
    N_MINIBATCHES = "n_minibatches"

    OP_LOAD = "load"
    OP_SHUFFLE = "shuffle"

    TR_SET = "tr"
    VA_SET = "va"
    TE_SET = "te"





