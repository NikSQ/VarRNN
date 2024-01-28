from copy import deepcopy
import os

import sys
sys.path.insert(0, '/clusterFS/home/student/kopp13/VarRNN')


from src.configuration.data_config import DataConfig
from src.configuration.nn_config import NNConfig, LSTMLayerConfig, FFLayerConfig, InputLayerConfig
from src.configuration.weight_config import WeightConfig, GaussianWeightConfig, DiscreteWeightConfig
from src.configuration.train_config import TrainConfig, LayerTrainConfig
from src.configuration.info_config import InfoConfig
from src.configuration.constants import WeightC, NetworkC, AlgorithmC

from src.network_gpu.experiment import Experiment

from src.data.loader import load_api_datasets
from src.data.t_metrics import save_to_file, print_results

try:
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
except KeyError:
    print('NO SLURM TASK ID FOUND')
    print('Using ID 0 instead')
    task_id = 0

runs = 3
epochs = 300
#
#
# =====================================================================================================================
# DATASET CONFIGURATION
# =====================================================================================================================
#
#

data_config = DataConfig()
data_config.add_mnist_small()

#
#
# =====================================================================================================================
# MODEL CONFIGURATION
# =====================================================================================================================
#
#
nn_config = NNConfig()

# Specification of weight properties which are then used in layers
b_config = GaussianWeightConfig()
bi_config = GaussianWeightConfig().set_initializers(mean_initializer=-1.5)
w_config = DiscreteWeightConfig(dist=WeightC.BINARY, parametrization=WeightC.SIGMOID)
#w_config = GaussianWeightConfig().set_initializers(mean_initializer=WeightC.XAVIER_INIT)


# For each layer, properties of weight and configuration function is set
lstm_layer_config = LSTMLayerConfig()\
    .set_gates_config(gates=[NetworkC.INPUT_GATE],
                      weight_config=w_config, bias_config=bi_config)\
    .set_gates_config(gates=[NetworkC.OUTPUT_GATE, NetworkC.CANDIDATE_GATE],
                      weight_config=w_config, bias_config=b_config)\
    .set_act_funcs_codomain(discrete_gates=[],
                            continuous_gates=[NetworkC.INPUT_GATE, NetworkC.CANDIDATE_GATE, NetworkC.OUTPUT_GATE])

ff_layer_config = FFLayerConfig()\
    .set_weight_config(weight_config=w_config, bias_config=b_config)

nn_config.add_layer_config(n_neurons=4, layer_config=InputLayerConfig())
nn_config.add_layer_config(n_neurons=50, layer_config=lstm_layer_config, var_scope="lstm_1")
nn_config.add_layer_config(n_neurons=30, layer_config=lstm_layer_config, var_scope="lstm_2")
nn_config.add_layer_config(n_neurons=10, layer_config=ff_layer_config, var_scope="ff_3")

#
#
# =====================================================================================================================
# TRAINING CONFIGURATION
# =====================================================================================================================
#
#
lstm_train_config = LayerTrainConfig().set_config(layer_norm_enabled=False,
                                                  dropout_enabled=False,
                                                  p_dropout=.95)
ff_train_config = LayerTrainConfig().set_config(layer_norm_enabled=False,
                                                dropout_enabled=False,
                                                p_dropout=.95)
#algos = [AlgorithmC.ARM, AlgorithmC.AR, AlgorithmC.LOG_DERIVATIVE, AlgorithmC.LOG_DERIVATIVE]
n_forward_passes = [1, 10, 100][int((task_id-1)/3)]
if task_id == 0:
    algorithm = AlgorithmC.REPARAMETRIZATION
else:
    algorithm = [AlgorithmC.ARM, AlgorithmC.AR, AlgorithmC.LOG_DERIVATIVE][(task_id -1) % 3]


train_config = TrainConfig(task_id=task_id)\
    .set_learning_rate(learning_rate=0.1)\
    .set_algorithm(algorithm=algorithm, n_forward_passes=n_forward_passes, ste_type=AlgorithmC.NO_STE)\
    .add_layer_train_config(var_scope=["lstm_1", "lstm_2"], layer_train_config=lstm_train_config) \
    .add_layer_train_config(var_scope=["ff_3"], layer_train_config=ff_train_config)


#
#
# =====================================================================================================================
# INFO CONFIGURATION
# =====================================================================================================================
#
#
info_config = InfoConfig(filename="exp", timer_enabled=False, profiling_enabled=False)
info_config.add_model_saver(filename="exp", task_id=task_id)
info_config.add_training_metrics_saver(path="../m/nr/")

exp = Experiment(nn_config=nn_config,
                 data_config=data_config,
                 train_config=train_config,
                 info_config=info_config,
                 task_id=task_id)
exp.train_multiple_runs(epochs, runs)

# Tested: Reparametrization and local reparametrization on gaussian weights, sigmoid binary / ternary and logistic binary / ternary (no ste)
