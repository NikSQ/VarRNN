class InfoConfig:
    def __init__(self,
                 filename="exp",
                 timer_enabled=False,
                 profiling_enabled=False,
                 profiling_path=None,
                 compute_tmetrics_every=1):
        self.filename = filename

        self.timer_enabled = timer_enabled
        self.profiling_enabled = profiling_enabled
        self.profiling_path = profiling_path

        self.save_weights = False
        self.save_weights_every = 1
        self.save_best_weights = False

        self.store_cell_access = False

        self.compute_tmetrics_every = compute_tmetrics_every

        self.tensorboard_config = TensorboardConfig()
        self.gradient_variance_config = GradientVarianceConfig()

        self.force_loading = False
        self.model_loader_config = None
        self.model_saver_config = None

        self.save_training_metrics = False
        self.training_metrics_path = ""

    def add_model_saver(self, filename, task_id=None):
        self.model_saver_config = ModelStorageConfig(filename=filename, task_id=task_id)

    def add_model_loader(self, filename, task_id=None, force_loading=True):
        self.force_loading = force_loading
        self.model_loader_config = ModelStorageConfig(filename=filename, task_id=task_id)

    def add_training_metrics_saver(self, path):
        self.save_training_metrics = True
        self.training_metrics_path = path

    def print_config(self):
        print("====================================")
        print("Info configuration")
        print("")
        print("Filename:  {}".format(self.filename))
        print("Timer: {}, \tProfiling: {}, \tProfiling path: {}".format(self.timer_enabled, self.profiling_enabled, self.profiling_path))
        print("Save training metrics: {}, \tPath: {}".format(self.save_training_metrics, self.training_metrics_path))
        print("Save weights: {}, \tEvery: {}, \tSave best weights: {}".format(self.save_weights, self.save_weights_every, self.save_best_weights))
        print("Compute T Metrics every: {}".format(self.compute_tmetrics_every))
        print("")
        if self.model_loader_config is not None:
            print("Load model from {}, \t Force it: {}".format(self.model_loader_config.filename, self.force_loading))
        else:
            print("No model loading")
        if self.model_saver_config is not None:
            print("Store model to " + self.model_saver_config.filename)
        else:
            print("No model saving")
        print("")


class ModelStorageConfig:
    def __init__(self, filename, task_id):
        self.filename = filename
        self.task_id = task_id

    def create_path(self):
        if self.task_id is None:
            return '../tr_models/' + self.filename
        else:
            return '../models/' + self.filename + '_' + str(self.task_id)



class GradientVarianceConfig:
    def __init__(self):
        self.enabled = False
        self.n_gradients = 1
        self.n_gradients_per_sample = 1


class TensorboardConfig:
    def __init__(self):
        self.enabled = False
        self.save_weights = False
        self.save_gradients = False
        self.save_results = False
        self.save_activations = False
        self.record_n_neurons = 0
        self.path = ""