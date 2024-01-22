class InfoConfig:
    def __init__(self,
                 filename="default",
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

    def add_model_saver(self, filename, task_id=None):
        self.model_saver_config = ModelStorageConfig(filename=filename, task_id=task_id)

    def add_model_loader(self, filename, task_id=None, force_loading=True):
        self.force_loading = force_loading
        self.model_loader_config = ModelStorageConfig(filename=filename, task_id=task_id)

    def print_config(self):
        print("====================================")
        print("Info configuration")
        print("")
        print(f"Filename: {self.filename}")
        print(f"Timer: {self.timer_enabled}, \tProfiling: {self.profiling_enabled}, \tProfiling path: {self.profiling_path}")
        print(f"Save weights: {self.save_weights}, \tEvery: {self.save_weights_every}, \tSave best weights: {self.save_best_weights}")
        print(f"Compute T Metrics every: {self.compute_tmetrics_every}")
        print("")
        if self.model_loader_config is not None:
            print(f"Load model from {self.model_loader_config.filename}, \t Force it: {self.force_loading}")
        else:
            print(f"No model loading")
        if self.model_saver_config is not None:
            print(f"Store model to {self.model_saver_config.filename}")
        else:
            print(f"No model saving")
        print("")


class ModelStorageConfig:
    def __init__(self, filename, task_id):
        self.filename = filename
        self.task_id = task_id

    def create_path(self):
        if self.task_id is None:
            return f'../tr_models/{self.filename}'
        else:
            return f'../models/{self.filename}_{self.task_id}'



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