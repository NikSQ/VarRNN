from src.data.pathes import get_mnist_path


class DataConfig:
    def __init__(self):
        self.ds_configs = {}

    def add_mnist_small(self,
                        data_keys=["tr", "va", "te"],
                        key_map={"tr": "tr", "va": "va", "te": "te"},
                        minibatch_enabled=True,
                        minibatch_size=1024,
                        in_seq_len=45,
                        remove_bias=True):
        for data_key in data_keys:
            dataset_config = DatasetConfig(filename=get_mnist_path(key_map[data_key], processing_type='all'),
                                           data_key=data_key,
                                           in_seq_len=in_seq_len,
                                           minibatch_enabled=minibatch_enabled,
                                           minibatch_size=minibatch_size,
                                           remove_bias=remove_bias,
                                           do_shuffle=(key_map[data_key] == "tr"),
                                           shuffle_buffer=4096)
            self.ds_configs[data_key] = dataset_config

    def print_config(self):
        print("====================================")
        print("Dataset configuration")
        for key in self.ds_configs.keys():
            print("")
            self.ds_configs[key].print_config()


class DatasetConfig:
    def __init__(self, filename, data_key, in_seq_len, minibatch_enabled, minibatch_size, remove_bias,
                 do_shuffle, shuffle_buffer):
        self.filename = filename
        self.data_key = data_key
        self.in_seq_len = in_seq_len
        self.minibatch_enabled = minibatch_enabled
        self.minibatch_size = minibatch_size
        self.remove_bias = remove_bias
        self.do_shuffle = do_shuffle
        self.shuffle_buffer = shuffle_buffer

    def print_config(self):
        print("Filename: {}".format(self.filename))
        print("Key: {}, \tIn seq len: {}".format(self.data_key, self.in_seq_len))
        print("Minibatch:  {}, \tSize: {}".format(self.minibatch_enabled, self.minibatch_size))
        print("Target bias removed: {}, \tShuffle: {}".format(self.remove_bias, self.do_shuffle))
