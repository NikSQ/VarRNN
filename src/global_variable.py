train_config = None
info_config = None
rnn_config = None

train_config_init = False
info_config_init = False
rnn_config_init = False


def set_train_config(value):
    global train_config_init
    #if train_config_init:
        #raise Exception('Train config set more than once')
    train_config_init = True

    global train_config
    train_config = value


def set_info_config(value):
    global info_config_init
    #if info_config_init:
        #raise Exception('Train config set more than once')
    info_config_init = True

    global info_config
    info_config = value


def set_rnn_config(value):
    global rnn_config_init
    #if rnn_config_init:
        #raise Exception('Train config set more than once')
    rnn_config_init = True

    global rnn_config
    rnn_config = value


def get_train_config():
    return train_config


def get_info_config():
    return info_config


def get_rnn_config():
    return rnn_config
