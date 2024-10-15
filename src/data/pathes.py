import os


def create_dirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def get_mnist_path(key, processing_type="all"):
    home_dir = os.path.expanduser("~")
    #return os.path.join(home_dir, "PycharmProjects", "VarRNN", "datasets", "mnist_pen_strokes",
                        #"mps_" + processing_type + "_" + key + ".mat")
    #return os.path.join(home_dir, "datasets", "mnist_pen_strokes", "mps_" + processing_type + "_" + key + ".mat")
    return os.path.join(home_dir, "workspace", "mnist", "mnist_pen_strokes", "mps_" + processing_type + "_" + key + ".mat")

def get_timit_path(key, processing_type="all"):
    home_dir = os.path.expanduser("~")
    #return os.path.join(home_dir, "PycharmProjects", "VarRNN", "datasets", "mnist_pen_strokes",
                        #"mps_" + processing_type + "_" + key + ".mat")
    #return os.path.join(home_dir, "datasets", "mnist_pen_strokes", "mps_" + processing_type + "_" + key + ".mat")
    return os.path.join(home_dir, "workspace", "timit", "timit" + "_" + key + ".mat")

def get_sign_path(key, processing_type="all"):
    home_dir = os.path.expanduser("~")
    #return os.path.join(home_dir, "PycharmProjects", "VarRNN", "datasets", "mnist_pen_strokes",
                        #"mps_" + processing_type + "_" + key + ".mat")
    #return os.path.join(home_dir, "datasets", "mnist_pen_strokes", "mps_" + processing_type + "_" + key + ".mat")
    return os.path.join(home_dir, "workspace", "sign", "sign_" + key + ".mat")





