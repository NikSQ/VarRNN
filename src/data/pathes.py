import os


def create_dirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def get_mnist_path(key, processing_type="all"):
    home_dir = os.path.expanduser("~")
    return os.path.join(home_dir, "PycharmProjects", "VarRNN", "datasets", "mnist_pen_strokes",
                        "mps_" + processing_type + "_" + key + ".mat")


