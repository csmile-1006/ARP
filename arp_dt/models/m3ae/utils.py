import pprint
import random
import time

import absl.flags
import cloudpickle as pickle
import gcsfs
import numpy as np
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags

from .jax_utils import init_rng


class Timer(object):
    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, "automatically defined flag")
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, "automatically defined flag")
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, "automatically defined flag")
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, "automatically defined flag")
        else:
            raise ValueError("Incorrect value type")
    return kwargs


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    init_rng(seed)


def print_flags(flags, flags_def):
    logging.info(
        "Running training with hyperparameters: \n{}".format(
            pprint.pformat(["{}: {}".format(key, val) for key, val in get_user_flags(flags, flags_def).items()])
        )
    )


def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            if prefix is not None:
                output["{}.{}".format(prefix, key)] = val
            else:
                output[key] = val
    return output


def prefix_metrics(metrics, prefix):
    return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}


def load_pickle(path):
    if path.startswith("gs://"):
        with gcsfs.GCSFileSystem().open(path) as fin:
            data = pickle.load(fin)
    else:
        with open(path, "rb") as fin:
            data = pickle.load(fin)
    return data


def load_checkpoint(path):
    data = load_pickle(path)
    logging.info(
        "Loading checkpoint from %s, saved at step %d",
        path,
        data["step"],
    )
    return data


def image_float2int(image):
    return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)


def create_log_images(images, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), n=5):
    images = [np.array(x) for x in images]
    images = [x.reshape(-1, *x.shape[2:]) for x in images]
    rows = np.concatenate(images, axis=2)
    rows = rows * std + mean
    return image_float2int(np.concatenate(rows[:n], axis=0))
