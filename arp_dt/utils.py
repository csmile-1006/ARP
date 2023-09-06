import os
import pprint
import random
import tempfile
import time
import uuid
from copy import copy
from socket import gethostname

import absl.flags
import cloudpickle as pickle
import flax
import gcsfs
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags


class JaxRNG(object):
    """A convenient stateful Jax RNG wrapper. Can be used to wrap RNG inside
    pure function.
    """

    @classmethod
    def from_seed(cls, seed):
        return cls(jax.random.PRNGKey(seed))

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, keys=None):
        if keys is None:
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng
        elif isinstance(keys, int):
            split_rngs = jax.random.split(self.rng, num=keys + 1)
            self.rng = split_rngs[0]
            return tuple(split_rngs[1:])
        else:
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)
            self.rng = split_rngs[0]
            return {key: val for key, val in zip(keys, split_rngs[1:])}


def wrap_function_with_rng(rng):
    """To be used as decorator, automatically bookkeep a RNG for the wrapped function."""

    def wrap_function(function):
        def wrapped(*args, **kwargs):
            nonlocal rng
            rng, split_rng = jax.random.split(rng)
            return function(split_rng, *args, **kwargs)

        return wrapped

    return wrap_function


def init_rng(seed):
    global jax_utils_rng
    jax_utils_rng = JaxRNG.from_seed(seed)


def next_rng(*args, **kwargs):
    global jax_utils_rng
    return jax_utils_rng(*args, **kwargs)


def get_metrics(metrics, unreplicate=False):
    if unreplicate:
        metrics = flax.jax_utils.unreplicate(metrics)
    metrics = jax.device_get(metrics)
    return {key: float(val) for key, val in metrics.items()}


def extend_and_repeat(tensor, axis, repeat):
    return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)


def mse_loss(val, target):
    return jnp.mean(jnp.square(val - target))


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


class WandBLogger(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.online = False
        config.prefix = "instructrl-procgen"
        config.project = "instructrl-procgen"
        config.output_dir = "/tmp/instructrl_procgen"
        config.gcs_output_dir = ""
        config.random_delay = 0.0
        config.experiment_id = config_dict.placeholder(str)
        config.experiment_name = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)
        # config.entity = ""

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant, enable=True):
        self.enable = enable
        self.config = self.get_default_config(config)

        if self.config.experiment_name is None:
            self.config.experiment_name = uuid.uuid4().hex

        if self.config.experiment_id is None:
            self.config.experiment_id = uuid.uuid4().hex

        if self.config.prefix != "":
            self.config.experiment_id = "{}--{}".format(self.config.prefix, self.config.experiment_id)

        if self.enable:
            if self.config.output_dir == "":
                self.config.output_dir = tempfile.mkdtemp()
            else:
                self.config.output_dir = os.path.join(self.config.output_dir, self.config.experiment_id)
                os.makedirs(self.config.output_dir, exist_ok=True)

            if self.config.gcs_output_dir != "":
                self.config.gcs_output_dir = os.path.join(self.config.gcs_output_dir, self.config.experiment_id)

        self._variant = copy(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        if self.enable:
            self.run = wandb.init(
                reinit=True,
                config=self._variant,
                project=self.config.project,
                dir=self.config.output_dir,
                name=self.config.experiment_name,
                id=self.config.experiment_id,
                anonymous=self.config.anonymous,
                notes=self.config.notes,
                # entity=self.config.entity,
                settings=wandb.Settings(
                    start_method="thread",
                    _disable_stats=True,
                ),
                mode="online" if self.config.online else "offline",
            )
        else:
            self.run = None

    def log(self, *args, **kwargs):
        if self.enable:
            self.run.log(*args, **kwargs)

    def save_pickle(self, obj, filename):
        if self.enable:
            with open(os.path.join(self.config.output_dir, filename), "wb") as fout:
                pickle.dump(obj, fout)

            if self.config.gcs_output_dir != "":
                path = os.path.join(self.config.gcs_output_dir, filename)
                with gcsfs.GCSFileSystem().open(path, "wb") as fout:
                    pickle.dump(obj, fout)

    def wandb_save_model(self, filename):
        wandb.save(os.path.join(self.config.output_dir, filename))

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir


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
        elif (isinstance(val, list) or isinstance(val, tuple)) and isinstance(val[0], int):
            absl.flags.DEFINE_multi_integer(key, val, "automatically defined flag")
        elif (isinstance(val, list) or isinstance(val, tuple)) and isinstance(val[0], str):
            absl.flags.DEFINE_multi_string(key, val, "automatically defined flag")
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


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    return jnp.expand_dims(
        get_1d_sincos_pos_embed_from_grid(embed_dim, jnp.arange(length, dtype=jnp.float32)),
        0,
    )


def get_2d_sincos_pos_embed(embed_dim, length):
    grid_size = int(length**0.5)
    assert grid_size * grid_size == length

    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return jnp.expand_dims(pos_embed, 0)


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


EPS = np.finfo(np.float32).eps


class StdNormalizer:
    """Maps an un-normalized vector to zmuv."""

    def __init__(self, mean, std):
        self._mean = np.asarray(mean).astype(np.float32)
        self._std = np.asarray(std).astype(np.float32)

    def __call__(self, vector, **kwargs):
        vector = vector.astype(np.float32)
        return (vector - self._mean) / np.maximum(self._std, EPS)


class StdDenormalizer:
    """Maps a zmuv-normalized vector back to its original mean and std."""

    def __init__(self, mean, std):
        self._mean = np.asarray(mean).astype(np.float32)
        self._std = np.asarray(std).astype(np.float32)

    def __call__(self, vector, mean_offset=True, **kwargs):
        vector = vector.astype(np.float32)
        result = vector * np.maximum(self._std, EPS)
        if mean_offset:
            result += self._mean
        return result


def flattened_traversal(fn):
    """Returns function that is called with `(path, param)` instead of pytree."""

    def mask(tree):
        flat = flax.traverse_util.flatten_dict(tree)
        return flax.traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})

    return mask


def symlog(x):
    return jnp.sign(x) * jnp.log(1 + jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def compute_scale(return_to_go):
    if return_to_go >= 0:
        max_digit = int(str(int(return_to_go))[0])
    else:
        max_digit = int(str(int(return_to_go))[1])
    # n = len(str(int(return_to_go)))
    if return_to_go >= 0:
        n = len(str(int(return_to_go))) - 1 if max_digit < 5 else len(str(int(return_to_go)))
    else:
        n = len(str(int(return_to_go))) - 2 if max_digit < 5 else len(str(int(return_to_go))) - 1
    return pow(10, n)
