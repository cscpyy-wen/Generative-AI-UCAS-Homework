"""The seven recorded protocols; all seeds and schedules are explicit."""
import copy
import random

import numpy as np
import torch

EXPERIMENTS = ("s0", "s1", "s2", "s3", "g", "f0", "f02")
BASE_CONFIG = dict(seed=20260920, sample_seed=20260921, split_seed=20260920,
                   channels=64, n_blocks=12, batch_size=128, epochs=100,
                   lr=1e-3, min_lr=1e-5, grad_clip=1., threshold=128,
                   n_train=55000, n_val=5000, n_test=10000, n_samples=64,
                   architecture="standard")


def experiment_config(name):
    if name not in EXPERIMENTS:
        raise ValueError("Unknown experiment: " + name)
    config = copy.deepcopy(BASE_CONFIG)
    if name == "s0":
        config.update(epochs=25, batch_size=256, min_lr=1e-4)
    elif name == "s1":
        config.update(batch_size=256)
    elif name == "s3":
        config.update(lr=5e-4)
    elif name in ("g", "f0", "f02"):
        config.update(architecture="dilated_gated", dilations=[1, 2, 4]*4)
    if name in ("f0", "f02"):
        config.update(epochs=20, lr=1e-4, ema_decay=.999,
                      early_stop_patience=5, early_stop_min_delta=.02,
                      plateau_patience=2, initial_epoch=9,
                      dropout_p=0. if name == "f0" else .2,
                      training_variant="finetune_control" if name == "f0" else "finetune_dropout")
    return config


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(value="auto"):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu.")
    return device
