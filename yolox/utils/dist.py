#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
"""
Single-GPU stubs for what used to be multi-GPU communication primitives.

This project assumes one GPU on one machine, so every distributed operation
collapses to trivial single-process behaviour (world_size == 1, rank == 0).
The function names are kept so existing call sites work unchanged.
"""

import os
import time
from contextlib import contextmanager

import numpy as np

import torch

__all__ = [
    "get_num_devices",
    "wait_for_the_master",
    "is_main_process",
    "synchronize",
    "get_world_size",
    "get_rank",
    "get_local_rank",
    "get_local_size",
    "time_synchronized",
    "gather",
    "all_gather",
    "shared_random_seed",
]


def get_num_devices():
    gpu_list = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if gpu_list is not None:
        return len(gpu_list.split(","))
    devices_info = os.popen("nvidia-smi -L").read().strip().split("\n")
    return len(devices_info)


@contextmanager
def wait_for_the_master(local_rank: int = None):
    # Single process: there is no master to wait for.
    yield


def synchronize():
    # No other processes to barrier against.
    pass


def get_world_size() -> int:
    return 1


def get_rank() -> int:
    return 0


def get_local_rank() -> int:
    return 0


def get_local_size() -> int:
    return 1


def is_main_process() -> bool:
    return True


def all_gather(data, group=None):
    return [data]


def gather(data, dst=0, group=None):
    return [data]


def shared_random_seed():
    return int(np.random.randint(2 ** 31))


def time_synchronized():
    """pytorch-accurate time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def _get_global_gloo_group():
    # Only referenced by allreduce_norm on >1 GPU; never reached on one GPU.
    return None
