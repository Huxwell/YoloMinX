#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
"""
Single-GPU stubs for what used to be multi-GPU communication primitives.

This project assumes one GPU on one machine, so every "distributed" value
collapses to the trivial single-process answer (world_size == 1, rank == 0).
The function names are kept so existing call sites work unchanged.
"""

import time
from contextlib import contextmanager

import torch

__all__ = [
    "wait_for_the_master",
    "is_main_process",
    "get_world_size",
    "get_rank",
    "get_local_rank",
    "time_synchronized",
]


@contextmanager
def wait_for_the_master(local_rank: int = None):
    # Single process: there is no master to wait for.
    yield


def get_world_size() -> int:
    return 1


def get_rank() -> int:
    return 0


def get_local_rank() -> int:
    return 0


def is_main_process() -> bool:
    return True


def time_synchronized():
    """pytorch-accurate time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
