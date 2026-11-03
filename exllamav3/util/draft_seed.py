"""Opt-in stream-ordered upload for the generator-owned DSpark seed buffer.

The caller must keep a pinned source unchanged until draft sampling completes.
This is deliberately not a global change to device-copy synchronization.
"""
import os
import torch
from .device_copy import to_device

enabled = os.environ.get("EXL3_ASYNC_DRAFT_SEED", "0") == "1"


def upload(seed, device):
    non_blocking = (enabled and seed.device.type == "cpu"
                    and seed.dtype == torch.long and seed.is_pinned())
    return to_device(seed, device, non_blocking=non_blocking)
