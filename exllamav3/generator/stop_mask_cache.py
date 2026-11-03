"""Opt-in, job-owned reuse of an invariant minimum-length stop-token mask.

This caches sampling metadata, not sampled tokens or decisions. Dynamic masks
and unknown samplers keep the ordinary preparation path.
"""
import os
import torch
from .sampler import custom

enabled = os.environ.get("EXL3_STOP_MASK_CACHE", "0") == "1"
_steps = frozenset(t for t in vars(custom).values()
                   if isinstance(t, type) and issubclass(t, custom.SS_Base))


def reuse(job):
    """Return True only when the existing device mask is ready for this stream."""
    job._stop_mask_pending = None
    # Filters/checkpoints/healing can change a mask after every accepted token.
    # Exclude even dormant filters and banned strings rather than relying on
    # their activation/rewind timing. A requeued Job is initialized afresh.
    if (job.filters or job.checkpoint is not None or job.banned_strings
            or job.new_tokens < 0 or job.new_tokens >= job.min_new_tokens
            or not job.stop_tokens_list or job.logit_masks or job.filter_futures
            or getattr(getattr(job.sampler, "forward", None), "__func__", None) is not custom.CustomSampler.forward
            or any(type(step) not in _steps for step in job.sampler.steps)):
        job._stop_mask_cache = None
        return False
    if job.logits_device is None:
        job._stop_mask_cache = None
        return False
    device = torch.device(job.logits_device)
    if device.type != "cuda":
        job._stop_mask_cache = None
        return False
    # Stream identity prevents reuse of an upload enqueued on another stream.
    # The source mask remains private to this job, just like the original path.
    stream = torch.cuda.current_stream(device)
    key = (tuple(job.stop_tokens_list), job.generator.padded_vocab_size,
           stream.device, stream.cuda_stream)
    cached = getattr(job, "_stop_mask_cache", None)
    if cached is not None and cached[0] == key:
        job.device_logit_mask = cached[1]
        return True
    job._stop_mask_cache = None
    job._stop_mask_pending = key
    return False


def remember(job):
    """Keep the mask produced by the unchanged preparation code on a miss."""
    key = job._stop_mask_pending
    if key is not None and job.device_logit_mask is not None:
        job._stop_mask_cache = (key, job.device_logit_mask)
    job._stop_mask_pending = None
