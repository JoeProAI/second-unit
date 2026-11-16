"""Shared helpers for PyTorch adapters. Import only from inside load()/process()."""

from __future__ import annotations

from typing import Sequence

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def to_device_bchw(frames: Sequence[np.ndarray], device: str):
    """uint8 HWC frames to a float BCHW tensor in [0, 1] on device."""
    import torch

    x = torch.from_numpy(np.stack(frames))
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
    return x.permute(0, 3, 1, 2).float().div_(255.0)


def resize(x, size: tuple[int, int]):
    import torch.nn.functional as F

    return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=False)


def resize_short_center_crop(x, side: int):
    _, _, h, w = x.shape
    scale = side / min(h, w)
    nh, nw = max(side, round(h * scale)), max(side, round(w * scale))
    x = resize(x, (nh, nw))
    top, left = (nh - side) // 2, (nw - side) // 2
    return x[:, :, top : top + side, left : left + side]


def normalize(x, mean, std):
    import torch

    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def frames_to_small_uint8(frames: Sequence[np.ndarray], device: str, size=(27, 48), chunk: int = 32):
    """Resize many full size frames down on the device, in chunks to bound host memory.
    Returns uint8 tensor [T, h, w, 3]."""
    import torch

    parts = []
    for i in range(0, len(frames), chunk):
        x = to_device_bchw(frames[i : i + chunk], device)
        x = resize(x, size)
        parts.append((x * 255.0).round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1))
    return torch.cat(parts, 0)


def windows(t, count: int, length: int = 100, stride: int = 50):
    """Stack `count` overlapping windows from a [T, ...] tensor."""
    import torch

    return torch.stack([t[i * stride : i * stride + length] for i in range(count)], 0)
