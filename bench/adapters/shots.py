"""Shot boundary detectors: TransNetV2, AutoShot, OmniShotCut."""

from __future__ import annotations

import sys
from pathlib import Path

from bench.adapter import VIDEO, Adapter

WINDOW = 100
STRIDE = 50


class TransNetV2(Adapter):
    name = "transnetv2"
    stage = "shots"
    kind = VIDEO
    label = "TransNetV2 (PyTorch port)"
    settings = ("default",)
    requires = ("torch", "transnetv2_pytorch")
    extra = "shots"
    max_batch = 16
    notes = (
        "frames resized on the GPU to 48x27, 100 frame windows with stride 50; "
        "batch = windows per forward pass; throughput counts 50 new frames per window"
    )

    def load(self, setting, ctx):
        import torch
        from transnetv2_pytorch import TransNetV2 as Net

        self.torch = torch
        self.model = Net(device=ctx.device).to(ctx.device).eval()

    def frames_for(self, batch):
        return STRIDE * batch + (WINDOW - STRIDE)

    def forward(self, x):
        return self.model(x)

    def process(self, batch):
        from bench.adapters._torch import frames_to_small_uint8, windows

        with self.torch.inference_mode():
            small = frames_to_small_uint8(self.take(self.frames_for(batch)), self.ctx.device)
            self.forward(windows(small, batch, WINDOW, STRIDE))
        return float(STRIDE * batch)

    def unload(self):
        self.model = None


class AutoShot(TransNetV2):
    name = "autoshot"
    label = "AutoShot (CVPR 2023 supernet)"
    requires = ("torch", "einops")
    files = ("supernet_flattransf_3_8_8_8_13_12_0_16_60.py", "linear.py", "ckpt_0_200_0.pth")
    notes = (
        "same 48x27 input and windowing as TransNetV2; needs the model source files and "
        "checkpoint placed in bench/weights/autoshot"
    )

    def folder(self, ctx) -> Path:
        return Path(ctx.weights_dir) / "autoshot"

    def check(self, setting, ctx):
        missing = [f for f in self.files if not (self.folder(ctx) / f).exists()]
        if missing:
            return f"weights or source not found in bench/weights/autoshot: {', '.join(missing)}"
        return None

    def load(self, setting, ctx):
        import pickle

        import torch

        self.torch = torch
        folder = str(self.folder(ctx))
        if folder not in sys.path:
            sys.path.insert(0, folder)
        from supernet_flattransf_3_8_8_8_13_12_0_16_60 import TransNetV2Supernet

        model = TransNetV2Supernet().eval()
        ckpt_path = self.folder(ctx) / "ckpt_0_200_0.pth"
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, RuntimeError):
            # The published checkpoint stores more than tensors.
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
        own = model.state_dict()
        matched = {k: v for k, v in state.items() if k in own}
        if not matched:
            raise RuntimeError("no checkpoint keys matched the AutoShot supernet")
        own.update(matched)
        model.load_state_dict(own)
        self.model = model.to(ctx.device)

    def forward(self, x):
        # AutoShot takes float [B, 3, T, H, W] in 0..255.
        return self.model(x.permute(0, 4, 1, 2, 3).float())


class OmniShotCut(Adapter):
    name = "omnishotcut"
    stage = "shots"
    kind = VIDEO
    label = "OmniShotCut v1.5"
    settings = ("v1.5",)
    requires = ("torch", "omnishotcut")
    extra = "shots"
    devices = ("cuda",)
    supports_batch = False
    chunk = 200
    notes = (
        "calls omnishotcut inference() on 200 frame numpy chunks; the package resizes to its "
        "own process resolution on the CPU and windows internally, so no batch sweep"
    )

    def load(self, setting, ctx):
        import omnishotcut

        self.model = omnishotcut.load("uva-cv-lab/OmniShotCut_v1.5", filename="OmniShotCut_ckpt.pth")

    def prepare(self, key, media):
        import numpy as np

        super().prepare(key, media)
        self.chunk_array = np.stack(self.take(self.chunk))

    def process(self, batch):
        self.model.inference(self.chunk_array, mode="clean_shot")
        return float(self.chunk)

    def unload(self):
        self.model = None
