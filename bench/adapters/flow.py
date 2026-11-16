"""Optical flow: torchvision RAFT large and small."""

from __future__ import annotations

from bench.adapter import VIDEO, Adapter


class Raft(Adapter):
    name = "raft"
    stage = "flow"
    kind = VIDEO
    label = "RAFT (torchvision)"
    settings = ("raft_large", "raft_small")
    requires = ("torch", "torchvision")
    extra = ""
    notes = "full resolution frame pairs, fp32, torchvision default flow updates (12); batch = pairs per call"

    def load(self, setting, ctx):
        import torch
        from torchvision.models import optical_flow as of

        self.torch = torch
        if setting == "raft_large":
            weights = of.Raft_Large_Weights.DEFAULT
            self.model = of.raft_large(weights=weights, progress=False)
        else:
            weights = of.Raft_Small_Weights.DEFAULT
            self.model = of.raft_small(weights=weights, progress=False)
        self.model = self.model.to(ctx.device).eval()

    def process(self, batch):
        from bench.adapters._torch import to_device_bchw

        frames = self.take(batch + 1)
        with self.torch.inference_mode():
            x = to_device_bchw(frames, self.ctx.device) * 2.0 - 1.0
            self.model(x[:-1].contiguous(), x[1:].contiguous())
        return float(batch)

    def unload(self):
        self.model = None
