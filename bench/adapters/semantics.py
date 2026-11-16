"""Semantics and eye line: DINOv2 and open_clip ViT-L/14 image towers."""

from __future__ import annotations

from pathlib import Path

from bench.adapter import VIDEO, Adapter


class DinoV2(Adapter):
    name = "dinov2"
    stage = "semantics"
    kind = VIDEO
    label = "DINOv2"
    settings = ("vitb14", "vitl14")
    requires = ("torch",)
    notes = (
        "torch.hub facebookresearch/dinov2; short side resized to 224 and center cropped on "
        "the GPU, ImageNet normalization, fp32, forward_features (CLS plus patch tokens)"
    )

    def load(self, setting, ctx):
        import torch

        self.torch = torch
        local = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        if local.is_dir():
            self.model = torch.hub.load(str(local), f"dinov2_{setting}", source="local")
        else:
            self.model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{setting}", trust_repo=True, skip_validation=True)
        self.model = self.model.to(ctx.device).eval()

    def process(self, batch):
        from bench.adapters._torch import IMAGENET_MEAN, IMAGENET_STD, normalize, resize_short_center_crop, to_device_bchw

        with self.torch.inference_mode():
            x = resize_short_center_crop(to_device_bchw(self.take(batch), self.ctx.device), 224)
            self.model.forward_features(normalize(x, IMAGENET_MEAN, IMAGENET_STD))
        return float(batch)

    def unload(self):
        self.model = None


class OpenClip(Adapter):
    name = "open_clip"
    stage = "semantics"
    kind = VIDEO
    label = "open_clip ViT-L/14"
    settings = ("ViT-L-14/laion2b_s32b_b82k",)
    requires = ("torch", "open_clip")
    extra = "semantics"
    notes = "image tower only; short side 224 and center crop on the GPU, CLIP normalization, fp32"

    def load(self, setting, ctx):
        import open_clip
        import torch

        self.torch = torch
        arch, pretrained = setting.split("/")
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=ctx.device)
        self.model = model.eval()

    def process(self, batch):
        from bench.adapters._torch import CLIP_MEAN, CLIP_STD, normalize, resize_short_center_crop, to_device_bchw

        with self.torch.inference_mode():
            x = resize_short_center_crop(to_device_bchw(self.take(batch), self.ctx.device), 224)
            self.model.encode_image(normalize(x, CLIP_MEAN, CLIP_STD))
        return float(batch)

    def unload(self):
        self.model = None
