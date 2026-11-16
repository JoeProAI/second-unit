"""Hands: MediaPipe Hand Landmarker (Tasks API)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from bench.adapter import VIDEO, Adapter

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)


class MediaPipeHands(Adapter):
    name = "mediapipe_hands"
    stage = "hands"
    kind = VIDEO
    label = "MediaPipe Hands"
    settings = ("landmarker_f16",)
    requires = ("mediapipe",)
    extra = "hands"
    uses_torch = False
    supports_batch = False
    notes = (
        "Hand Landmarker task in VIDEO mode, up to 4 hands, CPU delegate (the Python GPU "
        "delegate is not available on Windows); runs on the CPU even on a cuda run"
    )

    def model_path(self, ctx) -> Path:
        return Path(ctx.weights_dir) / "mediapipe" / "hand_landmarker.task"

    def load(self, setting, ctx):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        path = self.model_path(ctx)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, path)
        self.mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=4,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.ts_ms = 0

    def process(self, batch):
        frame = self.take(1)[0]
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame)
        self.ts_ms += 33
        self.landmarker.detect_for_video(image, self.ts_ms)
        return 1.0

    def unload(self):
        if getattr(self, "landmarker", None) is not None:
            self.landmarker.close()
            self.landmarker = None
