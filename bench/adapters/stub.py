"""CPU stand ins used by --smoke and the tests. They exercise every code path
(load, prepare, batch sweep, simulated out of memory, skip) without weights."""

from __future__ import annotations

import time
import zlib

import numpy as np

from bench.adapter import AUDIO, RENDER, VIDEO, Adapter


class StubVideo(Adapter):
    name = "stub_video"
    stage = "smoke"
    kind = VIDEO
    label = "Stub video model"
    settings = ("tiny",)
    uses_torch = False
    notes = "numpy block means; raises a simulated out of memory error above oom_at"

    def load(self, setting, ctx):
        time.sleep(0.01)
        self.oom_at = int(ctx.options.get("stub_oom_at", 4))
        self.kernel = np.ones((4, 4), dtype=np.float32) / 16.0

    def process(self, batch):
        if batch > self.oom_at:
            raise RuntimeError("CUDA out of memory (simulated by stub_video)")
        x = np.stack(self.take(batch)).astype(np.float32)
        b, h, w, c = x.shape
        h4, w4 = h // 4 * 4, w // 4 * 4
        x[:, :h4, :w4].reshape(b, h4 // 4, 4, w4 // 4, 4, c).mean(axis=(2, 4))
        return float(batch)


class StubAudio(Adapter):
    name = "stub_audio"
    stage = "smoke"
    kind = AUDIO
    label = "Stub audio model"
    settings = ("tiny",)
    uses_torch = False
    clip_seconds = 2.0
    max_batch = 4
    notes = "numpy FFT over the generated audio"

    def load(self, setting, ctx):
        self.sr = 16000

    def prepare(self, key, media):
        super().prepare(key, media)
        self.audio = media.audio(self.sr, 1, self.clip_seconds)[0]

    def process(self, batch):
        for _ in range(batch):
            np.abs(np.fft.rfft(self.audio))
        return batch * self.audio.shape[0] / self.sr


class StubRender(Adapter):
    name = "stub_render"
    stage = "smoke"
    kind = RENDER
    label = "Stub encoder"
    settings = ("zlib",)
    uses_torch = False
    supports_batch = False
    notes = "zlib compresses raw frames in place of a video encoder"

    def load(self, setting, ctx):
        pass

    def process(self, batch):
        frames = self.take(8)
        for f in frames:
            zlib.compress(f.tobytes(), 1)
        return float(len(frames))


class StubMissing(Adapter):
    name = "stub_missing"
    stage = "smoke"
    kind = VIDEO
    label = "Stub with a missing dependency"
    settings = ("default",)
    requires = ("second_unit_dependency_that_does_not_exist",)
    extra = "nothing"
    uses_torch = False

    def load(self, setting, ctx):
        raise AssertionError("never called: missing() reports the skip first")
