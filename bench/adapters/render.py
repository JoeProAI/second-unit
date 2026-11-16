"""Render: ffmpeg encode test with h264_nvenc, hevc_nvenc and libx264."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from bench.adapter import RENDER, Adapter

ENCODERS = {
    "h264_nvenc": ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23"],
    "hevc_nvenc": ["-c:v", "hevc_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "25"],
    "libx264": ["-c:v", "libx264", "-preset", "medium", "-crf", "20"],
}
LOOPS = 4


class Ffmpeg(Adapter):
    name = "ffmpeg"
    stage = "render"
    kind = RENDER
    label = "ffmpeg encode"
    settings = tuple(ENCODERS)
    requires_bins = ("ffmpeg",)
    uses_torch = False
    supports_batch = False
    notes = (
        "raw rgb24 frames read from a temp file and looped 4 times, yuv420p conversion included, "
        "output to the null muxer; wall time includes ffmpeg start up"
    )

    def check(self, setting, ctx):
        ffmpeg = shutil.which("ffmpeg")
        listing = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        if f" {setting} " not in listing:
            return f"this ffmpeg build has no {setting} encoder"
        if "nvenc" in setting:
            probe = subprocess.run(
                [ffmpeg, "-hide_banner", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=256x256:d=0.1",
                 "-frames:v", "1", *ENCODERS[setting], "-f", "null", "-"],
                capture_output=True, text=True,
            )
            if probe.returncode != 0:
                last = (probe.stderr.strip().splitlines() or ["no output"])[-1]
                return f"{setting} probe failed: {last[:200]}"
        return None

    def load(self, setting, ctx):
        self.ffmpeg = shutil.which("ffmpeg")
        self.tmp = Path(tempfile.mkdtemp(prefix="second-unit-render-"))

    def prepare(self, key, media):
        super().prepare(key, media)
        self.w, self.h = media.size(key)
        self.raw = self.tmp / f"{key}.rgb"
        with open(self.raw, "wb") as f:
            for frame in self.frames:
                f.write(frame.tobytes())
        self.count = len(self.frames) * LOOPS

    def process(self, batch):
        cmd = [
            self.ffmpeg, "-hide_banner", "-v", "error", "-y", "-stream_loop", str(LOOPS - 1),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.w}x{self.h}", "-r", "30", "-i", str(self.raw),
            "-pix_fmt", "yuv420p", *ENCODERS[self.setting], "-f", "null", "-",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip()[-300:])
        return float(self.count)

    def unload(self):
        if getattr(self, "tmp", None) is not None:
            shutil.rmtree(self.tmp, ignore_errors=True)
