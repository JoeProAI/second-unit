"""Audio: demucs stems, beat and downbeat tracking, faster-whisper ASR.

Throughput for audio adapters is a realtime factor: seconds of audio processed
per wall clock second. 20 means twenty times faster than realtime.
"""

from __future__ import annotations

from pathlib import Path

from bench.adapter import AUDIO, Adapter, add_torch_cuda_dlls


class Demucs(Adapter):
    name = "demucs"
    stage = "stems"
    kind = AUDIO
    label = "Demucs"
    settings = ("htdemucs",)
    requires = ("torch", "demucs")
    extra = "audio"
    clip_seconds = 30.0
    max_batch = 8
    notes = "apply_model with split=True, overlap 0.25, shifts 0; batch = 30 s stereo 44.1 kHz clips per call"

    def load(self, setting, ctx):
        import torch
        from demucs.pretrained import get_model

        self.torch = torch
        self.model = get_model(setting).to(ctx.device).eval()

    def prepare(self, key, media):
        super().prepare(key, media)
        self.wave = self.torch.from_numpy(media.audio(self.model.samplerate, 2, self.clip_seconds).copy())

    def process(self, batch):
        from demucs.apply import apply_model

        mix = self.wave.unsqueeze(0).repeat(batch, 1, 1)
        with self.torch.inference_mode():
            apply_model(self.model, mix, shifts=0, split=True, overlap=0.25, progress=False, device=self.ctx.device)
        return batch * self.clip_seconds

    def unload(self):
        self.model = None


class BeatThis(Adapter):
    name = "beat_this"
    stage = "beats"
    kind = AUDIO
    label = "Beat This!"
    settings = ("final0",)
    requires = ("torch", "beat_this")
    extra = "audio"
    supports_batch = False
    clip_seconds = 120.0
    notes = "Audio2Beats with minimal postprocessing (no DBN), fp32, 120 s mono 22.05 kHz input"

    def load(self, setting, ctx):
        from beat_this.inference import Audio2Beats

        local = Path(ctx.weights_dir) / "beat_this" / f"{setting}.ckpt"
        checkpoint = str(local) if local.exists() else setting
        self.model = Audio2Beats(checkpoint_path=checkpoint, device=ctx.device, dbn=False)
        self.sr = 22050

    def prepare(self, key, media):
        super().prepare(key, media)
        self.signal = media.audio(self.sr, 1, self.clip_seconds)[0].copy()

    def process(self, batch):
        self.model(self.signal, self.sr)
        return self.clip_seconds

    def unload(self):
        self.model = None


class Madmom(Adapter):
    name = "madmom"
    stage = "beats"
    kind = AUDIO
    label = "madmom downbeats"
    settings = ("rnn_dbn",)
    requires = ("madmom",)
    extra = "madmom"
    uses_torch = False
    supports_batch = False
    clip_seconds = 120.0
    notes = "RNNDownBeatProcessor plus DBNDownBeatTrackingProcessor (3 and 4 beats per bar); CPU only"

    def load(self, setting, ctx):
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

        self.rnn = RNNDownBeatProcessor()
        self.dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
        self.sr = 44100

    def prepare(self, key, media):
        from madmom.audio.signal import Signal

        super().prepare(key, media)
        mono = media.audio(self.sr, 1, self.clip_seconds)[0]
        self.signal = Signal(mono.copy(), sample_rate=self.sr, num_channels=1)

    def process(self, batch):
        self.dbn(self.rnn(self.signal))
        return self.clip_seconds

    def unload(self):
        self.rnn = self.dbn = None


class FasterWhisper(Adapter):
    name = "faster_whisper"
    stage = "asr"
    kind = AUDIO
    label = "faster-whisper"
    settings = ("small", "large-v3")
    requires = ("faster_whisper", "ctranslate2")
    extra = "audio"
    uses_torch = False
    max_batch = 16
    clip_seconds = 300.0
    notes = (
        "float16 on cuda, int8 on cpu, beam 5, language en, no VAD; batch 1 uses "
        "WhisperModel.transcribe, larger batches use BatchedInferencePipeline over 30 s chunks; "
        "pass a real speech or vocal track with --audio, a synthetic track gives little text"
    )

    def check(self, setting, ctx):
        add_torch_cuda_dlls()
        import ctranslate2

        if ctx.device == "cuda" and ctranslate2.get_cuda_device_count() == 0:
            return "ctranslate2 sees no CUDA device (check CUDA 12 cuBLAS and cuDNN 9 DLLs on PATH)"
        return None

    def load(self, setting, ctx):
        add_torch_cuda_dlls()
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        compute = "float16" if ctx.device == "cuda" else "int8"
        self.model = WhisperModel(setting, device=ctx.device, compute_type=compute)
        self.batched = BatchedInferencePipeline(model=self.model)

    def prepare(self, key, media):
        super().prepare(key, media)
        self.audio = media.audio(16000, 1, self.clip_seconds)[0].copy()
        self.seconds = self.audio.shape[0] / 16000

    def process(self, batch):
        if batch == 1:
            segments, _ = self.model.transcribe(
                self.audio, beam_size=5, language="en", vad_filter=False, condition_on_previous_text=False
            )
        else:
            n = len(self.audio) // 16000
            clips = [{"start": float(s), "end": float(min(s + 30, n))} for s in range(0, n, 30)]
            segments, _ = self.batched.transcribe(
                self.audio, beam_size=5, language="en", batch_size=batch, vad_filter=False, clip_timestamps=clips
            )
        for _ in segments:
            pass
        return self.seconds

    def unload(self):
        self.batched = None
        self.model = None


