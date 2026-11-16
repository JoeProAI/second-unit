"""WAV I/O and audio metrics in numpy only, no scipy, no network.

Reads/writes 16-bit PCM WAVs (the format Suno stem downloads convert to
cleanly). Loudness is a mono-gated RMS estimate reported as approximate
integrated LUFS; it is intentionally labelled "approx" everywhere because
true BS.1770 K-weighting needs the full filter chain, which we deliberately
don't fake.
"""
from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# RMS->LUFS offset: a full-scale sine reads -3.01 dBFS RMS and about -3.01
# LUFS, so identity mapping is a defensible approximation for mixes.
LUFS_OFFSET_DB = 0.0


@dataclass
class Stem:
    name: str
    path: Path
    samples: np.ndarray  # float32 mono or (n, ch) -> collapsed to mono
    sample_rate: int
    seconds: float
    clipped: bool = False
    silent: bool = False
    peak_dbfs: float = -np.inf
    approx_lufs: float = -np.inf
    issues: list[str] = field(default_factory=list)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Return (float32 mono samples in [-1, 1], sample_rate)."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise ValueError(f"{path.name}: expected 16-bit PCM, got {sampwidth * 8}-bit")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, rate


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write float samples in [-1, 1] to a 16-bit PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def dbfs(peak: float) -> float:
    return float(20 * np.log10(max(peak, 1e-9)))


def estimate_lufs(samples: np.ndarray) -> float:
    """Approximate integrated loudness with a simple absolute-mean-square
    gate (BS.1770-style gating without K-weighting)."""
    if samples.size == 0:
        return -np.inf
    ms = float(np.mean(samples.astype(np.float64) ** 2))
    if ms <= 0:
        return -np.inf
    gated = samples[np.abs(samples) ** 2 > ms / 10]  # -10 dB relative gate
    ms = float(np.mean(gated.astype(np.float64) ** 2)) if gated.size else ms
    return float(10 * np.log10(max(ms, 1e-12)) - 0.691 + LUFS_OFFSET_DB)


def normalize_peak(samples: np.ndarray, target_dbfs: float) -> np.ndarray:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 0:
        return samples
    gain = 10 ** (target_dbfs / 20) / peak
    return (samples * gain).astype(np.float32)


def validate_stem(stem: Stem, cfg) -> None:
    """Fill in quality flags + issues for one stem."""
    if stem.sample_rate != cfg.sample_rate:
        stem.issues.append(
            f"sample rate {stem.sample_rate} != {cfg.sample_rate}"
        )
    abs_s = np.abs(stem.samples)
    stem.peak_dbfs = dbfs(float(abs_s.max()) if abs_s.size else 0.0)
    stem.approx_lufs = estimate_lufs(stem.samples)
    p95 = float(np.percentile(abs_s, 95)) if abs_s.size else 0.0
    stem.silent = p95 < cfg.silence_floor
    if stem.silent:
        stem.issues.append("stem is silent")
    full_scale = np.mean(abs_s > 0.999) if abs_s.size else 0.0
    stem.clipped = bool(full_scale > cfg.clip_fraction)
    if stem.clipped:
        stem.issues.append(f"clipping: {full_scale:.4%} of samples at full scale")


def load_stems(folder: Path, cfg) -> dict[str, Stem]:
    """Load every stem-named WAV in a song inbox folder. Stem names come from
    the WAV filenames themselves (minus -stem suffix if present), so any stem
    set Suno exports just works."""
    stems: dict[str, Stem] = {}
    for wav in sorted(folder.glob("*.wav")):
        name = wav.stem
        if name in {"mix", "full-mix", "full_mix"}:
            continue  # optional full mix is not a stem
        if name.endswith("-stem"):
            name = name[: -len("-stem")]
        samples, rate = read_wav(wav)
        stem = Stem(
            name=name,
            path=wav,
            samples=samples,
            sample_rate=rate,
            seconds=len(samples) / rate,
        )
        validate_stem(stem, cfg)
        stems[name] = stem
    return stems


def spectral_flux_onsets(
    samples: np.ndarray, rate: int, hop: int = 512, frame: int = 1024
) -> list[float]:
    """Onset times (seconds) via spectral flux: half-rectified increase of
    magnitude spectrum between frames. Good enough for drums."""
    if len(samples) < frame * 4:
        return []
    n_frames = 1 + (len(samples) - frame) // hop
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(n_frames, frame),
        strides=(samples.strides[0] * hop, samples.strides[0]),
    )
    window = np.hanning(frame).astype(np.float32)
    mags = np.abs(np.fft.rfft(frames * window, axis=1))
    flux = np.maximum(np.diff(mags, axis=0), 0).sum(axis=1)
    if flux.size == 0 or flux.max() <= 0:
        return []
    thresh = flux.mean() + 1.5 * flux.std()
    peaks = np.where(flux > thresh)[0]
    if peaks.size == 0:
        return []
    # merge peaks closer than 80 ms
    times = (peaks + 1) * hop / rate
    merged = [times[0]]
    for t in times[1:]:
        if t - merged[-1] > 0.08:
            merged.append(float(t))
    return merged
