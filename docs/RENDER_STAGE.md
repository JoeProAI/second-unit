# Render stage: how Pressing Plant fits

Pressing Plant (`pressing-plant/`, package `press`) already does three things Second Unit needs: find the beat in a song, lay out shots against it, and build an ffmpeg command that encodes with NVENC. This page maps each piece onto the Second Unit pipeline. Nothing here has rendered on real hardware yet.

## Pipeline position

```
takes + song stems
  -> defect scoring (per frame, not in this repo yet)
  -> cut solver        <- beat times, downbeat times (from Pressing Plant)
  -> cuts.json
  -> renderer          <- ffmpeg NVENC plan builder (from Pressing Plant)
  -> video file, OTIO/EDL
```

## Piece by piece

| Pressing Plant | Where it lives | Second Unit role |
| --- | --- | --- |
| Onset detection | `press/audio.py`, `spectral_flux_onsets` | Candidate cut points. Spectral flux on the drum stem, peaks merged within 80 ms. Feeds the solver as a soft "cut on a hit" signal. |
| Beat grid | `press/beatgrid.py`, `beat_times`, `report_beat_times` | Solver input. A plain JSON list of beat times in seconds, built from the BPM in `song.yaml`. Written with `write_beat_grid`. |
| Shot list | `press/video.py`, `plan_video` | Reference only. Today it snaps every other beat to the nearest onset and labels sections by RMS energy. The cut solver replaces this logic; the `Shot` shape (start, end, section, note) is close to what a cut in `cuts.json` needs. |
| ffmpeg NVENC plan | `press/video.py`, `plan_video` and `detect_encoder` | Becomes the renderer for `cuts.json`. `detect_encoder` picks `h264_nvenc` or `hevc_nvenc` when the local ffmpeg lists it and falls back to `libx264` or `libx265`. NVENC flags are `-preset p5 -rc vbr -cq 24`. |

## Beat and downbeat times

- Beats: `report_beat_times(report)` returns one time every `60 / bpm` seconds from 0 up to the longest stem's length, rounded to milliseconds. Example at 120 BPM over 2 s: `[0.0, 0.5, 1.0, 1.5]`.
- Downbeats: not produced yet. The solver can take every fourth beat under a 4/4 assumption until there is a real downbeat tracker.
- First beat offset: assumed to be 0 s. `beat_times` takes an `offset_s` argument for when a real offset is known.

## What changes for `cuts.json`

The current plan builder renders audio scopes over a solid backdrop from the song stems. To render a cut it needs to take a list of `(take file, in point, out point)` segments in order and concatenate them over the song audio. The encoder choice, NVENC flags and CPU fallback carry over unchanged. The `cuts.json` schema is not defined yet.

## What is mocked

- Stems: `press fixture` writes sine and noise WAVs. Drum hits sit exactly on the beat, so onset tests are easy to pass.
- BPM: read from `song.yaml`, not detected from audio.
- Visuals: `MockVisualGenerator` writes 64x64 PPM placeholders and records prompts.
- Loudness: RMS gated estimate, not BS.1770.
- Render: CI only builds the plan (`--dry-run`). No GPU runs in CI, and no NVENC encode has been checked on real hardware.
