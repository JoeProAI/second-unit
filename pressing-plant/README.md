# Pressing Plant

Turns a song's stems into a remix-ready stem pack and a music video render plan. Render uses ffmpeg with NVENC on an NVIDIA RTX card, with a libx264 CPU fallback.

Stems arrive as files dropped into `inbox/<song-slug>/`, for example a stem export from a music generator. There is no network access, no API key and no scraping. Every step is testable with synthetic stem fixtures.

## Layout of an inbox drop

```
inbox/<slug>/
  vocals.wav        # any stem set; names come from the filenames
  drums.wav
  bass.wav
  instrumental.wav
  other.wav
  mix.wav           # optional full mix
  song.yaml         # title, bpm, key, lyrics (all optional except title)
```

## Run it (clean clone, no keys, no GPU)

```bash
cd pressing-plant
pip install -e .            # or: pip install numpy pyyaml pytest

press fixture demo-song     # drop a synthetic stem set into the inbox
press all demo-song --dry-run
```

Outputs land in `out/` (or `$PRESS_OUTPUT_DIR`):

- `<slug>-stems.zip`: normalized WAVs, `manifest.json`, a `LICENSE.txt` placeholder, `LOUDNESS.md`, a cover slot
- `<slug>-video-plan.json`: onsets, sections, shot list and the full ffmpeg command (NVENC when the local ffmpeg has it, libx264 otherwise)
- `<slug>-release-copy.txt`: post text for the release
- `gallery/index.html`: a local gallery of pressings

Drop `--dry-run` from `press all` or `press video` to run the ffmpeg render.

## Test

```bash
cd pressing-plant
python3 -m pytest -q
```

## What is mocked

- Stems: generated sine and noise WAVs stand in for real exports.
- Visuals: `MockVisualGenerator` records prompts and writes PPM placeholders. A real generator implements the `VisualGenerator` protocol and plugs in unchanged.
- Loudness: an RMS-gated approximation of LUFS. `LOUDNESS.md` says so.
- This code has not yet rendered on real hardware.
