# Second Unit

Second Unit is a local editor for AI-generated video takes. It is being built to run on one 16 GB NVIDIA card (developed against an RTX 5080).

What it will do:

1. Score every frame of every take for generation defects: flicker, morphing, identity drift and broken hands.
2. Solve the whole cut at once. Cuts land on the beat, just before a flagged frame, or on a motion or eye-line match. Flagged frames are never shown.
3. Write a reason for every cut, for example: `cut at 3.41 s: downbeat, hides hand morph at 3.52 s, motion match 0.82`.
4. Render with ffmpeg NVENC and export OTIO/EDL for hand-off to other editors.
5. Ship CutBench, a test set and scoring script that compares an automatic cut with a human cut of the same takes.

## Status

Early. Nothing here is ready to use yet. Work lands as reviewed pull requests:

- `bench/`: VRAM and throughput harness for the candidate models. No numbers are measured yet.
- `pressing-plant/`: beat grid, stem tools and the NVENC render planner, imported from an earlier prototype.

## License

Apache-2.0. See `LICENSE`.
