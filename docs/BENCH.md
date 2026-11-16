# Bench: VRAM and throughput per candidate model

Second Unit runs one model stage at a time on one RTX 5080 (16 GB) under Windows. This page holds the measured cost of each candidate model and the steps to measure it.

No number on this page is an estimate. Every measurement cell reads `not yet measured` until a real run on the RTX 5080 replaces it with `python -m bench report --write docs/BENCH.md`.

## Results

<!-- bench:results:start -->

### Environment

| Field | Value |
| --- | --- |
| GPU | not yet measured |
| Driver | not yet measured |
| GPU memory | not yet measured |
| PyTorch | not yet measured |
| PyTorch CUDA build | not yet measured |
| cuDNN | not yet measured |
| Python | not yet measured |
| OS | not yet measured |
| ffmpeg | not yet measured |
| Harness commit | not yet measured |
| Runs merged | not yet measured |

### Video models

| Stage | Model | Setting | Status | Load time | Peak VRAM torch, batch 1, 1080p | Peak VRAM nvidia-smi, batch 1, 1080p | Best throughput 720p | Best throughput 1080p | Largest batch 1080p | Peak VRAM nvidia-smi, largest batch 1080p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shots | TransNetV2 (PyTorch port) | default | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| shots | AutoShot (CVPR 2023 supernet) | default | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| shots | OmniShotCut v1.5 | v1.5 | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| flow | RAFT (torchvision) | raft_large | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| flow | RAFT (torchvision) | raft_small | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| face | ArcFace via insightface | buffalo_l | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| semantics | DINOv2 | vitb14 | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| semantics | DINOv2 | vitl14 | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| semantics | open_clip ViT-L/14 | ViT-L-14/laion2b_s32b_b82k | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| vlm | Qwen3-VL Instruct GGUF (llama.cpp) | 2b | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| vlm | Qwen3-VL Instruct GGUF (llama.cpp) | 4b | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| vlm | Qwen3-VL Instruct GGUF (llama.cpp) | 8b | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| hands | MediaPipe Hands | landmarker_f16 | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |

### Audio models

| Stage | Model | Setting | Status | Load time | Peak VRAM torch, batch 1 | Peak VRAM nvidia-smi, batch 1 | Realtime factor, batch 1 | Best realtime factor | Largest batch | Peak VRAM nvidia-smi, largest batch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stems | Demucs | htdemucs | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| beats | Beat This! | final0 | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| beats | madmom downbeats | rnn_dbn | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| asr | faster-whisper | small | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| asr | faster-whisper | large-v3 | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |

### Render

| Stage | Encoder | Setting | Status | Throughput 720p | Throughput 1080p | Peak VRAM nvidia-smi |
| --- | --- | --- | --- | --- | --- | --- |
| render | ffmpeg encode | h264_nvenc | not yet measured | not yet measured | not yet measured | not yet measured |
| render | ffmpeg encode | hevc_nvenc | not yet measured | not yet measured | not yet measured | not yet measured |
| render | ffmpeg encode | libx264 | not yet measured | not yet measured | not yet measured | not yet measured |

<!-- bench:results:end -->

## What each column means

* **Load time**: wall seconds for the adapter `load()` call, model weights from the local cache to the device, synchronized. Run `python -m bench fetch` first so downloads are not counted.
* **Peak VRAM torch**: `torch.cuda.max_memory_allocated()` for the timed batch, reset before each batch size. Only torch models fill this column. onnxruntime (insightface), CTranslate2 (faster-whisper), llama.cpp, MediaPipe and ffmpeg show `n/a`; use the nvidia-smi column for them.
* **Peak VRAM nvidia-smi**: highest whole card `memory.used` seen by a sampler thread (NVML, falling back to `nvidia-smi`, every 50 ms) while that batch size ran. The value in brackets subtracts the idle reading taken just before the stage started, so the Windows desktop and other apps are mostly removed. Close other GPU apps before a run.
* **Throughput**: frames per second for video models and ffmpeg, measured after untimed warm up calls. For audio models it is the realtime factor: seconds of audio processed per wall second, so `20x` means twenty times faster than realtime.
* **Largest batch**: the batch sweep tries 1, 2, 4, 8 and so on up to the cap (default 32). The cell shows the largest batch that ran and why the sweep stopped: `out of memory at N`, `reached cap`, `limit at N` (the model itself refused), `no batching` (the adapter processes one item per call), or `error at N`.
* **Status**: `ok`, `skipped: <reason>` (a dependency, weight file, encoder or CUDA device is missing, nothing ran), or `error: <reason>` (it ran and failed, details in the JSON and the stage log).

Each stage setting runs in its own child process, so one model's memory cannot leak into the next reading and a crash only loses that row.

### What a unit of work is

| Stage | Model | Unit counted by throughput |
| --- | --- | --- |
| shots | TransNetV2, AutoShot | Frames: 100 frame windows at 48x27, stride 50, so a batch of N windows counts 50 N new frames. Decode and resize to 48x27 happen outside the timer. |
| shots | OmniShotCut | Frames: 200 frame chunks through the upstream inference path. One chunk per call. |
| flow | RAFT large and small | Frame pairs at full input resolution, 12 refinement iterations (torchvision default). |
| face | insightface buffalo_l | Frames: detector at 640x640 plus one ArcFace embedding per frame. |
| semantics | DINOv2, open_clip | Frames: short side resized to 224, center crop 224x224, image tower only. |
| vlm | Qwen3-VL GGUF | Frames: N JPEG frames in one chat request with the defect checklist prompt, up to 128 output tokens, temperature 0. |
| hands | MediaPipe Hands | Frames: Hand Landmarker in video mode, up to 4 hands. MediaPipe runs on the CPU delegate on Windows. |
| stems | demucs htdemucs | 30 second stereo clips, split mode, overlap 0.25, no shift augmentation. |
| beats | beat_this, madmom | 120 seconds of mono audio, one clip per call. |
| asr | faster-whisper | 300 seconds of 16 kHz mono audio. Batch 1 is the plain `transcribe` call, larger batches use `BatchedInferencePipeline`. |
| render | ffmpeg | Frames encoded from raw RGB at the input resolution to the null muxer, yuv420p conversion included. |

The input is a generated test clip (moving gradients and a square, with a hard cut halfway) and a generated click and bass track. Pass `--clip` and `--audio` to use real files instead. faster-whisper speed depends on how much speech it finds, so for the asr rows pass a track with vocals; the generated track has none.

## Run it on Windows with an RTX 5080

These steps start from a clean clone. Run them in PowerShell. Nothing here needs an account, an API key or a paid service. Model weights are downloaded from their public hosts.

### 1. System tools

```powershell
winget install -e --id Python.Python.3.11
winget install -e --id Git.Git
winget install -e --id Gyan.FFmpeg
```

Open a new PowerShell window so the new PATH is picked up, then check:

```powershell
py -3.11 --version
ffmpeg -hide_banner -encoders | Select-String nvenc
nvidia-smi
```

The top right of the `nvidia-smi` output shows the newest CUDA version the driver supports. PyTorch needs 12.8 or newer. The llama.cpp build in the Qwen3-VL section needs 13.5 or newer; update the driver if it shows less. Use Python 3.11 for the virtual environment.

### 2. Clone and create the environment

```powershell
git clone https://github.com/JoeProAI/second-unit.git
cd second-unit
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell refuses to run the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

### 3. CUDA build of PyTorch

The RTX 5080 (Blackwell, sm_120) needs a CUDA 12.8 or newer build. Install these exact versions:

```powershell
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

The second line must print a `+cu128` version and the GPU name. If it prints an error or `False`, stop and fix this first: every torch row will show `skipped: no CUDA device`.

### 4. Harness and per model extras

```powershell
pip install -e ".[test,gpu]"
```

Then add the extras for the stages you want. Each one is optional. A stage without its extra shows `skipped: missing dependency ...` and the rest of the run carries on.

| Stage | Install | Notes |
| --- | --- | --- |
| shots | `pip install -e ".[shots]"` | TransNetV2 and OmniShotCut. AutoShot needs files by hand, see below. OmniShotCut is installed from its GitHub repo, so Git must be on PATH. |
| flow | nothing extra | RAFT comes with torchvision. |
| face | `pip install -e ".[face]"` then `pip uninstall -y onnxruntime` then `pip install --force-reinstall --no-deps onnxruntime-gpu` | insightface pulls in the CPU `onnxruntime`, which hides the GPU one. The two reinstall commands leave only `onnxruntime-gpu`. |
| semantics | `pip install -e ".[semantics]"` | DINOv2 is fetched with `torch.hub`, open_clip from Hugging Face. |
| vlm | `pip install -e ".[vlm]"` plus llama.cpp, see below | |
| hands | `pip install -e ".[hands]"` | |
| stems, beats, asr | `pip install -e ".[audio]"` | demucs, beat_this and faster-whisper. |
| beats (madmom) | `pip install -e ".[madmom]"` | Optional second beat tracker. Builds from source, needs Visual Studio Build Tools with the C++ workload. Skip it if the build fails. |
| render | nothing extra | Uses the ffmpeg from step 1. |

Or install everything except madmom in one line, then do the onnxruntime swap from the face row:

```powershell
pip install -e ".[all]"
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps onnxruntime-gpu
```

faster-whisper and onnxruntime-gpu use the cuBLAS and cuDNN DLLs that ship inside the cu128 torch build; the harness adds that folder to the DLL search path, so no separate CUDA toolkit is needed.

#### AutoShot files

AutoShot has no pip package. Put three files in `bench\weights\autoshot\`:

1. `supernet_flattransf_3_8_8_8_13_12_0_16_60.py` and `linear.py` from the AutoShot GitHub repository (https://github.com/wentaozhu/AutoShot).
2. `ckpt_0_200_0.pth`, from the Google Drive folder linked at the top of that repository's README.

Until all three are there the row shows `skipped: weights or source not found`.

#### Qwen3-VL with llama.cpp

1. From https://github.com/ggml-org/llama.cpp/releases download the newest `llama-<build>-bin-win-cuda-13.5-x64.zip` and `cudart-llama-bin-win-cuda-13.5-x64.zip`. Unzip both into one folder, for example `C:\tools\llama`. Check it with `C:\tools\llama\llama-server.exe --version`.
2. Download one quantization of the model and its projector per size. With the `hf` command from `pip install huggingface_hub`:

```powershell
hf download Qwen/Qwen3-VL-2B-Instruct-GGUF Qwen3VL-2B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-2B-Instruct-F16.gguf --local-dir bench\weights\qwen3-vl\2b
hf download Qwen/Qwen3-VL-4B-Instruct-GGUF Qwen3VL-4B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-4B-Instruct-F16.gguf --local-dir bench\weights\qwen3-vl\4b
hf download Qwen/Qwen3-VL-8B-Instruct-GGUF Qwen3VL-8B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-8B-Instruct-F16.gguf --local-dir bench\weights\qwen3-vl\8b
```

Keep exactly one model `.gguf` and one `mmproj` `.gguf` in each folder. To compare quantizations, run once per quantization and keep the JSON files apart.

3. Pass the server path on every `fetch` and `run` with `--llama-server C:\tools\llama\llama-server.exe`, or put that folder on PATH. If `llama-server` is not found the adapter falls back to `llama-cpp-python` (`pip install -e ".[vlm-python]"`), which needs a CUDA build of that package.

### 5. Check what can run

```powershell
python -m bench list
```

Every row prints `ready` or `skipped: <reason>`. Fix whatever you want measured before the next step.

### 6. Download weights

```powershell
python -m bench fetch --llama-server C:\tools\llama\llama-server.exe
```

This loads each ready setting once, untimed, so weights land in the local caches (`bench\weights` and the usual torch, Hugging Face and insightface cache folders). Run it until every row prints `cached`. The timed run then measures load from disk, not download.

### 7. Run all stages

Close other apps that use the GPU (browsers with hardware acceleration, games, video players), then:

```powershell
python -m bench run --llama-server C:\tools\llama\llama-server.exe --audio path\to\song_with_vocals.wav
```

This runs every stage setting one after another, each in a fresh process, and writes `bench\results\run-<time>-<id>.json`. It prints a table at the end. The JSON is rewritten after each row, so a crash keeps everything measured up to that point.

Useful options:

* `--stages flow,semantics` or `--only raft:raft_small,dinov2` runs a subset. A later run of a subset is merged with earlier runs by the report.
* `--batch-cap 64` raises the batch sweep cap (default 32).
* `--clip path\to\take.mp4` uses real frames instead of the generated clip. The first 64 frames (`--frames`) are decoded at 720p and 1080p before timing starts.
* `--timeout 3600` is the per stage limit in seconds before the child process is killed.

`python -m bench run --help` lists the rest.

### 8. Paste the results into this page

```powershell
python -m bench report
python -m bench report --write docs/BENCH.md
git diff docs/BENCH.md
```

The first command prints the merged tables. The second replaces the block between the `bench:results` markers on this page with them. For each stage setting the newest result across all CUDA run files in `bench\results` wins, except that a later `skipped` never hides an earlier measurement. Smoke and CPU runs are never written to this page.

Then open a pull request with the updated `docs/BENCH.md`. The JSON files stay out of git (`bench/results` is ignored), so attach them to the pull request as a zip. Before sharing, check the `trace` and `log_tail` fields of any `error` row, they can contain local folder names.

## CPU smoke mode

`python -m bench run --smoke` runs the full harness on the CPU with stub adapters and a tiny generated clip: child processes, batch sweep, a simulated out of memory stop, a skipped row, JSON output and the table. It needs only numpy and downloads nothing. CI runs it on every pull request together with `python -m pytest`. Its numbers measure the stubs, not any model, and the report refuses to write them here.

## JSON layout

One file per run, `bench/results/<run id>.json`:

* `env`: Python, OS, GPU name, driver, total VRAM, torch version, torch CUDA build, cuDNN, ffmpeg version, versions of every candidate package, harness git commit.
* `config`: batch cap, warm up count, frame pool size, audio length and the other run options.
* `input`: generated or file input, resolutions, frame count, audio length.
* `rows`: one per stage setting with `status`, `reason`, `load_seconds`, `load_vram`, `baseline_smi_mib`, and `sweeps`. Each sweep has one entry per batch size with `throughput`, `units`, `seconds`, `iters`, `torch` (`max_allocated_mib`, `max_reserved_mib`), `smi_peak_mib` and `smi_samples`, plus a `stop` record.
