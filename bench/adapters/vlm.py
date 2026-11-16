"""VLM defect checklist: Qwen3-VL GGUF through llama.cpp.

Two backends, picked in this order:

1. llama-server (the llama.cpp server binary) when it is on PATH or given with
   --llama-server. The adapter starts it, waits for /health, and sends OpenAI
   style chat requests with images. VRAM comes from the nvidia-smi sampler.
2. llama-cpp-python (import llama_cpp) with its MTMDChatHandler.

Model files go in bench/weights/qwen3-vl/<setting>/: one main .gguf and one
mmproj .gguf (file name containing "mmproj").
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from bench.adapter import VIDEO, Adapter, LimitReached, has_module

PROMPT = (
    "You are checking frames from AI generated video for defects. For each image answer with "
    "one JSON object per image with these keys: flicker, morphing, identity_drift, broken_hands, "
    "extra_limbs, warped_text, other. Each value is true or false."
)
MAX_TOKENS = 128
CTX = 16384


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Qwen3VL(Adapter):
    name = "qwen3_vl"
    stage = "vlm"
    kind = VIDEO
    label = "Qwen3-VL Instruct GGUF (llama.cpp)"
    settings = ("2b", "4b", "8b")
    requires = ("PIL",)
    extra = "vlm"
    uses_torch = False
    max_batch = 8
    notes = (
        f"all layers on the GPU, context {CTX}, temperature 0, max {MAX_TOKENS} output tokens; "
        "batch = images in one request; frames JPEG encoded before timing; "
        "throughput depends on how many tokens the model writes"
    )

    def folder(self, setting, ctx) -> Path:
        return Path(ctx.weights_dir) / "qwen3-vl" / setting

    def files(self, setting, ctx):
        folder = self.folder(setting, ctx)
        ggufs = sorted(folder.glob("*.gguf")) if folder.exists() else []
        mmproj = [p for p in ggufs if "mmproj" in p.name.lower()]
        main = [p for p in ggufs if "mmproj" not in p.name.lower()]
        return (main[0] if main else None), (mmproj[0] if mmproj else None)

    def server_bin(self, ctx) -> str | None:
        explicit = ctx.options.get("llama_server")
        if explicit:
            return explicit if Path(explicit).exists() else None
        return shutil.which("llama-server")

    def check(self, setting, ctx):
        main, mmproj = self.files(setting, ctx)
        if main is None or mmproj is None:
            return f"GGUF files not found in bench/weights/qwen3-vl/{setting} (need model and mmproj)"
        if ctx.device != "cuda":
            return "VLM bench only runs on cuda"
        if self.server_bin(ctx) is None and not has_module("llama_cpp"):
            return "neither llama-server on PATH nor llama-cpp-python installed"
        return None

    def load(self, setting, ctx):
        main, mmproj = self.files(setting, ctx)
        self.backend = "server" if self.server_bin(ctx) else "python"
        if self.backend == "server":
            self.port = _free_port()
            cmd = [
                self.server_bin(ctx), "-m", str(main), "--mmproj", str(mmproj), "-ngl", "999",
                "-c", str(CTX), "-np", "1", "--host", "127.0.0.1", "--port", str(self.port),
            ]
            log_dir = Path(ctx.options.get("log_dir") or ctx.weights_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.server_log = open(log_dir / f"llama-server-{setting}.log", "a", encoding="utf-8")
            self.proc = subprocess.Popen(cmd, stdout=self.server_log, stderr=subprocess.STDOUT)
            deadline = time.time() + 600
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"llama-server exited with code {self.proc.returncode}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                        if r.status == 200:
                            return
                except (urllib.error.URLError, ConnectionError, OSError):
                    pass
                time.sleep(0.25)
            raise RuntimeError("llama-server did not become healthy within 600 s")
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import MTMDChatHandler

        self.llm = Llama(
            model_path=str(main), chat_handler=MTMDChatHandler(clip_model_path=str(mmproj)),
            n_gpu_layers=-1, n_ctx=CTX, verbose=False,
        )

    def prepare(self, key, media):
        from PIL import Image

        super().prepare(key, media)
        self.jpegs = []
        for frame in self.frames:
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="JPEG", quality=90)
            self.jpegs.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
        self.frames = self.jpegs

    def messages(self, batch):
        content = [{"type": "image_url", "image_url": {"url": u}} for u in self.take(batch)]
        content.append({"type": "text", "text": PROMPT})
        return [{"role": "user", "content": content}]

    def process(self, batch):
        body = {"messages": self.messages(batch), "max_tokens": MAX_TOKENS, "temperature": 0}
        if self.backend == "python":
            try:
                self.llm.create_chat_completion(**body)
            except ValueError as exc:
                raise LimitReached(str(exc)) from exc
            return float(batch)
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                json.loads(r.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if "out of memory" in detail.lower():
                raise RuntimeError(f"out of memory: {detail}") from exc
            raise LimitReached(f"HTTP {exc.code}: {detail}") from exc
        return float(batch)

    def unload(self):
        if getattr(self, "proc", None) is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
            self.server_log.close()
        self.llm = None
