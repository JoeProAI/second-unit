from bench import registry
from bench.runner import Config, make_context, run_setting


def test_every_requested_candidate_has_an_adapter():
    pairs = {(a.name, s) for a, s in registry.select()}
    for want in [
        ("transnetv2", "default"), ("autoshot", "default"), ("omnishotcut", "v1.5"),
        ("raft", "raft_large"), ("raft", "raft_small"), ("insightface", "buffalo_l"),
        ("dinov2", "vitb14"), ("dinov2", "vitl14"), ("open_clip", "ViT-L-14/laion2b_s32b_b82k"),
        ("qwen3_vl", "4b"), ("mediapipe_hands", "landmarker_f16"), ("demucs", "htdemucs"),
        ("beat_this", "final0"), ("madmom", "rnn_dbn"), ("faster_whisper", "small"),
        ("faster_whisper", "large-v3"), ("ffmpeg", "h264_nvenc"), ("ffmpeg", "hevc_nvenc"),
        ("ffmpeg", "libx264"),
    ]:
        assert want in pairs, want


def test_select_filters():
    assert {a.stage for a, _ in registry.select(stages=["flow"])} == {"flow"}
    assert [(a.name, s) for a, s in registry.select(only=["raft:raft_small"])] == [("raft", "raft_small")]


def test_missing_never_raises_for_real_adapters(tmp_path):
    for device in ("cuda", "cpu"):
        ctx = make_context(Config(device=device, weights_dir=str(tmp_path)))
        for adapter, setting in registry.select():
            reason = adapter.missing(setting, ctx)
            assert reason is None or isinstance(reason, str)


def test_real_torch_adapters_skip_without_cuda(tmp_path):
    cfg = Config(device="cuda", weights_dir=str(tmp_path))
    ctx = make_context(cfg)
    if ctx.cuda_ok:
        return
    for adapter, setting in registry.select():
        if not adapter.uses_torch:
            continue
        row = run_setting(adapter, setting, cfg)
        assert row["status"] == "skipped", (adapter.name, row)
