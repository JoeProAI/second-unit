"""Face identity: insightface buffalo_l (RetinaFace detector plus ArcFace R50)."""

from __future__ import annotations

from bench.adapter import VIDEO, Adapter, add_torch_cuda_dlls


class InsightFace(Adapter):
    name = "insightface"
    stage = "face"
    kind = VIDEO
    label = "ArcFace via insightface"
    settings = ("buffalo_l",)
    requires = ("insightface", "onnxruntime", "cv2")
    extra = "face"
    uses_torch = False
    notes = (
        "per frame: detector at det_size 640x640, then the ArcFace embedder on one 112x112 "
        "center crop per frame, embedder batched across the frames in the call; the crop is "
        "synthetic so the embedder cost does not depend on faces being found"
    )

    def check(self, setting, ctx):
        add_torch_cuda_dlls()
        import onnxruntime as ort

        if ctx.device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
            return "onnxruntime has no CUDAExecutionProvider (install onnxruntime-gpu, not onnxruntime)"
        return None

    def load(self, setting, ctx):
        add_torch_cuda_dlls()
        from insightface.app import FaceAnalysis

        providers = ["CUDAExecutionProvider"] if ctx.device == "cuda" else ["CPUExecutionProvider"]
        root = str(ctx.weights_dir / "insightface")
        self.app = FaceAnalysis(
            name=setting, root=root, providers=providers, allowed_modules=["detection", "recognition"]
        )
        self.app.prepare(ctx_id=0 if ctx.device == "cuda" else -1, det_size=(640, 640))
        self.det = self.app.det_model
        self.rec = self.app.models["recognition"]

    def process(self, batch):
        import cv2

        crops = []
        for frame in self.take(batch):
            bgr = frame[:, :, ::-1]
            self.det.detect(bgr, max_num=0, metric="default")
            h, w = bgr.shape[:2]
            s = min(h, w) // 2
            crop = bgr[(h - s) // 2 : (h + s) // 2, (w - s) // 2 : (w + s) // 2]
            crops.append(cv2.resize(crop, (112, 112)))
        self.rec.get_feat(crops)
        return float(batch)

    def unload(self):
        self.app = self.det = self.rec = None
