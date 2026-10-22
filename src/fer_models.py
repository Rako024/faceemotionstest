"""FER2013 ailəsinin modelləri üçün ümumi ONNX adapteri (DeepFace, ViT, Kaggle CNN).

Giriş formatı ONNX qrafından oxunur (NHWC və ya NCHW, 1 və ya 3 kanal); ön emal
parametrləri (rəng, ölçü, miqyas, mean/std) model reyestrindən gəlir.
Çıxış ehtimal deyilsə (logit), softmax tətbiq olunur.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .config import EMOTION_CLASSES
from .model_zoo import FER_LABELS, ModelSpec
from .recognizer import EmotionPrediction

_FER_TO_OURS = np.array([EMOTION_CLASSES.index(label) for label in FER_LABELS])


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def preprocess_face(face_bgr: np.ndarray, spec: ModelSpec, channels: int, nchw: bool) -> np.ndarray:
    """Üz kəsiyini modelin gözlədiyi tensora çevirir: (1, H, W, C) və ya (1, C, H, W)."""
    if channels == 1:
        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    # Kiçiltmədə INTER_AREA, böyütmədə (ViT 224) INTER_LINEAR — HF prosessorunun bilinear-ına yaxın
    interp = cv2.INTER_AREA if max(img.shape[:2]) > spec.size else cv2.INTER_LINEAR
    img = cv2.resize(img, (spec.size, spec.size), interpolation=interp).astype(np.float32)
    img = (img * spec.scale - spec.mean) / spec.std
    if img.ndim == 2:
        img = img[..., None]
    if nchw:
        img = img.transpose(2, 0, 1)
    return np.ascontiguousarray(img[None, ...], dtype=np.float32)


class FerOnnxRecognizer:
    def __init__(self, spec: ModelSpec) -> None:
        import onnxruntime as ort

        path = spec.onnx_path
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"{spec.label} modeli tapılmadı: {path}. "
                "Əvvəlcə 'python scripts/download_models.py --extra' işə salın."
            )
        self.spec = spec
        self.model_name = spec.key
        self.engine = "onnx"
        self.device = "cpu"
        self.is_mtl = False
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        shape = list(inp.shape)
        # [N, H, W, C] və ya [N, C, H, W]
        self._nchw = isinstance(shape[1], int) and shape[1] in (1, 3) and not (isinstance(shape[-1], int) and shape[-1] in (1, 3))
        channels = shape[1] if self._nchw else shape[-1]
        self._channels = channels if isinstance(channels, int) else (1 if spec.color == "gray" else 3)
        out = self._session.get_outputs()[0].shape
        if out[-1] != len(FER_LABELS):
            raise RuntimeError(f"{spec.key}: gözlənilən 7 sinif, alındı {out}")

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        return preprocess_face(face_bgr, self.spec, self._channels, self._nchw)

    def predict(self, face_bgr: np.ndarray) -> EmotionPrediction:
        start = time.perf_counter()
        out = self._session.run(None, {self._input_name: self._preprocess(face_bgr)})[0][0].astype(np.float64)
        latency_ms = (time.perf_counter() - start) * 1000.0
        probs7 = out if (out.min() >= 0 and abs(out.sum() - 1) < 1e-3) else _softmax(out)
        probs = np.zeros(len(EMOTION_CLASSES))
        probs[_FER_TO_OURS] = probs7
        return EmotionPrediction(probs, None, None, latency_ms)
