"""EmotiEffLib üzərində nazik qat: 8 sinfin ehtimalları + (mtl modellərində) valence/arousal."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import EMOTION_CLASSES, ModelConfig

log = logging.getLogger(__name__)

# EmotiEffLib çəkiləri bu qovluğa yükləyir (emotiefflib/utils.py).
WEIGHTS_DIR = Path.home() / ".emotiefflib"


@dataclass(frozen=True)
class EmotionPrediction:
    probs: np.ndarray  # (8,), EMOTION_CLASSES sırası ilə, cəmi 1
    valence: float | None
    arousal: float | None
    latency_ms: float


def weights_path(model_name: str, engine: str) -> Path:
    return WEIGHTS_DIR / f"{model_name}.{'onnx' if engine == 'onnx' else 'pt'}"


def _import_emotiefflib(engine: str):
    """emotiefflib.facial_analysis modulunu qaytarır.

    Kitabxana modul yüklənəndə torch varsa onu import edir (~2 s, +150 MB RAM) —
    hətta ONNX istifadə olunanda belə. torch yalnız çevirmə üçün qurulubsa, ONNX
    rejimində bunun qarşısını alırıq. Sonra torch engine lazım olsa, modul yenidən yüklənir.
    """
    import importlib
    import sys

    name = "emotiefflib.facial_analysis"
    module = sys.modules.get(name)
    if engine == "onnx":
        if module is not None:
            return module
        if "torch" in sys.modules:  # torch artıq yüklənib — gizlətməyin mənası yoxdur
            return importlib.import_module(name)
        sys.modules["torch"] = None  # type: ignore[assignment]  # "import torch" → ImportError
        try:
            return importlib.import_module(name)
        finally:
            del sys.modules["torch"]
    if module is not None and not hasattr(module, "torch"):
        return importlib.reload(module)  # əvvəl torch-suz yüklənmişdi
    return importlib.import_module(name)


def resolve_device(engine: str, device: str) -> str:
    """Faktiki işləyəcək cihazı qaytarır. CUDA yoxdursa CPU-ya keçir və xəbərdarlıq edir."""
    if engine == "onnx":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "engine: torch üçün torch qurulmayıb — 'pip install -r requirements-gpu.txt'"
        ) from exc
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA əlçatan deyil (torch.cuda.is_available() = False) — CPU istifadə olunur")
        return "cpu"
    return device


class EmotionRecognizer:
    def __init__(self, cfg: ModelConfig) -> None:
        path = weights_path(cfg.name, cfg.engine)
        if not path.is_file():
            # Kitabxana çatışmayan çəkini avtomatik endirərdi; iş vaxtı şəbəkəyə
            # çıxmamaq üçün bunun qarşısını alırıq.
            flag = " --torch" if cfg.engine == "torch" else ""
            raise FileNotFoundError(
                f"Model çəkisi tapılmadı: {path}. "
                f"Əvvəlcə 'python scripts/download_models.py{flag}' işə salın."
            )

        EmotiEffLibRecognizer = _import_emotiefflib(cfg.engine).EmotiEffLibRecognizer

        from .model_zoo import spec

        self.spec = spec(cfg.name)
        self.model_name = cfg.name
        self.engine = cfg.engine
        self.device = resolve_device(cfg.engine, cfg.device)
        self.is_mtl = cfg.is_mtl
        self._fer = EmotiEffLibRecognizer(engine=cfg.engine, model_name=cfg.name, device=self.device)

        classes = tuple(self._fer.idx_to_emotion_class[i] for i in range(len(EMOTION_CLASSES)))
        if classes != EMOTION_CLASSES:
            raise RuntimeError(f"Gözlənilməz sinif sırası: {classes}")

    def predict(self, face_bgr: np.ndarray) -> EmotionPrediction:
        start = time.perf_counter()
        # Kitabxana RGB gözləyir (ImageNet normallaşdırması RGB sırası ilə).
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        _, scores = self._fer.predict_emotions(rgb, logits=False)
        latency_ms = (time.perf_counter() - start) * 1000.0

        row = np.asarray(scores, dtype=np.float64)[0]
        probs = row[: len(EMOTION_CLASSES)]
        valence = arousal = None
        if self.is_mtl:
            valence, arousal = float(row[-2]), float(row[-1])
        return EmotionPrediction(probs, valence, arousal, latency_ms)
