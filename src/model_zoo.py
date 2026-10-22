"""Bütün dəstəklənən emosiya modellərinin reyestri.

İki ailə var:
- EmotiEffLib (AffectNet, 8 sinif; bəziləri valence/arousal da verir)
- FER2013 modelləri (7 sinif — "Contempt" yoxdur): DeepFace, ViT (Hugging Face), Kaggle CNN

Hamısı eyni interfeysə uyğunlaşdırılır: EMOTION_CLASSES sırası ilə 8 ehtimal.
Modelin dəstəkləmədiyi sinfin ehtimalı 0-dır və CSV-də boş yazılır.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import EMOTION_CLASSES, PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"

# FER2013 etiket sırası → bizim siniflər
FER_LABELS: tuple[str, ...] = ("Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral")
FER_CLASSES = frozenset(FER_LABELS)


@dataclass(frozen=True)
class ModelSpec:
    key: str  # config və CSV-dəki ad
    label: str  # UI-dakı qısa ad
    family: str  # "emotiefflib" | "fer"
    dataset: str
    license: str
    classes: frozenset[str]
    has_va: bool = False
    onnx_file: str | None = None  # FER modelləri üçün models/ altında
    # FER ön emalı
    color: str = "gray"  # "gray" | "rgb"
    size: int = 48
    scale: float = 1 / 255
    mean: float = 0.0
    std: float = 1.0
    source: str = ""

    @property
    def onnx_path(self) -> Path | None:
        return MODELS_DIR / self.onnx_file if self.onnx_file else None

    @property
    def is_mtl(self) -> bool:
        return self.has_va


ALL8 = frozenset(EMOTION_CLASSES)

SPECS: dict[str, ModelSpec] = {
    s.key: s
    for s in (
        ModelSpec("enet_b0_8_best_afew", "enet_b0 · afew", "emotiefflib", "AffectNet", "Apache-2.0 (çəkilər: AffectNet, qeyri-kommersiya)", ALL8,
                  source="github.com/sb-ai-lab/EmotiEffLib"),
        ModelSpec("enet_b0_8_best_vgaf", "enet_b0 · vgaf", "emotiefflib", "AffectNet", "Apache-2.0 (çəkilər: AffectNet, qeyri-kommersiya)", ALL8,
                  source="github.com/sb-ai-lab/EmotiEffLib"),
        ModelSpec("enet_b0_8_va_mtl", "enet_b0 · VA", "emotiefflib", "AffectNet", "Apache-2.0 (çəkilər: AffectNet, qeyri-kommersiya)", ALL8,
                  has_va=True, source="github.com/sb-ai-lab/EmotiEffLib"),
        ModelSpec("enet_b2_8", "enet_b2", "emotiefflib", "AffectNet", "Apache-2.0 (çəkilər: AffectNet, qeyri-kommersiya)", ALL8,
                  source="github.com/sb-ai-lab/EmotiEffLib"),
        ModelSpec("mbf_va_mtl", "mbf · VA", "emotiefflib", "AffectNet", "Apache-2.0 (çəkilər: AffectNet, qeyri-kommersiya)", ALL8,
                  has_va=True, source="github.com/sb-ai-lab/EmotiEffLib"),
        ModelSpec("mobilevit_va_mtl", "mobilevit · VA", "emotiefflib", "AffectNet", "Apache-2.0 (çəkilər: AffectNet, qeyri-kommersiya)", ALL8,
                  has_va=True, source="github.com/sb-ai-lab/EmotiEffLib"),
        ModelSpec("deepface_emotion", "DeepFace", "fer", "FER2013", "MIT (DeepFace)", FER_CLASSES,
                  onnx_file="deepface_emotion.onnx", color="gray", size=48, scale=1 / 255,
                  source="github.com/serengil/deepface"),
        ModelSpec("vit_fer2013", "ViT (HF)", "fer", "FER2013", "Apache-2.0", FER_CLASSES,
                  onnx_file="vit_fer2013.onnx", color="rgb", size=224, scale=1 / 255, mean=0.5, std=0.5,
                  source="huggingface.co/abhilash88/face-emotion-detection"),
        ModelSpec("kaggle_fer_cnn", "Kaggle CNN", "fer", "FER2013", "CC0", FER_CLASSES,
                  onnx_file="kaggle_fer_cnn.onnx", color="gray", size=48, scale=1 / 255,
                  source="kaggle.com/datasets/abhisheksingh016/machine-model-for-emotion-detection"),
    )
}

EMOTIEFFLIB_MODELS = tuple(k for k, s in SPECS.items() if s.family == "emotiefflib")
FER_MODELS = tuple(k for k, s in SPECS.items() if s.family == "fer")


def spec(key: str) -> ModelSpec:
    try:
        return SPECS[key]
    except KeyError:
        raise KeyError(f"Naməlum model: {key}") from None


def is_installed(key: str, engine: str = "onnx") -> bool:
    s = spec(key)
    if s.family == "fer":
        return s.onnx_path is not None and s.onnx_path.is_file()
    from .recognizer import weights_path

    return weights_path(key, engine).is_file()


def installed_models() -> list[str]:
    return [k for k in SPECS if is_installed(k)]


def make_recognizer(key: str, engine: str = "onnx", device: str = "cpu"):
    """Reyestrdəki istənilən model üçün eyni interfeysli tanıyıcı (predict(face_bgr))."""
    s = spec(key)
    if s.family == "fer":
        from .fer_models import FerOnnxRecognizer

        return FerOnnxRecognizer(s)
    from .config import ModelConfig
    from .recognizer import EmotionRecognizer

    return EmotionRecognizer(ModelConfig(key, engine, device))
