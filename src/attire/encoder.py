"""SigLIP 2 zero-shot: geyimin növü, rəngi, qalstuk və pencək.

Şəkil və mətn eyni vektor fəzasına salınır. Hər sualın (məs. "garment") hər cavabı
(məs. "shirt") bir neçə mətn təsviri ilə verilir; təsvirlərin vektorları ortalanır
(CLIP-də standart "prompt ensembling"), sonra cavablar arasında softmax. Mətn təsvirləri
config.yaml-dadır (attire.prompts) — bank öz qaydasına görə kodu dəyişmədən redaktə edə bilər.

Model yalnız lokal qovluqdan yüklənir (şəbəkə yox). torch/transformers yalnız bu modulda.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

TEXT_MAX_LENGTH = 64  # SigLIP mətnləri bu uzunluğa qədər doldurulmuş halda öyrədilib
BATCH = 16


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _tensor(out):
    """transformers versiyasından asılı olaraq get_*_features tensor və ya çıxış obyekti qaytarır."""
    return getattr(out, "pooler_output", out)


class SiglipEncoder:
    def __init__(self, model_dir: Path, threads: int | None = None) -> None:
        model_dir = Path(model_dir)
        if not (model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"SigLIP 2 modeli tapılmadı: {model_dir}. Əvvəlcə 'python scripts/download_models.py --attire'."
            )
        import torch
        from transformers import AutoModel, AutoTokenizer

        if threads:  # canlı rejimdə emosiya pipeline-ı CPU-suz qalmasın
            torch.set_num_threads(threads)
        self._torch = torch
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.size = int(self.model.config.vision_config.image_size)
        self.logit_scale = float(self.model.logit_scale.detach().exp())

    def _pixels(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        """SiglipImageProcessor ilə eyni: RGB, bilinear ölçü, [-1, 1]."""
        batch = np.empty((len(images_bgr), 3, self.size, self.size), np.float32)
        for i, img in enumerate(images_bgr):
            rgb = cv2.cvtColor(cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR),
                               cv2.COLOR_BGR2RGB)
            batch[i] = (rgb.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
        return batch

    def encode_images(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        """(N, D) normallaşdırılmış şəkil vektorları."""
        out = []
        with self._torch.inference_mode():
            for i in range(0, len(images_bgr), BATCH):
                pixels = self._torch.from_numpy(self._pixels(images_bgr[i : i + BATCH]))
                out.append(_tensor(self.model.get_image_features(pixel_values=pixels)).numpy())
        return _normalize(np.concatenate(out).astype(np.float32))

    def drop_text_model(self) -> None:
        """Mətn vektorları hesablandıqdan sonra mətn hissəsini yaddaşdan çıxarır (~1.1 GB — 256k sözlük)."""
        import gc

        self.model.text_model = None
        self.tokenizer = None
        gc.collect()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """(N, D) normallaşdırılmış mətn vektorları. SigLIP 2 kiçik hərflə öyrədilib."""
        if self.tokenizer is None:
            raise RuntimeError("SigLIP 2-nin mətn hissəsi yaddaşdan çıxarılıb")
        tokens = self.tokenizer([t.lower() for t in texts], padding="max_length", max_length=TEXT_MAX_LENGTH,
                                truncation=True, return_tensors="pt")
        with self._torch.inference_mode():
            feats = _tensor(self.model.get_text_features(**tokens)).numpy()
        return _normalize(feats.astype(np.float32))


@dataclass
class Question:
    """Bir sual (məs. "garment") və onun cavablarının ortalanmış mətn vektorları."""

    name: str
    labels: list[str]
    embeddings: np.ndarray  # (L, D)

    def probs(self, image_emb: np.ndarray, logit_scale: float) -> np.ndarray:
        """(N, L) — cavablar arasında softmax."""
        logits = logit_scale * image_emb @ self.embeddings.T
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)


def build_question(name: str, answers: dict[str, list[str]], encode_texts) -> Question:
    labels = list(answers)
    flat = [p for label in labels for p in answers[label]]
    emb = encode_texts(flat)
    out, start = [], 0
    for label in labels:
        n = len(answers[label])
        out.append(emb[start : start + n].mean(axis=0))
        start += n
    return Question(name, labels, _normalize(np.stack(out)))


@dataclass
class Models:
    """Geyim yoxlaması üçün hər şey: üz detektoru, enkoder, zero-shot suallar."""

    detector: Any
    encoder: Any  # encode_images / encode_texts / logit_scale
    zero_shot: "ZeroShot"


class ZeroShot:
    """Bütün suallar birlikdə. encoder: encode_images / encode_texts / logit_scale olan obyekt."""

    def __init__(self, encoder, prompts: dict[str, dict[str, list[str]]]) -> None:
        self.encoder = encoder
        self.questions = {name: build_question(name, answers, encoder.encode_texts) for name, answers in prompts.items()}

    def __call__(self, image_emb: np.ndarray) -> dict[str, np.ndarray]:
        """Sual → (N, L) ehtimallar."""
        return {name: q.probs(image_emb, self.encoder.logit_scale) for name, q in self.questions.items()}
