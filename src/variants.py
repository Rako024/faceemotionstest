"""Ön emal variantları: kəsik boşluğu, göz xəttinə görə düzləndirmə, flip TTA.

Eyni funksiyalar həm canlı rejimdə (config: detector.crop_margin / detector.align /
model.flip_tta), həm də oflayn qiymətləndiricidə (python -m src.evaluate) istifadə olunur —
beləliklə oflayn tapılan ən yaxşı ayar canlıda eyni cür işləyir.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .recognizer import EmotionPrediction


@dataclass(frozen=True)
class Variant:
    margin: float = 0.1  # üz qutusunun hər tərəfinə, qutu ölçüsünə nisbətən
    align: bool = False  # gözləri üfüqi et
    flip: bool = False  # orijinal + güzgü əksinin ortalaması

    @property
    def name(self) -> str:
        parts = [f"boşluq {round(self.margin * 100)}%"]
        if self.align:
            parts.append("düzləndirmə")
        if self.flip:
            parts.append("flip")
        return " + ".join(parts)

    @property
    def key(self) -> str:
        return f"m{round(self.margin * 100)}" + ("_al" if self.align else "") + ("_fl" if self.flip else "")


def roll_degrees(landmarks: np.ndarray) -> float:
    """Göz xəttinin üfüqə bucağı. YuNet sırası: [0] sağ göz (şəkildə solda), [1] sol göz."""
    (x0, y0), (x1, y1) = landmarks[0], landmarks[1]
    return math.degrees(math.atan2(float(y1 - y0), float(x1 - x0)))


def alignment_matrix(landmarks: np.ndarray) -> np.ndarray:
    """Gözləri üfüqi edən fırlanma matrisi (göz ortası ətrafında)."""
    center = (landmarks[0] + landmarks[1]) / 2.0
    return cv2.getRotationMatrix2D((float(center[0]), float(center[1])), roll_degrees(landmarks), 1.0)


def face_crop(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    landmarks: np.ndarray | None,
    margin: float,
    align: bool,
) -> np.ndarray:
    """Üz kəsiyi: (x, y, w, h) qutusu + hər tərəfə margin·ölçü, istəyə görə düzləndirilmiş.

    margin=0 — EmotiEffLib-in öyrədilməsindəki kimi (AffectNet qutusu boşluqsuz kəsilib).
    """
    x, y, w, h = box
    if align and landmarks is not None:
        m = alignment_matrix(landmarks)
        # Fırlanma bütün şəklə yox, üz ətrafındakı sahəyə tətbiq olunsa da nəticə eynidir;
        # sadəlik üçün bütün şəkil fırladılır (sərhəd təkrarlanır — qara künclər yaranmır).
        image = cv2.warpAffine(image, m, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        cx, cy = m @ np.array([x + w / 2.0, y + h / 2.0, 1.0])
        x, y = cx - w / 2.0, cy - h / 2.0
    height, width = image.shape[:2]
    mx, my = w * margin, h * margin
    x0 = max(0, int(math.floor(x - mx)))
    y0 = max(0, int(math.floor(y - my)))
    x1 = min(width, int(math.ceil(x + w + mx)))
    y1 = min(height, int(math.ceil(y + h + my)))
    return image[y0:y1, x0:x1]


def predict(recognizer, crop: np.ndarray, flip: bool) -> EmotionPrediction:
    """Flip TTA ilə və ya onsuz proqnoz. Flip-də ehtimallar və VA ortalanır, vaxt cəmlənir."""
    p = recognizer.predict(crop)
    if not flip:
        return p
    start = time.perf_counter()
    q = recognizer.predict(cv2.flip(crop, 1))
    latency = p.latency_ms + (time.perf_counter() - start) * 1000.0
    probs = (np.asarray(p.probs) + np.asarray(q.probs)) / 2.0
    valence = None if p.valence is None else (p.valence + q.valence) / 2.0
    arousal = None if p.arousal is None else (p.arousal + q.arousal) / 2.0
    return EmotionPrediction(probs, valence, arousal, latency)
