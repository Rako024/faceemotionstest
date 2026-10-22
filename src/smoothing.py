"""Zaman pəncərəsi üzrə hərəkətli ortalama.

- Son `window_seconds` ərzində üzü olan kadrların ehtimalları ortalanır.
- Tək kadrın nəticəsi göstərilmir: pəncərədə `min_frames`-dən az kadr varsa → warming_up.
- Ən yüksək orta ehtimal `uncertain_threshold`-dan aşağıdırsa → uncertain.
- Hazırkı kadrda üz yoxdursa → no_face; köhnə təxmin göstərilmir.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from .config import EMOTION_CLASSES, SmoothingConfig
from .recognizer import EmotionPrediction


class SmoothStatus(str, Enum):
    OK = "ok"
    UNCERTAIN = "uncertain"
    NO_FACE = "no_face"
    WARMING_UP = "warming_up"


@dataclass(frozen=True)
class SmoothedResult:
    status: SmoothStatus
    probs: np.ndarray | None  # pəncərə ortalaması; no_face/warming_up-da None
    top_emotion: str | None  # ən yüksək ehtimallı sinif (uncertain-da da doldurulur)
    top_prob: float | None
    valence: float | None
    arousal: float | None
    n_frames: int  # pəncərədəki üzlü kadr sayı

    @property
    def has_estimate(self) -> bool:
        return self.status in (SmoothStatus.OK, SmoothStatus.UNCERTAIN)


@dataclass(frozen=True)
class _Sample:
    t: float
    probs: np.ndarray
    valence: float | None
    arousal: float | None


class EmotionSmoother:
    def __init__(self, cfg: SmoothingConfig, clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self._clock = clock
        self._samples: deque[_Sample] = deque()
        self._face_present = False

    def reset(self) -> None:
        self._samples.clear()
        self._face_present = False

    def add(self, prediction: EmotionPrediction | None, t: float | None = None) -> SmoothedResult:
        """Yeni kadrın nəticəsini əlavə edir. `None` — bu kadrda istifadəyə yararlı üz yoxdur."""
        now = self._clock() if t is None else t
        self._face_present = prediction is not None
        if prediction is not None:
            self._samples.append(
                _Sample(now, np.asarray(prediction.probs, dtype=np.float64), prediction.valence, prediction.arousal)
            )
        return self.current(now)

    def current(self, t: float | None = None) -> SmoothedResult:
        """Vəziyyəti dəyişmədən hazırkı nəticə (məs. özünüqiymətləndirmə anı üçün)."""
        now = self._clock() if t is None else t
        self._prune(now)
        n = len(self._samples)
        if not self._face_present:
            return SmoothedResult(SmoothStatus.NO_FACE, None, None, None, None, None, n)
        if n < self.cfg.min_frames:
            return SmoothedResult(SmoothStatus.WARMING_UP, None, None, None, None, None, n)

        probs = np.mean([s.probs for s in self._samples], axis=0)
        idx = int(np.argmax(probs))
        top_prob = float(probs[idx])
        status = SmoothStatus.UNCERTAIN if top_prob < self.cfg.uncertain_threshold else SmoothStatus.OK
        return SmoothedResult(
            status,
            probs,
            EMOTION_CLASSES[idx],
            top_prob,
            _mean_or_none([s.valence for s in self._samples]),
            _mean_or_none([s.arousal for s in self._samples]),
            n,
        )

    def _prune(self, now: float) -> None:
        cutoff = now - self.cfg.window_seconds
        while self._samples and self._samples[0].t <= cutoff:
            self._samples.popleft()


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None
