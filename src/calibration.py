"""Nəzarətli test: göstərişlə ifadələr və hər modelin düzgünlüyü.

Zaman xətti: giriş → [hazırlıq → saxlama → rahatlama] × (ifadə sayı × raund).
Yalnız "saxlama" mərhələsinin kadrları qiymətləndirilir və CSV-yə
record_type="test", expected=<ifadə> kimi yazılır. Qiymətləndirmə hər kadrın öz
nəticəsi ilədir (smoothing yox) — əks halda əvvəlki ifadə növbətiyə qarışar.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import EMOTION_CLASSES, CalibrationConfig
from .recognizer import EmotionPrediction
from .smoothing import SmoothedResult, SmoothStatus

PREPARE, HOLD, REST, INTRO, DONE = "prepare", "hold", "rest", "intro", "done"


@dataclass(frozen=True)
class Step:
    index: int
    round: int
    expression: str


@dataclass(frozen=True)
class Position:
    phase: str
    step: Step | None
    seconds_left: float
    phase_seconds: float


def build_plan(cfg: CalibrationConfig, seed: int | None = None) -> list[Step]:
    """Raund daxilində sıra təsadüfi olur, amma Neytral (varsa) həmişə birinci — baza ifadə."""
    rng = random.Random(seed)
    steps: list[Step] = []
    for r in range(cfg.rounds):
        order = list(cfg.expressions)
        if cfg.shuffle:
            rest = [e for e in order if e != "Neutral"]
            rng.shuffle(rest)
            order = (["Neutral"] if "Neutral" in order else []) + rest
        steps += [Step(len(steps) + i, r + 1, e) for i, e in enumerate(order)]
    return steps


def raw_result(prediction: EmotionPrediction | None, threshold: float) -> SmoothedResult:
    """Tək kadrın nəticəsini CSV/hesabat formatına salır (n_frames=1)."""
    if prediction is None:
        return SmoothedResult(SmoothStatus.NO_FACE, None, None, None, None, None, 0)
    probs = np.asarray(prediction.probs, dtype=np.float64)
    idx = int(np.argmax(probs))
    top_prob = float(probs[idx])
    status = SmoothStatus.OK if top_prob >= threshold else SmoothStatus.UNCERTAIN
    return SmoothedResult(status, probs, EMOTION_CLASSES[idx], top_prob, prediction.valence, prediction.arousal, 1)


@dataclass
class ModelTally:
    frames: int = 0  # saxlama mərhələsindəki bütün kadrlar
    face: int = 0  # üz olan kadrlar
    correct: int = 0  # argmax == tapşırılan ifadə
    uncertain: int = 0  # ən yüksək ehtimal < hədd
    per_expr: dict[str, list[int]] = field(default_factory=dict)  # ifadə → [düzgün, üzlü kadr]

    def add(self, expected: str, result: SmoothedResult) -> None:
        self.frames += 1
        if result.status is SmoothStatus.NO_FACE:
            return
        self.face += 1
        hit = result.top_emotion == expected
        self.correct += hit
        self.uncertain += result.status is SmoothStatus.UNCERTAIN
        bucket = self.per_expr.setdefault(expected, [0, 0])
        bucket[0] += hit
        bucket[1] += 1

    def summary(self, expressions: list[str]) -> dict:
        per = {e: (self.per_expr[e][0] / self.per_expr[e][1] if self.per_expr.get(e, [0, 0])[1] else None)
               for e in expressions}
        known = [v for v in per.values() if v is not None]
        return {
            "frames": self.frames,
            "face": self.face,
            "accuracy": self.correct / self.face if self.face else None,
            "balanced": float(np.mean(known)) if known else None,
            "uncertain": self.uncertain / self.face if self.face else None,
            "per_expression": per,
        }


class CalibrationRun:
    def __init__(self, cfg: CalibrationConfig, clock: Callable[[], float] = time.monotonic, seed: int | None = None) -> None:
        self.cfg = cfg
        self._clock = clock
        self.steps = build_plan(cfg, seed)
        self.started = clock()
        self.cancelled = False
        self.tallies: dict[str, ModelTally] = {}
        self._step_len = cfg.prepare_seconds + cfg.hold_seconds + cfg.rest_seconds
        self.total_seconds = cfg.intro_seconds + self._step_len * len(self.steps)

    def position(self, now: float | None = None) -> Position:
        t = (self._clock() if now is None else now) - self.started
        c = self.cfg
        if self.cancelled or t >= self.total_seconds:
            return Position(DONE, None, 0.0, 0.0)
        if t < c.intro_seconds:
            return Position(INTRO, self.steps[0], c.intro_seconds - t, c.intro_seconds)
        t -= c.intro_seconds
        i = min(int(t // self._step_len), len(self.steps) - 1)
        within = t - i * self._step_len
        step = self.steps[i]
        if within < c.prepare_seconds:
            return Position(PREPARE, step, c.prepare_seconds - within, c.prepare_seconds)
        within -= c.prepare_seconds
        if within < c.hold_seconds:
            return Position(HOLD, step, c.hold_seconds - within, c.hold_seconds)
        within -= c.hold_seconds
        return Position(REST, step, c.rest_seconds - within, c.rest_seconds)

    @property
    def done(self) -> bool:
        return self.position().phase == DONE

    def expected_now(self) -> str | None:
        pos = self.position()
        return pos.step.expression if pos.phase == HOLD else None

    def record(self, model: str, expected: str, result: SmoothedResult) -> None:
        self.tallies.setdefault(model, ModelTally()).add(expected, result)

    def cancel(self) -> None:
        self.cancelled = True

    def summary(self) -> dict[str, dict]:
        return {m: t.summary(self.cfg.expressions) for m, t in self.tallies.items()}
