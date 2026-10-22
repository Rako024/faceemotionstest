"""Hesabat testləri və nümayiş üçün sintetik CSV logları (real CsvLogger sxemi ilə)."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.config import EMOTION_CLASSES
from src.csv_logger import CsvLogger
from src.detector import FaceStatus
from src.smoothing import SmoothedResult, SmoothStatus

CATS = ["neutral", "happy", "tired", "angry", "sad", "stressed"]
CAT_TO_CLASS = {"neutral": "Neutral", "happy": "Happiness", "angry": "Anger", "sad": "Sadness", "stressed": "Fear"}


def smoothed(status: str, top: str | None = None, p: float = 0.7, valence=None, arousal=None, n=10) -> SmoothedResult:
    st = SmoothStatus(status)
    if st in (SmoothStatus.NO_FACE, SmoothStatus.WARMING_UP):
        return SmoothedResult(st, None, None, None, None, None, n)
    probs = np.full(len(EMOTION_CLASSES), (1 - p) / (len(EMOTION_CLASSES) - 1))
    probs[EMOTION_CLASSES.index(top)] = p
    return SmoothedResult(st, probs, top, p, valence, arousal, n)


def write_synthetic_test(
    log_dir: Path,
    accuracy: dict[str, dict[str, float]],
    frames_per_expr: int = 20,
    operator: str = "OP-5T",
    seed: int = 3,
) -> None:
    """Nəzarətli test sətirləri: accuracy = {model: {ifadə: düzgün kadr payı}}.

    Düzgün olmayan kadrlar "Neutral" (Neytral üçün "Happiness") verir; hər 4-cü düzgün kadr
    qeyri-müəyyəndir (ehtimal 0.45) — argmax-ın bərpasını yoxlamaq üçün.
    """
    from src.calibration import raw_result
    from src.csv_logger import ModelTag
    from src.model_zoo import SPECS
    from src.recognizer import EmotionPrediction

    rng = random.Random(seed)
    logger = CsvLogger(log_dir, operator)
    start = datetime(2026, 10, 21, 15, 0)
    for model, accs in accuracy.items():
        spec = SPECS[model]
        tag = ModelTag(model, "onnx", "cpu", spec.classes if len(spec.classes) < 8 else None)
        i = 0
        for expr, acc in accs.items():
            n_ok = round(acc * frames_per_expr)
            for k in range(frames_per_expr):
                hit = k < n_ok
                top = expr if hit else ("Happiness" if expr == "Neutral" else "Neutral")
                p = 0.45 if hit and k % 4 == 3 else 0.8
                probs = np.full(8, (1 - p) / 7)
                probs[EMOTION_CLASSES.index(top)] = p
                if model in ("deepface_emotion", "vit_fer2013", "kaggle_fer_cnn"):
                    probs[EMOTION_CLASSES.index("Contempt")] = 0.0
                res = raw_result(EmotionPrediction(probs, None, None, 5.0), 0.5)
                ts = start + timedelta(seconds=i)
                logger.log_test(ts, res, FaceStatus.OK, 6.0, rng.uniform(1, 20), 10.0, expr, model=tag)
                i += 1
        # bir üzsüz kadr — məxrəcə düşməməlidir
        logger.log_test(start + timedelta(seconds=i), raw_result(None, 0.5), FaceStatus.NOT_FOUND, 6.0, None, 10.0,
                        "Neutral", model=tag)
    logger.close()


def write_synthetic_logs(
    log_dir: Path,
    operators=("OP-1A", "OP-2B", "OP-3C"),
    days: int = 2,
    hours=range(9, 18),
    samples_per_hour: int = 40,
    reports_per_hour: int = 1,
    agreement: float = 0.6,
    model: str = "enet_b0_8_best_afew",
    seed: int = 7,
    extra_models: dict[str, float] | None = None,
) -> None:
    """extra_models: {model_key: əsas modellə eyni nəticə ehtimalı} — eyni timestamp ilə əlavə sətirlər."""
    from src.csv_logger import ModelTag
    from src.model_zoo import SPECS

    rng = random.Random(seed)
    mtl = model.endswith("_mtl")
    start = datetime(2026, 10, 20)
    reasons = [FaceStatus.NOT_FOUND, FaceStatus.OCCLUDED, FaceStatus.PROFILE, FaceStatus.PARTIAL, FaceStatus.LOW_CONFIDENCE]
    extras = extra_models or {}

    def variants(res: SmoothedResult):
        """Əlavə modellərin nəticəsi: əsas modelə bənzər, amma bəzən fərqli."""
        out = []
        for key, same in extras.items():
            spec = SPECS[key]
            tag = ModelTag(key, "onnx", "cpu", spec.classes if len(spec.classes) < 8 else None)
            if res.status in (SmoothStatus.NO_FACE, SmoothStatus.WARMING_UP):
                out.append((tag, res))
                continue
            top = res.top_emotion if rng.random() < same else rng.choice(EMOTION_CLASSES)
            if top not in spec.classes:
                top = "Neutral"
            p = min(0.97, max(0.3, (res.top_prob or 0.6) + rng.uniform(-0.15, 0.15)))
            out.append((tag, smoothed("ok" if p >= 0.5 else "uncertain", top, p)))
        return out

    for op in operators:
        logger = CsvLogger(log_dir, op, model, "onnx", "cpu")
        _log_sample, _log_report = logger.log_sample, logger.log_self_report

        def log_sample(ts, res, reason, det, model_ms, cpu, _ls=_log_sample):
            _ls(ts, res, reason, det, model_ms, cpu)
            for tag, r2 in variants(res):
                _ls(ts, r2, reason, det, None if model_ms is None else model_ms * rng.uniform(0.05, 8), cpu, model=tag)

        def log_report(ts, res, answer, _lr=_log_report):
            _lr(ts, res, answer)
            for tag, r2 in variants(res):
                _lr(ts, r2, answer, model=tag)

        logger.log_sample, logger.log_self_report = log_sample, log_report
        for d in range(days):
            for h in hours:
                base = start + timedelta(days=d, hours=h)
                happy_w = 0.35 if h < 12 else 0.15
                for i in range(samples_per_hour):
                    ts = base + timedelta(seconds=i * 60 // max(1, samples_per_hour) * 1)
                    r = rng.random()
                    if r < 0.14:
                        res, reason = smoothed("no_face"), rng.choice(reasons)
                    elif r < 0.17:
                        res, reason = smoothed("warming_up", n=1), FaceStatus.OK
                    else:
                        top = rng.choices(
                            ["Neutral", "Happiness", "Sadness", "Anger", "Surprise", "Fear", "Disgust", "Contempt"],
                            [0.45, happy_w, 0.1, 0.05, 0.05, 0.03, 0.02, 0.02],
                        )[0]
                        p = rng.uniform(0.35, 0.95)
                        status = "ok" if p >= 0.5 else "uncertain"
                        va = (rng.uniform(-0.5, 0.6), rng.uniform(-0.6, 0.4)) if mtl else (None, None)
                        res, reason = smoothed(status, top, p, *va), FaceStatus.OK
                    det = rng.uniform(3, 9)
                    model_ms = rng.uniform(9, 20) if res.status is not SmoothStatus.NO_FACE else None
                    logger.log_sample(ts, res, reason, det, model_ms, rng.uniform(5, 20))
                for _ in range(reports_per_hour):
                    ts = base + timedelta(minutes=rng.randint(0, 59))
                    answer = rng.choices(CATS + ["declined", "timeout"], [0.3, 0.2, 0.12, 0.08, 0.1, 0.1, 0.05, 0.05])[0]
                    if answer in ("declined", "timeout") or answer == "tired":
                        top = rng.choice(["Neutral", "Sadness"])
                    elif rng.random() < agreement:
                        top = CAT_TO_CLASS[answer]
                    else:
                        top = rng.choice([c for c in EMOTION_CLASSES if c != CAT_TO_CLASS[answer]])
                    status = rng.choices(["ok", "uncertain", "no_face"], [0.8, 0.12, 0.08])[0]
                    va = (rng.uniform(-0.5, 0.6), -0.5 if answer == "tired" else rng.uniform(-0.2, 0.4)) if mtl else (None, None)
                    logger.log_self_report(ts, smoothed(status, top, 0.75 if status == "ok" else 0.42, *va), answer)
        logger.close()
