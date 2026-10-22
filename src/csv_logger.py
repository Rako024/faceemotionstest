"""Rəqəmsal nəticələrin CSV-yə yazılması. Şəkil, video və ad heç vaxt yazılmır.

Fayl: <logging.dir>/YYYY-MM-DD_<operator_id>.csv (gün dəyişəndə yeni fayl).
Hər analiz olunan kadr üçün HƏR MODEL üzrə bir "sample" sətri, hər
özünüqiymətləndirmə üçün HƏR MODEL üzrə bir "self_report" sətri yazılır.
Eyni kadrın sətirləri eyni timestamp-ı paylaşır.
"""

from __future__ import annotations

import csv
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO

from .config import EMOTION_CLASSES
from .detector import FaceStatus
from .smoothing import SmoothedResult, SmoothStatus

PROB_COLUMNS = [f"p_{name.lower()}" for name in EMOTION_CLASSES]

COLUMNS = [
    "timestamp",
    "operator_id",
    "model_name",
    "engine",
    "device",
    "record_type",
    "status",
    "face_detected",
    "face_reason",
    "top_emotion",
    "top_prob",
    *PROB_COLUMNS,
    "valence",
    "arousal",
    "n_frames",
    "self_report",
    "latency_ms",
    "detector_ms",
    "model_ms",
    "cpu_percent",
    "expected",  # yalnız record_type=test: tapşırılan ifadə
]
OPTIONAL_COLUMNS = {"expected"}  # köhnə fayllarda olmaya bilər

SAMPLE = "sample"
SELF_REPORT = "self_report"
TEST = "test"
UNCERTAIN_LABEL = "uncertain"


@dataclass(frozen=True)
class ModelTag:
    """Sətrin aid olduğu model. classes=None → 8 sinifin hamısı dəstəklənir."""

    name: str
    engine: str
    device: str
    classes: frozenset[str] | None = None

    @classmethod
    def of(cls, recognizer) -> "ModelTag":
        spec = getattr(recognizer, "spec", None)
        classes = spec.classes if spec is not None and len(spec.classes) < len(EMOTION_CLASSES) else None
        return cls(recognizer.model_name, recognizer.engine, recognizer.device, classes)


def _fmt(value: float | None, digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def timestamp(dt: datetime | None = None) -> str:
    return (dt or datetime.now()).astimezone().isoformat(timespec="milliseconds")


def build_row(
    *,
    ts: datetime,
    operator_id: str,
    model_name: str,
    engine: str,
    device: str,
    record_type: str,
    smoothed: SmoothedResult,
    face_reason: FaceStatus | None,
    self_report: str = "",
    latency_ms: float | None = None,
    detector_ms: float | None = None,
    model_ms: float | None = None,
    cpu_percent: float | None = None,
    classes: frozenset[str] | None = None,
    expected: str = "",
) -> dict[str, str]:
    if smoothed.status is SmoothStatus.OK:
        top_emotion = smoothed.top_emotion or ""
    elif smoothed.status is SmoothStatus.UNCERTAIN:
        top_emotion = UNCERTAIN_LABEL
    else:
        top_emotion = ""

    row = {
        "timestamp": timestamp(ts),
        "operator_id": operator_id,
        "model_name": model_name,
        "engine": engine,
        "device": device,
        "record_type": record_type,
        "status": smoothed.status.value,
        "face_detected": "0" if smoothed.status is SmoothStatus.NO_FACE else "1",
        "face_reason": face_reason.value if face_reason else "",
        "top_emotion": top_emotion,
        "top_prob": _fmt(smoothed.top_prob, 4),
        "valence": _fmt(smoothed.valence, 4),
        "arousal": _fmt(smoothed.arousal, 4),
        "n_frames": str(smoothed.n_frames),
        "self_report": self_report,
        "latency_ms": _fmt(latency_ms, 1),
        "detector_ms": _fmt(detector_ms, 1),
        "model_ms": _fmt(model_ms, 1),
        "cpu_percent": _fmt(cpu_percent, 1),
        "expected": expected,
    }
    probs = smoothed.probs
    for i, col in enumerate(PROB_COLUMNS):
        supported = classes is None or EMOTION_CLASSES[i] in classes
        row[col] = "" if probs is None or not supported else f"{float(probs[i]):.4f}"
    return row


class CsvLogger:
    def __init__(self, log_dir: Path, operator_id: str, model_name: str = "", engine: str = "", device: str = "") -> None:
        self.log_dir = Path(log_dir)
        self.operator_id = operator_id
        # Default model — sətir üçün model=ModelTag verilməyəndə istifadə olunur
        self.default_model = ModelTag(model_name, engine, device)
        self._lock = threading.Lock()
        self._fh: IO[str] | None = None
        self._writer: csv.DictWriter | None = None
        self._date: str | None = None
        self.rows_written = 0

    def path_for(self, ts: datetime) -> Path:
        """Günün faylı. Köhnə sxemli fayl varsa, onu pozmamaq üçün _2, _3… yaradılır."""
        base = f"{ts.strftime('%Y-%m-%d')}_{self.operator_id}"
        expected = ",".join(COLUMNS)
        n = 1
        while True:
            path = self.log_dir / (f"{base}.csv" if n == 1 else f"{base}_{n}.csv")
            if not path.exists() or path.stat().st_size == 0:
                return path
            with path.open(encoding="utf-8") as fh:
                if fh.readline().strip() == expected:
                    return path
            n += 1

    def log_sample(
        self,
        ts: datetime,
        smoothed: SmoothedResult,
        face_reason: FaceStatus,
        detector_ms: float | None,
        model_ms: float | None,
        cpu_percent: float | None,
        model: ModelTag | None = None,
    ) -> None:
        latency = None
        if detector_ms is not None:
            latency = detector_ms + (model_ms or 0.0)
        self._write(
            ts,
            SAMPLE,
            smoothed,
            model,
            face_reason=face_reason,
            latency_ms=latency,
            detector_ms=detector_ms,
            model_ms=model_ms,
            cpu_percent=cpu_percent,
        )

    def log_test(
        self,
        ts: datetime,
        result: SmoothedResult,
        face_reason: FaceStatus,
        detector_ms: float | None,
        model_ms: float | None,
        cpu_percent: float | None,
        expected: str,
        model: ModelTag | None = None,
        tester_id: str | None = None,
    ) -> None:
        """Nəzarətli test kadrı: result — tək kadrın nəticəsi (smoothing yox), expected — tapşırılan ifadə.

        tester_id verilərsə operator_id sütununa yazılır (test olunan şəxsin anonim ID-si).
        """
        latency = None if detector_ms is None else detector_ms + (model_ms or 0.0)
        self._write(
            ts, TEST, result, model, face_reason=face_reason, latency_ms=latency, detector_ms=detector_ms,
            model_ms=model_ms, cpu_percent=cpu_percent, expected=expected, operator_id=tester_id,
        )

    def log_self_report(self, ts: datetime, smoothed: SmoothedResult, answer: str, model: ModelTag | None = None) -> None:
        """ts — sorğunun göstərildiyi an; smoothed — həmin andakı model nəticəsi."""
        self._write(ts, SELF_REPORT, smoothed, model, face_reason=None, self_report=answer)

    def _write(self, ts: datetime, record_type: str, smoothed: SmoothedResult, model: ModelTag | None, **extra) -> None:
        m = model or self.default_model
        row = build_row(
            ts=ts,
            operator_id=extra.pop("operator_id", None) or self.operator_id,
            model_name=m.name,
            engine=m.engine,
            device=m.device,
            record_type=record_type,
            smoothed=smoothed,
            classes=m.classes,
            **extra,
        )
        with self._lock:
            self._ensure_file(ts)
            assert self._writer is not None and self._fh is not None
            self._writer.writerow(row)
            self._fh.flush()
            self.rows_written += 1

    def _ensure_file(self, ts: datetime) -> None:
        date = ts.strftime("%Y-%m-%d")
        if self._writer is not None and date == self._date:
            return
        self._close_file()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(ts)
        new_file = not path.exists() or path.stat().st_size == 0
        self._fh = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=COLUMNS)
        if new_file:
            self._writer.writeheader()
        self._date = date

    def _close_file(self) -> None:
        if self._fh is not None:
            self._fh.close()
        self._fh = None
        self._writer = None

    def close(self) -> None:
        with self._lock:
            self._close_file()
