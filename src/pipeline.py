"""Kadr → üz → model(lər) → smoothing → CSV axını.

Üz bir dəfə aşkarlanır və kəsilir; eyni kəsik bütün aktiv modellərə verilir ki,
müqayisə ədalətli olsun. Hər modelin öz smoothing-i və gecikmə statistikası var.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import psutil

from .capture import FrameSource, iter_video_samples
from .csv_logger import CsvLogger, ModelTag
from .detector import Detection, FaceDetector
from .recognizer import EmotionPrediction
from .smoothing import EmotionSmoother, SmoothedResult, SmoothStatus
from .variants import predict as predict_variant

log = logging.getLogger(__name__)

HISTORY_LEN = 90  # UI zaman xətti: 1.5 kadr/san-da ~60 saniyə
EMA_ALPHA = 0.3


@dataclass
class ModelRunner:
    key: str
    recognizer: Any  # predict(face_bgr) -> EmotionPrediction
    smoother: EmotionSmoother
    enabled: bool = True
    disabled_reason: str | None = None
    ms_ema: float | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))

    @property
    def tag(self) -> ModelTag:
        return ModelTag.of(self.recognizer)


@dataclass(frozen=True)
class ModelOutcome:
    key: str
    prediction: EmotionPrediction | None  # tək kadrın nəticəsi — UI-da yalnız “anlıq” kimi; CSV smoothed-dir
    smoothed: SmoothedResult
    model_ms: float | None
    tag: ModelTag


@dataclass(frozen=True)
class FrameResult:
    detection: Detection
    prediction: EmotionPrediction | None  # əsas modelin tək kadr nəticəsi
    smoothed: SmoothedResult  # əsas modelin smoothed nəticəsi
    detector_ms: float
    model_ms: float | None  # əsas model
    outcomes: dict[str, ModelOutcome] = field(default_factory=dict)  # bütün aktiv modellər

    @property
    def latency_ms(self) -> float:
        """Əsas model üçün detektor + model vaxtı."""
        return self.detector_ms + (self.model_ms or 0.0)

    @property
    def models_ms(self) -> float:
        return sum(o.model_ms or 0.0 for o in self.outcomes.values())


def _history_code(s: SmoothedResult) -> str:
    if s.status is SmoothStatus.NO_FACE:
        return "no_face"
    if s.status is SmoothStatus.WARMING_UP:
        return "warming_up"
    if s.status is SmoothStatus.UNCERTAIN:
        return "uncertain"
    return s.top_emotion or "uncertain"


class Analyzer:
    """Bir detektor + bir və ya bir neçə model. Əsas (primary) model UI-da böyük göstərilir."""

    def __init__(self, detector: FaceDetector, recognizer, smoother: EmotionSmoother) -> None:
        self.detector = detector
        self.lock = threading.RLock()
        self.runners: dict[str, ModelRunner] = {}
        self.add_model(recognizer, smoother)
        self.primary = recognizer.model_name
        self.missing: dict[str, str] = {}  # quraşdırılmamış / yüklənməmiş modellər → səbəb
        self.flip_tta = False  # config: model.flip_tta — bütün modellərə aiddir

    # --- modellərin idarəsi ----------------------------------------------------------

    @property
    def recognizer(self):
        return self.runners[self.primary].recognizer

    @property
    def smoother(self) -> EmotionSmoother:
        return self.runners[self.primary].smoother

    def add_model(self, recognizer, smoother: EmotionSmoother, enabled: bool = True) -> ModelRunner:
        runner = ModelRunner(recognizer.model_name, recognizer, smoother, enabled)
        with self.lock:
            self.runners[runner.key] = runner
        return runner

    def set_primary(self, key: str) -> None:
        with self.lock:
            if key not in self.runners:
                raise KeyError(f"Model yüklənməyib: {key}")
            self.primary = key
            self.runners[key].enabled = True
            self.runners[key].disabled_reason = None

    def set_enabled(self, key: str, enabled: bool, reason: str | None = None) -> None:
        with self.lock:
            runner = self.runners[key]
            if key == self.primary and not enabled:
                return  # əsas model həmişə işləyir
            runner.enabled = enabled
            runner.disabled_reason = None if enabled else reason
            if not enabled:
                runner.smoother.reset()
                runner.history.clear()

    def active(self) -> list[ModelRunner]:
        """Əsas model birinci, sonra digər aktiv modellər (əlavə olunma sırası ilə)."""
        with self.lock:
            others = [r for k, r in self.runners.items() if r.enabled and k != self.primary]
            return [self.runners[self.primary], *others]

    def reset(self) -> None:
        with self.lock:
            for r in self.runners.values():
                r.smoother.reset()
                r.history.clear()

    # --- emal ------------------------------------------------------------------------

    def process(self, frame_bgr: np.ndarray, t: float | None = None) -> FrameResult:
        with self.lock:
            start = time.perf_counter()
            detection = self.detector.detect(frame_bgr)
            detector_ms = (time.perf_counter() - start) * 1000.0
            # Üz keyfiyyət yoxlamasından keçməyibsə heç bir model işə salınmır.
            crop = self.detector.crop(frame_bgr, detection.face) if detection.ok else None

            outcomes: dict[str, ModelOutcome] = {}
            for runner in self.active():
                prediction = predict_variant(runner.recognizer, crop, self.flip_tta) if crop is not None else None
                smoothed = runner.smoother.add(prediction, t)
                ms = prediction.latency_ms if prediction else None
                if ms is not None:
                    runner.ms_ema = ms if runner.ms_ema is None else EMA_ALPHA * ms + (1 - EMA_ALPHA) * runner.ms_ema
                runner.history.append(_history_code(smoothed))
                outcomes[runner.key] = ModelOutcome(runner.key, prediction, smoothed, ms, runner.tag)

            p = outcomes[self.primary]
            return FrameResult(detection, p.prediction, p.smoothed, detector_ms, p.model_ms, outcomes)

    def snapshot_all(self, t: float | None = None) -> dict[str, SmoothedResult]:
        with self.lock:
            return {r.key: r.smoother.current(t) for r in self.active()}

    def tags(self) -> dict[str, ModelTag]:
        with self.lock:
            return {r.key: r.tag for r in self.runners.values()}

    # --- hesablama büdcəsi -------------------------------------------------------------

    def calibrate(self, repeats: int = 3) -> None:
        """Hər modelin vaxtını sintetik üz kəsiyində ölçür (büdcə qərarı üçün)."""
        crop = np.random.default_rng(0).integers(0, 255, (180, 150, 3), dtype=np.uint8)
        with self.lock:
            for runner in self.runners.values():
                runner.recognizer.predict(crop)  # isinmə
                times = [runner.recognizer.predict(crop).latency_ms for _ in range(repeats)]
                runner.ms_ema = float(np.median(times))

    def enforce_budget(self, limit_ms: float) -> list[str]:
        """Aktiv modellərin cəmi vaxtı limiti keçirsə, ən ağır qeyri-əsas modelləri söndürür.

        Söndürülən modellərin açarlarını qaytarır.
        """
        disabled = []
        with self.lock:
            while True:
                active = self.active()
                total = sum(r.ms_ema or 0.0 for r in active)
                if total <= limit_ms:
                    return disabled
                candidates = [r for r in active if r.key != self.primary]
                if not candidates:
                    return disabled
                heaviest = max(candidates, key=lambda r: r.ms_ema or 0.0)
                self.set_enabled(
                    heaviest.key, False,
                    f"Büdcə: cəmi {total:.0f} ms > {limit_ms:.0f} ms — ən ağır model söndürüldü",
                )
                disabled.append(heaviest.key)
                log.warning("Model söndürüldü (büdcə): %s", heaviest.key)


class CpuSampler:
    """Prosesin ümumi CPU-dan payı (%), son çağırışdan bəri."""

    def __init__(self) -> None:
        self._proc = psutil.Process()
        self._cores = psutil.cpu_count() or 1
        self._proc.cpu_percent(None)

    def sample(self) -> float:
        return self._proc.cpu_percent(None) / self._cores


def render_preview(frame_bgr: np.ndarray, detection: Detection | None, width: int = 360) -> bytes:
    """Önizləmə üçün kiçildilmiş, güzgü əksi JPEG — YALNIZ yaddaşda (imencode)."""
    h, w = frame_bgr.shape[:2]
    scale = width / w
    small = cv2.resize(frame_bgr, (width, round(h * scale)), interpolation=cv2.INTER_AREA)
    if detection is not None and detection.face is not None:
        f = detection.face
        color = (88, 209, 48) if detection.ok else (10, 159, 255)  # BGR: yaşıl / narıncı
        x0, y0 = int(f.x * scale), int(f.y * scale)
        x1, y1 = int((f.x + f.w) * scale), int((f.y + f.h) * scale)
        cv2.rectangle(small, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    small = cv2.flip(small, 1)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else b""


class LivePipeline:
    """Canlı mənbəni (kamera və ya real-vaxt video) analysis_fps tezliyi ilə emal edir."""

    def __init__(
        self,
        source: FrameSource,
        analyzer: Analyzer,
        logger: CsvLogger,
        analysis_fps: float,
        on_result: Callable[[FrameResult, float], None] | None = None,
        on_finished: Callable[[], None] | None = None,
        budget_ms: Callable[[], float | None] | None = None,
        log_hook: Callable[[FrameResult, datetime, float, np.ndarray], bool] | None = None,
    ) -> None:
        self.source = source
        self.analyzer = analyzer
        self.logger = logger
        self.interval = 1.0 / analysis_fps
        self._on_result = on_result
        self._on_finished = on_finished
        self._budget_ms = budget_ms or (lambda: None)
        # True qaytarırsa, sətrin yazılmasını özü həll edib (məs. nəzarətli test)
        self._log_hook = log_hook
        self._over_budget = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last: FrameResult | None = None
        self._cpu = CpuSampler()
        self.frames_processed = 0
        self.error: BaseException | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_result(self) -> FrameResult | None:
        return self._last

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="LivePipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=5.0)

    def snapshot(self) -> SmoothedResult:
        """Əsas modelin hazırkı smoothed nəticəsi."""
        with self._lock:
            return self.analyzer.smoother.current()

    def snapshot_all(self) -> dict[str, SmoothedResult]:
        """Bütün aktiv modellərin hazırkı nəticəsi (özünüqiymətləndirmə anı üçün)."""
        with self._lock:
            return self.analyzer.snapshot_all()

    def _check_budget(self) -> None:
        """Ardıcıl 5 analizdə büdcə aşılırsa, ən ağır əlavə model söndürülür."""
        limit = self._budget_ms()
        if limit is None:
            return
        total = sum(r.ms_ema or 0.0 for r in self.analyzer.active())
        self._over_budget = self._over_budget + 1 if total > limit else 0
        if self._over_budget >= 5:
            self.analyzer.enforce_budget(limit)
            self._over_budget = 0

    def preview_jpeg(self) -> bytes | None:
        frame = self.source.latest()
        if frame is None:
            return None
        last = self._last
        return render_preview(frame, last.detection if last else None)

    def _run(self) -> None:
        next_t = time.monotonic()
        try:
            while not self._stop.is_set():
                frame = self.source.latest()
                if frame is None:
                    if self.source.finished:
                        break
                    self._stop.wait(0.05)
                    continue
                ts = datetime.now()
                with self._lock:
                    result = self.analyzer.process(frame)
                cpu = self._cpu.sample()
                if not (self._log_hook and self._log_hook(result, ts, cpu, frame)):
                    log_outcomes(self.logger, ts, result, cpu)
                self._check_budget()
                self._last = result
                self.frames_processed += 1
                if self._on_result:
                    self._on_result(result, cpu)
                if self.source.finished:
                    break
                next_t += self.interval
                delay = next_t - time.monotonic()
                if delay < 0:  # geri qalmışıqsa, yığılmış kadrları tutmağa çalışmırıq
                    next_t = time.monotonic()
                    delay = 0
                self._stop.wait(delay)
        except BaseException as exc:  # noqa: BLE001 - UI-ya bildirmək üçün saxlanılır
            log.exception("Pipeline xətası")
            self.error = exc
        finally:
            if self._on_finished:
                self._on_finished()


def log_outcomes(logger: CsvLogger, ts: datetime, result: FrameResult, cpu: float | None) -> None:
    """Hər aktiv model üçün bir sətir — eyni timestamp ilə."""
    for o in result.outcomes.values():
        logger.log_sample(ts, o.smoothed, result.detection.status, result.detector_ms, o.model_ms, cpu, model=o.tag)


@dataclass
class VideoRunSummary:
    frames: int
    status_counts: dict[str, int]
    mean_latency_ms: float
    csv_path: Path | None


def run_video(
    path: str | Path,
    analyzer: Analyzer,
    logger: CsvLogger,
    analysis_fps: float,
    start_wall: datetime | None = None,
    on_result: Callable[[float, FrameResult], None] | None = None,
) -> VideoRunSummary:
    """Video faylını (real vaxtı gözləmədən) emal edir; CSV vaxtları video zaman oxu ilə yazılır."""
    start_wall = start_wall or datetime.now()
    cpu = CpuSampler()
    counts: dict[str, int] = {}
    latencies: list[float] = []
    csv_path = None
    for t_video, frame in iter_video_samples(path, analysis_fps):
        ts = start_wall + timedelta(seconds=t_video)
        result = analyzer.process(frame, t=t_video)
        log_outcomes(logger, ts, result, cpu.sample())
        csv_path = logger.path_for(ts)
        counts[result.smoothed.status.value] = counts.get(result.smoothed.status.value, 0) + 1
        latencies.append(result.latency_ms)
        if on_result:
            on_result(t_video, result)
    return VideoRunSummary(
        frames=len(latencies),
        status_counts=counts,
        mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        csv_path=csv_path,
    )
