"""Monitorinq sessiyası: razılıq → model yüklənməsi → canlı emal → dayandırma.

UI-dan (pywebview) asılı deyil — pəncərələr yalnız callback-lər vasitəsilə
xəbərdar edilir. Bu, məntiqi pəncərəsiz test etməyə imkan verir.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from .calibration import DONE, HOLD, CalibrationRun, raw_result
from .cameras import CameraInfo, list_cameras, load_choice, resolve, save_choice
from .capture import CameraSource, FrameSource, PacedVideoSource, SwitchableSource
from .config import EMOTION_CLASSES, AppConfig, ConfigError, validate_operator_id
from .dataset import DatasetWriter
from .csv_logger import CsvLogger
from .labels import EMOTION_LABELS, EXPRESSION_HINTS, FACE_REASON_LABELS, STATUS_LABELS
from .pipeline import Analyzer, FrameResult, LivePipeline
from .privacy import ConsentToken, RuntimeGuards, grant_consent, install_runtime_guards
from .selfreport import Prompt, SelfReportScheduler

log = logging.getLogger(__name__)


class Phase(str, Enum):
    CONSENT = "consent"
    LOADING = "loading"
    RUNNING = "running"
    STOPPED = "stopped"
    FINISHED = "finished"  # video faylı bitdi
    ERROR = "error"


def _default_analyzer(cfg: AppConfig) -> Analyzer:
    from .app import build_analyzer

    return build_analyzer(cfg)


class Session:
    def __init__(
        self,
        cfg: AppConfig,
        video: str | None = None,
        source_factory: Callable[[ConsentToken], FrameSource] | None = None,
        analyzer_factory: Callable[[AppConfig], Analyzer] = _default_analyzer,
        tick_interval: float = 0.5,
        camera_factory: Callable[[ConsentToken, CameraInfo | None], FrameSource] | None = None,
        camera_lister: Callable[[], list[CameraInfo]] = list_cameras,
    ) -> None:
        self.cfg = cfg
        self.video = video
        self._source_factory = source_factory or self._default_source
        self._camera_factory = camera_factory or self._open_camera
        self._list_cameras = camera_lister
        self._consent: ConsentToken | None = None
        self.cameras: list[CameraInfo] = []
        self.camera: CameraInfo | None = None  # seçilmiş (işləyərkən — açıq olan) kamera
        self._analyzer_factory = analyzer_factory
        self._tick_interval = tick_interval

        self.phase = Phase.CONSENT
        self.error: str | None = None
        self.preview_enabled = cfg.ui.show_preview

        self._lock = threading.RLock()
        # Sorğunun açılması ilə testin başlanması eyni anda baş verməsin (tick ↔ körpü axını)
        self._prompt_lock = threading.Lock()
        self._analyzer: Analyzer | None = None
        self._guards: RuntimeGuards | None = None
        self._logger: CsvLogger | None = None
        self._source: FrameSource | None = None
        self._pipeline: LivePipeline | None = None
        self._scheduler: SelfReportScheduler | None = None
        self._last_cpu: float | None = None
        self._ticker_stop = threading.Event()
        self._ticker: threading.Thread | None = None
        self.self_reports = 0
        self.compare_mode = cfg.compare.mode
        self._loading_models: set[str] = set()
        self._model_errors: dict[str, str] = {}
        self._test: CalibrationRun | None = None
        self._dataset: DatasetWriter | None = None
        self._tester_id: str | None = None
        self.test_error: str | None = None

        # UI callback-ləri (window.py təyin edir)
        self.on_change: Callable[[], None] = lambda: None
        self.on_prompt: Callable[[Prompt | None], None] = lambda p: None

        if self.source_kind == "camera":
            self.refresh_cameras()

    # --- mənbə -------------------------------------------------------------------

    @property
    def source_kind(self) -> str:
        return "video" if self.video else "camera"

    def _default_source(self, consent: ConsentToken) -> FrameSource:
        if self.video:
            return PacedVideoSource(self.video)
        self._consent = consent  # eyni razılıqla sessiya ərzində kamera dəyişdirilə bilər
        return SwitchableSource(self._camera_factory(consent, self.camera))

    def _open_camera(self, consent: ConsentToken, cam: CameraInfo | None) -> FrameSource:
        return CameraSource(self.cfg.capture, consent, cam.index if cam else None, cam.name if cam else None)

    # --- kamera seçimi -------------------------------------------------------------

    def refresh_cameras(self) -> list[CameraInfo]:
        """Qoşulmuş kameraları yenidən oxuyur (kameranı açmır)."""
        cams = self._list_cameras()
        with self._lock:
            self.cameras = cams
            current = self.camera
            if self.camera_active and current is not None:
                # Açıq kamera eyni qalır; yalnız siyahıdakı index-i yenilənir
                same = [c for c in cams if (current.path and c.path == current.path) or c.name == current.name]
                self.camera = same[0] if same else current
            else:
                choice = {"name": current.name, "path": current.path} if current else load_choice()
                self.camera = resolve(cams, choice, self.cfg.capture.camera_index)
        return cams

    def select_camera(self, index: int) -> tuple[bool, str | None]:
        """Kameranı seçir; işləyirsə dərhal keçir (yeni kamera açılmasa köhnəsi davam edir)."""
        cam = next((c for c in self.cameras if c.index == index), None)
        if cam is None:
            self.refresh_cameras()
            cam = next((c for c in self.cameras if c.index == index), None)
            if cam is None:
                return False, "Kamera tapılmadı — siyahını yeniləyin"
        if self.test_active:
            return False, "Nəzarətli test gedərkən kamera dəyişdirilmir"
        with self._lock:
            source, consent = self._source, self._consent
            live = self.phase is Phase.RUNNING and isinstance(source, SwitchableSource) and consent is not None
            if not live:
                self.camera = cam
        if live:
            if self.camera == cam:
                return True, None
            try:
                new = self._camera_factory(consent, cam)
            except Exception as exc:  # noqa: BLE001 - istifadəçiyə göstərilir
                log.warning("Kamera dəyişdirilmədi: %s", exc)
                return False, str(exc)
            if not source.switch(new):
                return False, "Kamera artıq bağlıdır"
            self.camera = cam
            log.info("Kamera dəyişdirildi: %s", cam.name)
        save_choice(cam)
        self._set_phase(self.phase)  # UI və indikator yenilənsin
        return True, None

    @property
    def camera_active(self) -> bool:
        return self.phase is Phase.RUNNING and self._source is not None

    # --- həyat dövrü -------------------------------------------------------------

    def accept_consent(self) -> None:
        with self._lock:
            if self.phase is not Phase.CONSENT:
                return
            consent = grant_consent("ui")
            self._set_phase(Phase.LOADING)
        threading.Thread(target=self._start, args=(consent,), name="SessionStart", daemon=True).start()

    def _start(self, consent: ConsentToken) -> None:
        try:
            if self._analyzer is None:
                self._analyzer = self._analyzer_factory(self.cfg)
                limit = self._budget_limit()
                if limit is not None and len(self._analyzer.runners) > 1:
                    # Kompüterin gücünə görə: sığmayan ağır modellər əvvəlcədən söndürülür
                    self._analyzer.calibrate()
                    self._analyzer.enforce_budget(limit)
            else:
                self._analyzer.reset()
            if self._guards is None:
                # Modellər yükləndikdən sonra — bundan sonra heç bir xarici bağlantı yoxdur.
                self._guards = install_runtime_guards(self.cfg.privacy.block_network)
            r = self._analyzer.recognizer
            self._logger = CsvLogger(self.cfg.logging.dir, self.cfg.operator_id, r.model_name, r.engine, r.device)
            self._scheduler = SelfReportScheduler(self.cfg.self_report)
            if self.source_kind == "camera":
                self.refresh_cameras()  # başlayandan bəri taxılıb-çıxarılan kameralar
            with self._lock:
                if self.phase is not Phase.LOADING:  # yüklənərkən dayandırıldı
                    self._logger.close()
                    return
                self._source = self._source_factory(consent)
                self._pipeline = LivePipeline(
                    self._source,
                    self._analyzer,
                    self._logger,
                    self.cfg.capture.analysis_fps,
                    on_result=self._on_result,
                    on_finished=self._on_pipeline_finished,
                    budget_ms=self._budget_limit,
                    log_hook=self._test_log_hook,
                )
                self._pipeline.start()
                self._set_phase(Phase.RUNNING)
            self._ticker_stop.clear()
            self._ticker = threading.Thread(target=self._tick_loop, name="SessionTicker", daemon=True)
            self._ticker.start()
        except Exception as exc:  # noqa: BLE001 - istifadəçiyə göstərilir
            log.exception("Sessiya başlamadı")
            self._teardown()
            self.error = str(exc)
            self._set_phase(Phase.ERROR)

    def stop(self) -> None:
        """Kameranı dərhal bağlayır. Açıq sorğu yazılmadan bağlanır."""
        with self._lock:
            if self.phase not in (Phase.RUNNING, Phase.LOADING):
                return
            # Faza əvvəlcə dəyişir ki, pipeline thread-inin on_finished-i kilidi gözləməsin.
            self.phase = Phase.STOPPED
        self._teardown()
        self._set_phase(Phase.STOPPED)

    def restart(self) -> None:
        """Dayandırılmış sessiyanı yenidən razılıq ekranına qaytarır."""
        with self._lock:
            if self.phase in (Phase.STOPPED, Phase.FINISHED, Phase.ERROR):
                self.error = None
                self._set_phase(Phase.CONSENT)

    def shutdown(self) -> None:
        with self._lock:
            if self.phase in (Phase.RUNNING, Phase.LOADING):
                self.phase = Phase.STOPPED
        self._teardown()
        self._close_dataset()
        if self._guards:
            self._guards.uninstall()
            self._guards = None

    def _teardown(self) -> None:
        self._ticker_stop.set()
        if self._test is not None and self.test_active:
            self._test.cancel()  # kamera dayandı — test də bitir
        self._close_dataset()
        if self._scheduler and self._scheduler.cancel():
            self.on_prompt(None)
        if self._pipeline:
            self._pipeline.stop()
        if self._source:
            self._source.close()
        if self._logger:
            self._logger.close()
        self._pipeline = None
        self._source = None

    def _on_pipeline_finished(self) -> None:
        if self.phase is not Phase.RUNNING:  # istifadəçi dayandırıb — teardown artıq gedir
            return
        with self._lock:
            if self.phase is not Phase.RUNNING:
                return
            pipeline = self._pipeline
            failed = pipeline is not None and pipeline.error is not None
            self._ticker_stop.set()
            if self._scheduler and self._scheduler.cancel():
                self.on_prompt(None)
            if self._source:
                self._source.close()
            if self._logger:
                self._logger.close()
            self._source = None
            if failed:
                self.error = str(pipeline.error)
                self._set_phase(Phase.ERROR)
            else:
                self._set_phase(Phase.FINISHED if self.video else Phase.STOPPED)

    def _set_phase(self, phase: Phase) -> None:
        self.phase = phase
        try:
            self.on_change()
        except Exception:  # noqa: BLE001
            log.exception("on_change callback xətası")

    def _on_result(self, result: FrameResult, cpu: float) -> None:
        self._last_cpu = cpu

    # --- özünüqiymətləndirmə ------------------------------------------------------

    def _tick_loop(self) -> None:
        while not self._ticker_stop.wait(self._tick_interval):
            self.tick()

    def tick(self) -> None:
        sched, pipeline, logger = self._scheduler, self._pipeline, self._logger
        if self.phase is not Phase.RUNNING or not sched or not pipeline or not logger:
            return
        expired = sched.expire()
        if expired:
            prompt, code = expired
            self._log_report(prompt, code)
            self.on_prompt(None)
        with self._prompt_lock:
            if self._test is not None:
                return  # nəzarətli test və onun nəticə ekranı zamanı sorğu göstərilmir
            if sched.due():
                self.on_prompt(sched.open(pipeline.snapshot_all()))

    def open_prompt_now(self) -> Prompt | None:
        """Sorğunu dərhal açır (demo/test üçün)."""
        sched, pipeline = self._scheduler, self._pipeline
        with self._prompt_lock:
            if self.phase is not Phase.RUNNING or not sched or not pipeline or sched.active or self._test is not None:
                return None
            prompt = sched.open(pipeline.snapshot_all())
            self.on_prompt(prompt)
            return prompt

    def _log_report(self, prompt: Prompt, code: str) -> None:
        """Sorğu anındakı HƏR modelin nəticəsi ayrıca sətir kimi yazılır."""
        logger, analyzer = self._logger, self._analyzer
        if not logger or not analyzer:
            return
        tags = analyzer.tags()
        for key, snap in prompt.snapshot.items():
            logger.log_self_report(prompt.shown_at, snap, code, model=tags.get(key))
        self.self_reports += 1

    def answer(self, prompt_id: int, code: str) -> bool:
        sched = self._scheduler
        if not sched or not self._logger:
            return False
        got = sched.answer(prompt_id, code)
        if got is None:
            return False
        prompt, code = got
        self._log_report(prompt, code)
        self.on_prompt(None)
        return True

    # --- nəzarətli test --------------------------------------------------------------

    @property
    def test_active(self) -> bool:
        run = self._test
        return run is not None and run.position().phase != DONE

    def start_test(self, seed: int | None = None, tester_id: str | None = None, save_frames: bool = False) -> bool:
        """Göstərişli ifadə testini başladır. Açıq sorğu yazılmadan bağlanır.

        tester_id — test olunan şəxsin anonim ID-si (ad yox). save_frames=True yalnız testerin
        həmin test üçün verdiyi razılıqdır; config-də research.save_test_frames söndürülübsə nəzərə alınmır.
        """
        self.test_error = None
        if self.phase is not Phase.RUNNING or self.test_active:
            return False
        try:
            tester = validate_operator_id(tester_id) if tester_id else self.cfg.operator_id
        except ConfigError:
            self.test_error = "Tester ID anonim olmalıdır: 2–32 simvol, ən azı bir rəqəm (məs. T-01)"
            return False
        self._close_dataset()
        writer = None
        if save_frames and self.cfg.research.save_test_frames:
            try:
                writer = DatasetWriter(self.cfg.research, tester, self.cfg.operator_id, self.cfg.test)
            except OSError as exc:
                self.test_error = f"Dataset qovluğu yaradılmadı: {exc}"
                return False
        # Tester üzünün çərçivədə olduğunu görsün — önizləmə test müddətində açıqdır
        self._preview_before_test = self.preview_enabled
        self.preview_enabled = True
        self._dataset = writer
        self._tester_id = tester
        with self._prompt_lock:
            self._test = CalibrationRun(self.cfg.test, seed=seed)
            if self._scheduler and self._scheduler.cancel():
                self.on_prompt(None)
        log.info("Nəzarətli test başladı: %d addım, tester %s, şəkil: %s",
                 len(self._test.steps), tester, writer.session_id if writer else "saxlanmır")
        return True

    def _close_dataset(self) -> None:
        writer = getattr(self, "_dataset", None)
        if writer is not None:
            writer.close()

    def cancel_test(self) -> bool:
        if not self.test_active:
            return False
        self._test.cancel()
        self._close_dataset()
        return True

    def dismiss_test(self) -> bool:
        """Nəticə ekranını bağlayır (panelə qayıdış)."""
        if self.test_active:
            return False
        if self._test is not None:
            self.preview_enabled = getattr(self, "_preview_before_test", self.preview_enabled)
        self._close_dataset()
        self._dataset = None
        with self._prompt_lock:
            if self._test is not None and self._scheduler:
                self._scheduler.postpone()  # panelə qayıdan kimi sorğu açılmasın — tam interval sonra
            self._test = None
        return True

    def _test_log_hook(self, result: FrameResult, ts: datetime, cpu: float, frame=None) -> bool:
        run, logger = self._test, self._logger
        if run is None or logger is None:
            return False
        pos = run.position()
        if pos.phase == DONE:
            self._close_dataset()
            return False  # test bitib — adi yazılışa qayıt
        if pos.phase != HOLD:
            return True  # hazırlıq/rahatlama kadrları heç yerə yazılmır
        expected = pos.step.expression
        threshold = self.cfg.smoothing.uncertain_threshold
        tester = getattr(self, "_tester_id", None)
        for key, o in result.outcomes.items():
            raw = raw_result(o.prediction, threshold)
            logger.log_test(ts, raw, result.detection.status, result.detector_ms, o.model_ms, cpu, expected,
                            model=o.tag, tester_id=tester)
            run.record(key, expected, raw)
        writer = getattr(self, "_dataset", None)
        if writer is not None and frame is not None:
            # Razılıqla: üz ətrafında kəsik, ifadə etiketi ilə (oflayn qiymətləndirmə üçün)
            writer.add(frame, result.detection, expected, pos.step.index, pos.step.round,
                       pos.phase_seconds - pos.seconds_left, ts)
        return True

    def _test_state(self) -> dict[str, Any] | None:
        from .model_zoo import SPECS

        run = self._test
        if run is None:
            return None
        pos = run.position()
        done = pos.phase == DONE
        step = pos.step
        data: dict[str, Any] = {
            "active": not done,
            "done": done,
            "cancelled": run.cancelled,
            "phase": pos.phase,
            "seconds_left": round(pos.seconds_left, 1),
            "phase_seconds": pos.phase_seconds,
            "step": step.index if step else len(run.steps),
            "total": len(run.steps),
            "round": step.round if step else self.cfg.test.rounds,
            "rounds": self.cfg.test.rounds,
            "expected": step.expression if step else None,
            "label": EMOTION_LABELS[step.expression] if step else None,
            "hint": EXPRESSION_HINTS[step.expression] if step else None,
            "steps": [s.expression for s in run.steps],
            "summary": None,
            "tester_id": getattr(self, "_tester_id", None),
            "saved_frames": self._dataset.count if getattr(self, "_dataset", None) else None,
            "dataset_session": self._dataset.session_id if getattr(self, "_dataset", None) else None,
        }
        if done:
            # Nəticə yalnız test bitəndə — tester ifadəsini modelə "uyğunlaşdırmasın"
            rows = []
            for key, s in run.summary().items():
                spec = SPECS.get(key)
                rows.append({"key": key, "label": spec.label if spec else key,
                             "dataset": spec.dataset if spec else "", **s})
            rows.sort(key=lambda r: (r["balanced"] is None, -(r["balanced"] or 0)))
            data["summary"] = rows
            data["expressions"] = [{"key": e, "label": EMOTION_LABELS[e]} for e in self.cfg.test.expressions]
        return data

    # --- modellərin idarəsi (işləyərkən) -------------------------------------------

    def _budget_limit(self) -> float | None:
        if self.compare_mode != "auto":
            return None
        return self.cfg.compare.budget_ratio * 1000.0 / self.cfg.capture.analysis_fps

    def model_keys(self) -> list[str]:
        """UI-da göstərilən modellər: əsas, müqayisə siyahısı, sonra digər quraşdırılmışlar."""
        from .model_zoo import installed_models

        primary = self._analyzer.primary if self._analyzer else self.cfg.model.name
        return list(dict.fromkeys([primary, self.cfg.model.name, *self.cfg.compare.models, *installed_models()]))

    def _ensure_loaded(self, key: str, then: Callable[[], None]) -> bool:
        """Model yüklənməyibsə, arxa planda yükləyir və sonra `then`-i çağırır."""
        analyzer = self._analyzer
        if analyzer is None:
            return False
        if key in analyzer.runners:
            then()
            return True
        if key in self._loading_models:
            return True

        def load() -> None:
            from .model_zoo import make_recognizer
            from .smoothing import EmotionSmoother

            try:
                # Fayl diskdədir — şəbəkə qoruyucusu aktiv qalır, endirmə olmur.
                analyzer.add_model(make_recognizer(key), EmotionSmoother(self.cfg.smoothing), enabled=False)
                analyzer.missing.pop(key, None)
                self._model_errors.pop(key, None)
                then()
            except Exception as exc:  # noqa: BLE001 - UI-da göstərilir
                log.warning("%s yüklənmədi: %s", key, exc)
                self._model_errors[key] = str(exc)
            finally:
                self._loading_models.discard(key)
                self.on_change()

        self._loading_models.add(key)
        threading.Thread(target=load, name=f"Load-{key}", daemon=True).start()
        return True

    def set_primary(self, key: str) -> bool:
        """Əsas modeli dəyişir (kamera dayanmadan)."""
        analyzer = self._analyzer
        if analyzer is None:
            return False

        def apply() -> None:
            previous = analyzer.primary
            analyzer.set_primary(key)
            if self.compare_mode == "off" and previous != key:
                analyzer.set_enabled(previous, False, "Müqayisə söndürülüb")

        return self._ensure_loaded(key, apply)

    def set_model_enabled(self, key: str, enabled: bool) -> bool:
        analyzer = self._analyzer
        if analyzer is None:
            return False
        if not enabled:
            if key in analyzer.runners:
                analyzer.set_enabled(key, False, "İstifadəçi söndürüb")
            return True
        if self.compare_mode == "off":
            self.compare_mode = "on"

        def apply() -> None:
            analyzer.set_enabled(key, True)

        return self._ensure_loaded(key, apply)

    def set_compare_mode(self, mode: str) -> bool:
        if mode not in ("off", "auto", "on"):
            return False
        self.compare_mode = mode
        analyzer = self._analyzer
        if analyzer is None:
            return True
        if mode == "off":
            for key in list(analyzer.runners):
                if key != analyzer.primary:
                    analyzer.set_enabled(key, False, "Müqayisə söndürülüb")
            return True
        for key in self.cfg.compare.models:
            if key == analyzer.primary:
                continue
            if key in analyzer.runners or key not in analyzer.missing:
                self._ensure_loaded(key, lambda k=key: analyzer.set_enabled(k, True))
        limit = self._budget_limit()
        if limit is not None:
            analyzer.enforce_budget(limit)
        return True

    def prompt_state(self) -> dict[str, Any] | None:
        sched = self._scheduler
        prompt = sched.active if sched else None
        if prompt is None:
            return None
        return {
            "id": prompt.id,
            "seconds_left": round(sched.seconds_left() or 0.0, 2),
            "timeout": self.cfg.self_report.timeout_seconds,
        }

    # --- UI üçün vəziyyət ----------------------------------------------------------

    def preview_data_url(self) -> str | None:
        pipeline = self._pipeline
        if not self.preview_enabled or self.phase is not Phase.RUNNING or pipeline is None:
            return None
        jpg = pipeline.preview_jpeg()
        if not jpg:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")

    def state(self) -> dict[str, Any]:
        pipeline, sched, logger = self._pipeline, self._scheduler, self._logger
        last = pipeline.last_result if pipeline else None
        data: dict[str, Any] = {
            "phase": self.phase.value,
            "camera_active": self.camera_active,
            "error": self.error,
            "frames": pipeline.frames_processed if pipeline else 0,
            "records": logger.rows_written if logger else 0,
            "self_reports": self.self_reports,
            "csv_path": str(logger.path_for(datetime.now())) if logger else None,
            "next_prompt_s": round(sched.seconds_until_next()) if sched and self.cfg.self_report.enabled else None,
            "prompt_open": bool(sched and sched.active),
            "preview_enabled": self.preview_enabled,
            "face": None,
            "result": None,
            "latency_ms": None,
            "cpu": None if self._last_cpu is None else round(self._last_cpu, 1),
            "compare_mode": self.compare_mode,
            "primary": self._analyzer.primary if self._analyzer else self.cfg.model.name,
            "models": self._models_state(last),
            "budget": None,
            "agreement": None,
            "test": self._test_state() if self.phase is Phase.RUNNING else None,
            "cameras": [{"index": c.index, "name": c.name} for c in self.cameras],
            "camera": {"index": self.camera.index, "name": self.camera.name} if self.camera else None,
        }
        if self._analyzer is not None:
            used = sum(r.ms_ema or 0.0 for r in self._analyzer.active())
            data["budget"] = {"limit_ms": self._budget_limit(), "used_ms": round(used, 1),
                              "interval_ms": round(1000.0 / self.cfg.capture.analysis_fps)}
        if last is None or self.phase is not Phase.RUNNING:
            return data

        reason = last.detection.status.value
        data["face"] = {"status": reason, "label": FACE_REASON_LABELS[reason], "ok": last.detection.ok}
        data["latency_ms"] = round(last.detector_ms + last.models_ms, 1)
        data["detector_ms"] = round(last.detector_ms, 1)
        data["result"] = self._result_dict(last.smoothed)
        data["agreement"] = self._agreement(last)
        return data

    def _result_dict(self, s, classes=None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": s.status.value,
            "status_label": STATUS_LABELS[s.status.value],
            "n_frames": s.n_frames,
            "window_s": self.cfg.smoothing.window_seconds,
        }
        if self.cfg.ui.show_prediction and s.has_estimate:
            result.update(
                top=s.top_emotion,
                top_label=EMOTION_LABELS[s.top_emotion],
                top_prob=round(s.top_prob, 4),
                probs=[
                    {"key": c, "label": EMOTION_LABELS[c],
                     "p": None if classes is not None and c not in classes else round(float(s.probs[i]), 4)}
                    for i, c in enumerate(EMOTION_CLASSES)
                ],
                valence=None if s.valence is None else round(s.valence, 3),
                arousal=None if s.arousal is None else round(s.arousal, 3),
            )
        return result

    def _models_state(self, last: FrameResult | None) -> list[dict[str, Any]]:
        from .model_zoo import SPECS, is_installed

        analyzer = self._analyzer
        runners = analyzer.runners if analyzer else {}
        primary = analyzer.primary if analyzer else self.cfg.model.name
        outcomes = last.outcomes if last is not None and self.phase is Phase.RUNNING else {}
        out = []
        for key in self.model_keys():
            spec = SPECS[key]
            runner = runners.get(key)
            enabled = bool(runner and runner.enabled)
            reason = (runner.disabled_reason if runner else None) or self._model_errors.get(key) \
                or (analyzer.missing.get(key) if analyzer else None)
            installed = is_installed(key)
            if not installed:
                reason = "quraşdırılmayıb — scripts/download_models.py --extra"
            history = list(runner.history)[-60:] if runner and enabled else []
            if not self.cfg.ui.show_prediction:
                history = [h if h in ("no_face", "warming_up") else "ok" for h in history]
            o = outcomes.get(key) if enabled else None
            classes = spec.classes if len(spec.classes) < len(EMOTION_CLASSES) else None
            out.append({
                "key": key,
                "label": spec.label,
                "dataset": spec.dataset,
                "family": spec.family,
                "has_contempt": classes is None,
                "has_va": spec.has_va,
                "installed": installed,
                "loaded": runner is not None,
                "loading": key in self._loading_models,
                "enabled": enabled,
                "primary": key == primary,
                "reason": None if enabled else reason,
                "ms": None if not (runner and runner.ms_ema) else round(runner.ms_ema, 1),
                "result": self._result_dict(o.smoothed, classes) if o else None,
                "history": history,
            })
        return out

    def _agreement(self, last: FrameResult) -> dict[str, Any] | None:
        """Neçə model eyni (müəyyən) emosiyanı deyir — UI-dakı "Razılaşma 3/4"."""
        if len(last.outcomes) < 2 or not self.cfg.ui.show_prediction:
            return None
        tops = [o.smoothed.top_emotion for o in last.outcomes.values() if o.smoothed.status.value == "ok"]
        if not tops:
            return {"agree": 0, "total": len(last.outcomes), "top": None, "top_label": None}
        best = max(set(tops), key=tops.count)
        return {"agree": tops.count(best), "total": len(last.outcomes), "top": best, "top_label": EMOTION_LABELS[best]}


def wait_for(predicate: Callable[[], bool], timeout: float = 10.0, step: float = 0.02) -> bool:
    """Testlər və UI üçün kiçik köməkçi."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()
