import csv
import threading
import time

import cv2
import pytest

from src.app import build_analyzer, build_logger, main
from src.capture import CameraSource, PacedVideoSource, StaticSource, iter_video_samples
from src.pipeline import LivePipeline, render_preview, run_video
from src.privacy import ConsentError
from src.smoothing import SmoothStatus

from .conftest import face_frame

pytestmark = pytest.mark.usefixtures("models_ready")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_iter_video_samples_respects_analysis_fps(test_video):
    times = [t for t, _ in iter_video_samples(test_video, analysis_fps=2.0)]
    assert len(times) == 16  # 8 s × 2 fps
    assert times[:4] == pytest.approx([0.0, 0.5, 1.0, 1.5])


def test_headless_video_writes_csv(tmp_config, test_video):
    analyzer = build_analyzer(tmp_config)
    logger = build_logger(tmp_config, analyzer)
    summary = run_video(test_video, analyzer, logger, analysis_fps=2.0)
    logger.close()

    rows = read_csv(summary.csv_path)
    assert len(rows) == summary.frames == 16
    statuses = [r["status"] for r in rows]
    assert statuses[:2] == ["warming_up", "warming_up"]  # tək kadr göstərilmir
    assert set(statuses[2:10]) == {"ok"}
    assert set(statuses[10:]) == {"no_face"}
    assert {r["top_emotion"] for r in rows if r["status"] == "ok"} == {"Happiness"}
    assert all(r["face_reason"] == "not_found" for r in rows[10:])
    assert all(r["operator_id"] == tmp_config.operator_id for r in rows)
    assert all(float(r["latency_ms"]) > 0 for r in rows)


def test_cli_headless_video(tmp_config, test_video, tmp_path, monkeypatch, capsys):
    import src.app as app_mod

    monkeypatch.setattr(app_mod, "load_config", lambda _path=None: tmp_config)
    assert main(["--headless", "--video", str(test_video)]) == 0
    out = capsys.readouterr().out
    assert "Emal olunan kadr" in out
    assert list((tmp_path / "logs").glob("*.csv"))


def test_cli_image_mode_writes_nothing(tmp_config, face_bgr, tmp_path, monkeypatch, capsys):
    import src.app as app_mod

    from .conftest import FACE_FIXTURE

    monkeypatch.setattr(app_mod, "load_config", lambda _path=None: tmp_config)
    assert main(["--image", str(FACE_FIXTURE)]) == 0
    out = capsys.readouterr().out
    assert "Xoşbəxtlik" in out
    assert not (tmp_path / "logs").exists()


def test_live_pipeline_with_static_source(tmp_config, face_bgr):
    analyzer = build_analyzer(tmp_config)
    logger = build_logger(tmp_config, analyzer)
    results = []
    pipe = LivePipeline(StaticSource([face_frame(face_bgr)]), analyzer, logger, 2.0,
                        on_result=lambda r, cpu: results.append((r, cpu)))
    pipe.start()
    deadline = time.monotonic() + 5
    while len(results) < 4 and time.monotonic() < deadline:
        time.sleep(0.05)
    pipe.stop()
    logger.close()
    assert not pipe.running
    assert len(results) >= 4
    snap = pipe.snapshot()
    assert snap.status is SmoothStatus.OK and snap.top_emotion == "Happiness"
    assert all(cpu >= 0 for _, cpu in results)
    # analysis_fps=2 → ~0.5 s aralıq; 30 FPS emal etmirik
    assert len(results) <= 5


def test_live_pipeline_finishes_with_video_source(tmp_config, test_video):
    analyzer = build_analyzer(tmp_config)
    logger = build_logger(tmp_config, analyzer)
    done = threading.Event()
    src = PacedVideoSource(test_video)
    src._frame_interval = 0.005  # testi sürətləndir
    pipe = LivePipeline(src, analyzer, logger, 2.0, on_finished=done.set)
    pipe.start()
    assert done.wait(10)
    src.close()
    logger.close()
    assert pipe.error is None


def test_camera_requires_consent(cfg, monkeypatch):
    def fail(*a, **k):
        pytest.fail("razılıq olmadan kamera açılmamalıdır")

    monkeypatch.setattr(cv2, "VideoCapture", fail)
    with pytest.raises(ConsentError):
        CameraSource(cfg.capture, consent=None)


def test_preview_is_in_memory_jpeg(face_bgr, cfg):
    from src.detector import FaceDetector

    frame = face_frame(face_bgr)
    det = FaceDetector(cfg.detector).detect(frame)
    jpg = render_preview(frame, det, width=320)
    assert jpg[:2] == b"\xff\xd8"
    img = cv2.imdecode(__import__("numpy").frombuffer(jpg, "uint8"), cv2.IMREAD_COLOR)
    assert img.shape[1] == 320
