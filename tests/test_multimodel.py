"""Bir neçə modelin eyni kadrda paralel işlədilməsi, büdcə, canlı keçid."""

import csv
from collections import defaultdict
from dataclasses import replace

import numpy as np
import pytest

from src.app import build_analyzer, build_logger
from src.capture import StaticSource
from src.config import EMOTION_CLASSES
from src.model_zoo import is_installed
from src.pipeline import run_video
from src.session import Phase, Session, wait_for

from .conftest import face_frame

pytestmark = pytest.mark.usefixtures("models_ready")


def _need(*keys):
    missing = [k for k in keys if not is_installed(k)]
    if missing:
        pytest.skip(f"quraşdırılmayıb: {missing}")


@pytest.fixture
def cmp_config(make_config, tmp_path):
    return make_config(
        logging__dir=str(tmp_path / "logs"),
        model__name="enet_b0_8_best_afew",
        compare__mode="on",
        compare__models=["enet_b0_8_best_afew", "deepface_emotion", "vit_fer2013", "kaggle_fer_cnn"],
    )


def read_rows(cfg):
    rows = []
    for p in sorted(cfg.logging.dir.glob("*.csv")):
        with p.open(encoding="utf-8", newline="") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def test_analyzer_runs_all_installed_models(cmp_config, face_bgr):
    _need("deepface_emotion", "vit_fer2013")
    analyzer = build_analyzer(cmp_config)
    assert analyzer.primary == "enet_b0_8_best_afew"
    assert {"enet_b0_8_best_afew", "deepface_emotion", "vit_fer2013"} <= set(analyzer.runners)
    if not is_installed("kaggle_fer_cnn"):
        assert "kaggle_fer_cnn" in analyzer.missing

    frame = face_frame(face_bgr)
    for t in range(4):
        result = analyzer.process(frame, t=float(t))
    assert set(result.outcomes) == set(analyzer.runners)
    assert result.smoothed is result.outcomes["enet_b0_8_best_afew"].smoothed
    for key, o in result.outcomes.items():
        assert o.smoothed.top_emotion == "Happiness", key
        assert o.model_ms > 0
        assert list(analyzer.runners[key].history)[-1] == "Happiness"
    fer = result.outcomes["deepface_emotion"].smoothed
    assert fer.probs[EMOTION_CLASSES.index("Contempt")] == 0.0
    assert result.models_ms == pytest.approx(sum(o.model_ms for o in result.outcomes.values()))


def test_video_writes_one_row_per_model_per_frame(cmp_config, test_video):
    _need("deepface_emotion")
    analyzer = build_analyzer(replace(cmp_config, compare=replace(cmp_config.compare,
                                                                   models=["enet_b0_8_best_afew", "deepface_emotion"])))
    logger = build_logger(cmp_config, analyzer)
    summary = run_video(test_video, analyzer, logger, analysis_fps=2.0)
    logger.close()
    rows = read_rows(cmp_config)
    assert len(rows) == 2 * summary.frames
    by_ts = defaultdict(set)
    for r in rows:
        by_ts[r["timestamp"]].add(r["model_name"])
    assert all(models == {"enet_b0_8_best_afew", "deepface_emotion"} for models in by_ts.values())
    fer_ok = [r for r in rows if r["model_name"] == "deepface_emotion" and r["status"] == "ok"]
    assert fer_ok and all(r["p_contempt"] == "" for r in fer_ok), "FER modelində Contempt sütunu boş olmalıdır"
    enet_ok = [r for r in rows if r["model_name"] == "enet_b0_8_best_afew" and r["status"] == "ok"]
    assert all(r["p_contempt"] != "" for r in enet_ok)


def test_budget_disables_heaviest_extra_model(cmp_config):
    _need("deepface_emotion", "vit_fer2013")
    analyzer = build_analyzer(cmp_config)
    analyzer.runners["enet_b0_8_best_afew"].ms_ema = 400.0  # əsas model ağır olsa da söndürülmür
    analyzer.runners["deepface_emotion"].ms_ema = 1.0
    analyzer.runners["vit_fer2013"].ms_ema = 150.0
    disabled = analyzer.enforce_budget(limit_ms=300.0)
    assert disabled == ["vit_fer2013"] or disabled[0] == "vit_fer2013"
    assert analyzer.runners["enet_b0_8_best_afew"].enabled
    assert not analyzer.runners["vit_fer2013"].enabled
    assert "Büdcə" in analyzer.runners["vit_fer2013"].disabled_reason


def test_calibrate_measures_each_model(cmp_config):
    _need("deepface_emotion")
    analyzer = build_analyzer(cmp_config)
    analyzer.calibrate(repeats=2)
    assert all(r.ms_ema and r.ms_ema > 0 for r in analyzer.runners.values())


@pytest.fixture
def session_factory(face_bgr):
    sessions = []

    def make(cfg):
        s = Session(cfg, source_factory=lambda consent: StaticSource([face_frame(face_bgr)]), tick_interval=0.05)
        sessions.append(s)
        return s

    yield make
    for s in sessions:
        s.shutdown()


def test_session_state_lists_models(cmp_config, session_factory):
    _need("deepface_emotion")
    s = session_factory(cmp_config)
    s.accept_consent()
    assert wait_for(lambda: s.phase is Phase.RUNNING, timeout=20)
    assert wait_for(lambda: (s.state()["agreement"] or {}).get("agree", 0) >= 2, timeout=15)
    st = s.state()
    models = {m["key"]: m for m in st["models"]}
    assert models["enet_b0_8_best_afew"]["primary"] is True
    assert models["deepface_emotion"]["enabled"] is True
    assert models["deepface_emotion"]["has_contempt"] is False
    assert models["deepface_emotion"]["result"]["top"] == "Happiness"
    contempt = next(p for p in models["deepface_emotion"]["result"]["probs"] if p["key"] == "Contempt")
    assert contempt["p"] is None
    assert models["deepface_emotion"]["history"][-1] == "Happiness"
    if not is_installed("kaggle_fer_cnn"):
        assert models["kaggle_fer_cnn"]["installed"] is False and "download_models" in models["kaggle_fer_cnn"]["reason"]
    assert st["budget"]["used_ms"] > 0
    assert st["agreement"]["top"] == "Happiness"


def test_self_report_logs_every_active_model(cmp_config, session_factory):
    _need("deepface_emotion")
    cfg = replace(cmp_config, compare=replace(cmp_config.compare, models=["enet_b0_8_best_afew", "deepface_emotion"]))
    s = session_factory(cfg)
    s.accept_consent()
    assert wait_for(lambda: s.state()["frames"] >= 3, timeout=20)
    prompt = s.open_prompt_now()
    assert set(prompt.snapshot) == {"enet_b0_8_best_afew", "deepface_emotion"}
    assert s.answer(prompt.id, "happy")
    s.stop()
    reports = [r for r in read_rows(cfg) if r["record_type"] == "self_report"]
    assert {r["model_name"] for r in reports} == {"enet_b0_8_best_afew", "deepface_emotion"}
    assert all(r["self_report"] == "happy" for r in reports)
    assert len({r["timestamp"] for r in reports}) == 1
    assert s.self_reports == 1


def test_switch_primary_while_running(make_config, tmp_path, session_factory):
    _need("deepface_emotion")
    cfg = make_config(logging__dir=str(tmp_path / "logs"), model__name="enet_b0_8_best_afew", compare__mode="off")
    s = session_factory(cfg)
    s.accept_consent()
    assert wait_for(lambda: s.phase is Phase.RUNNING, timeout=20)
    assert s.set_primary("deepface_emotion")
    assert wait_for(lambda: s.state()["primary"] == "deepface_emotion", timeout=15)
    assert wait_for(lambda: (s.state()["result"] or {}).get("status") == "ok", timeout=15)
    models = {m["key"]: m for m in s.state()["models"]}
    assert models["enet_b0_8_best_afew"]["enabled"] is False  # müqayisə söndürülüb → köhnə əsas dayanır
    assert s.camera_active
    s.stop()
    names = [r["model_name"] for r in read_rows(cfg)]
    assert "deepface_emotion" in names and "enet_b0_8_best_afew" in names


def test_enable_extra_model_turns_compare_on(make_config, tmp_path, session_factory):
    _need("deepface_emotion")
    cfg = make_config(logging__dir=str(tmp_path / "logs"), compare__mode="off")
    s = session_factory(cfg)
    s.accept_consent()
    assert wait_for(lambda: s.phase is Phase.RUNNING, timeout=20)
    s.set_model_enabled("deepface_emotion", True)
    assert wait_for(lambda: any(m["key"] == "deepface_emotion" and m["enabled"] for m in s.state()["models"]), timeout=15)
    assert s.state()["compare_mode"] == "on"
    s.set_model_enabled("deepface_emotion", False)
    assert not next(m for m in s.state()["models"] if m["key"] == "deepface_emotion")["enabled"]


def test_compare_off_keeps_only_primary(cmp_config, session_factory):
    _need("deepface_emotion")
    s = session_factory(cmp_config)
    s.accept_consent()
    assert wait_for(lambda: s.phase is Phase.RUNNING, timeout=20)
    s.set_compare_mode("off")
    enabled = [m["key"] for m in s.state()["models"] if m["enabled"]]
    assert enabled == ["enet_b0_8_best_afew"]


def test_blind_mode_masks_history(cmp_config, session_factory):
    _need("deepface_emotion")
    cfg = replace(cmp_config, ui=replace(cmp_config.ui, show_prediction=False))
    s = session_factory(cfg)
    s.accept_consent()
    assert wait_for(lambda: s.state()["frames"] >= 4, timeout=20)
    for m in s.state()["models"]:
        assert all(h in ("ok", "no_face", "warming_up") for h in m["history"])
        assert m["result"] is None or "top" not in m["result"]
    assert s.state()["agreement"] is None


def test_unsupported_contempt_column_blank_in_row():
    from datetime import datetime

    from src.csv_logger import build_row
    from src.model_zoo import spec
    from src.smoothing import SmoothedResult, SmoothStatus

    probs = np.full(8, 1 / 7)
    probs[EMOTION_CLASSES.index("Contempt")] = 0
    s = SmoothedResult(SmoothStatus.OK, probs, "Neutral", 0.6, None, None, 5)
    row = build_row(ts=datetime.now(), operator_id="OP-1", model_name="deepface_emotion", engine="onnx", device="cpu",
                    record_type="sample", smoothed=s, face_reason=None, classes=spec("deepface_emotion").classes)
    assert row["p_contempt"] == "" and row["p_neutral"] != ""
