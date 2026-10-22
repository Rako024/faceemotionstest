"""Nəzarətli test: zaman xətti, qeydiyyat, sessiya inteqrasiyası, CSV."""

import csv
from dataclasses import replace

import numpy as np
import pytest

from src.calibration import DONE, HOLD, INTRO, PREPARE, REST, CalibrationRun, ModelTally, build_plan, raw_result
from src.config import EMOTION_CLASSES, CalibrationConfig
from src.recognizer import EmotionPrediction
from src.smoothing import SmoothStatus

CFG = CalibrationConfig(expressions=["Neutral", "Happiness", "Anger"], rounds=2, intro_seconds=4,
                 prepare_seconds=3, hold_seconds=8, rest_seconds=2, shuffle=True)


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def pred(top, p=0.8):
    probs = np.full(8, (1 - p) / 7)
    probs[EMOTION_CLASSES.index(top)] = p
    return EmotionPrediction(probs, None, None, 5.0)


def test_plan_neutral_first_and_all_expressions_each_round():
    steps = build_plan(CFG, seed=1)
    assert len(steps) == 6
    for r in (1, 2):
        exprs = [s.expression for s in steps if s.round == r]
        assert exprs[0] == "Neutral"
        assert sorted(exprs) == sorted(CFG.expressions)
    assert [s.index for s in steps] == list(range(6))


def test_plan_without_shuffle_keeps_order():
    steps = build_plan(replace(CFG, shuffle=False, rounds=1))
    assert [s.expression for s in steps] == CFG.expressions


def test_timeline_phases():
    clock = Clock()
    run = CalibrationRun(CFG, clock=clock, seed=1)
    assert run.total_seconds == 4 + 13 * 6
    assert run.position().phase == INTRO
    clock.t += 4.5
    p = run.position()
    assert p.phase == PREPARE and p.step.index == 0 and p.seconds_left == pytest.approx(2.5)
    assert run.expected_now() is None
    clock.t += 3
    p = run.position()
    assert p.phase == HOLD and run.expected_now() == "Neutral"
    clock.t += 8
    assert run.position().phase == REST
    clock.t += 2
    p = run.position()
    assert p.phase == PREPARE and p.step.index == 1
    clock.t = 100 + run.total_seconds
    assert run.position().phase == DONE and run.done


def test_cancel_finishes_immediately():
    clock = Clock()
    run = CalibrationRun(CFG, clock=clock)
    run.cancel()
    assert run.position().phase == DONE


def test_raw_result_statuses():
    assert raw_result(None, 0.5).status is SmoothStatus.NO_FACE
    r = raw_result(pred("Anger", 0.7), 0.5)
    assert (r.status, r.top_emotion, r.n_frames) == (SmoothStatus.OK, "Anger", 1)
    assert raw_result(pred("Anger", 0.4), 0.5).status is SmoothStatus.UNCERTAIN


def test_tally_accuracy_and_balanced():
    t = ModelTally()
    for _ in range(8):
        t.add("Happiness", raw_result(pred("Happiness"), 0.5))
    for _ in range(2):
        t.add("Happiness", raw_result(pred("Neutral"), 0.5))
    for _ in range(2):
        t.add("Anger", raw_result(pred("Anger", 0.45), 0.5))  # düzgün, amma qeyri-müəyyən
    for _ in range(2):
        t.add("Anger", raw_result(pred("Neutral"), 0.5))
    t.add("Anger", raw_result(None, 0.5))  # üz yoxdur — məxrəcə daxil deyil
    s = t.summary(["Neutral", "Happiness", "Anger"])
    assert (s["frames"], s["face"]) == (15, 14)
    assert s["accuracy"] == pytest.approx(10 / 14)
    assert s["per_expression"] == {"Neutral": None, "Happiness": 0.8, "Anger": 0.5}
    assert s["balanced"] == pytest.approx(0.65)
    assert s["uncertain"] == pytest.approx(2 / 14)


def test_config_validation(make_config):
    from src.config import ConfigError

    with pytest.raises(ConfigError, match="expressions"):
        make_config(test__expressions=["Happy"])
    with pytest.raises(ConfigError):
        make_config(test__hold_seconds=1)
    assert make_config(test__expressions=["Neutral", "Contempt"]).test.expressions == ["Neutral", "Contempt"]


# --- sessiya ilə ----------------------------------------------------------------

@pytest.mark.usefixtures("models_ready")
def test_session_test_writes_labelled_rows_only(make_config, tmp_path, face_bgr):
    from src.capture import StaticSource
    from src.session import Phase, Session, wait_for

    from .conftest import face_frame

    cfg = make_config(
        logging__dir=str(tmp_path / "logs"), compare__mode="off", capture__analysis_fps=2.0,
        test__expressions=["Happiness", "Neutral"], test__rounds=1, test__intro_seconds=0.3,
        test__prepare_seconds=0.3, test__hold_seconds=3.5, test__rest_seconds=0.2, test__shuffle=False,
        self_report__interval_minutes=0.005,
    )
    s = Session(cfg, source_factory=lambda c: StaticSource([face_frame(face_bgr)]), tick_interval=0.05)
    prompts = []
    s.on_prompt = prompts.append
    try:
        s.accept_consent()
        assert wait_for(lambda: s.phase is Phase.RUNNING, timeout=20)
        assert s.start_test(seed=0)
        prompts.clear()  # testdən əvvəl açılmış sorğu start_test tərəfindən bağlanır — yalnız test müddəti yoxlanılır
        assert s.preview_enabled is True
        st = s.state()["test"]
        assert st["active"] and st["summary"] is None
        assert wait_for(lambda: s.state()["test"]["done"], timeout=40)
        summary = {r["key"]: r for r in s.state()["test"]["summary"]}
        enet = summary[cfg.model.name]
        # Yüklü maşında kadr sayı dəyişə bilər — yalnız istiqaməti yoxlayırıq
        assert enet["per_expression"]["Happiness"] == 1.0  # test şəkli gülümsəyir
        assert enet["per_expression"]["Neutral"] == 0.0
        assert 0.2 <= enet["accuracy"] <= 0.8
        assert all(p is None for p in prompts), "test və nəticə ekranı zamanı sorğu açılmamalıdır"
        assert s.dismiss_test() and s.state()["test"] is None
        assert s._scheduler.seconds_until_next() > 0, "panelə qayıdan kimi sorğu açılmamalıdır"
        assert s.preview_enabled is False
        s.stop()
    finally:
        s.shutdown()

    rows = []
    for p in (tmp_path / "logs").glob("*.csv"):
        with p.open(encoding="utf-8", newline="") as fh:
            rows += list(csv.DictReader(fh))
    test_rows = [r for r in rows if r["record_type"] == "test"]
    assert test_rows and {r["expected"] for r in test_rows} == {"Happiness", "Neutral"}
    assert all(r["n_frames"] == "1" for r in test_rows), "test kadrı smoothing-siz olmalıdır"
    assert all(r["expected"] == "" for r in rows if r["record_type"] != "test")


@pytest.mark.usefixtures("models_ready")
def test_cancel_keeps_partial_summary(make_config, tmp_path, face_bgr):
    from src.capture import StaticSource
    from src.session import Phase, Session, wait_for

    from .conftest import face_frame

    cfg = make_config(logging__dir=str(tmp_path / "logs"), compare__mode="off", capture__analysis_fps=2.0,
                      test__intro_seconds=0, test__prepare_seconds=0, test__hold_seconds=3)
    s = Session(cfg, source_factory=lambda c: StaticSource([face_frame(face_bgr)]), tick_interval=0.05)
    try:
        s.accept_consent()
        assert wait_for(lambda: s.phase is Phase.RUNNING, timeout=20)
        s.start_test()
        assert wait_for(lambda: bool(s._test.tallies), timeout=10)
        assert s.cancel_test()
        st = s.state()["test"]
        assert st["done"] and st["cancelled"] and st["summary"]
    finally:
        s.shutdown()
