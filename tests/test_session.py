import csv

import pytest

from src.app import build_analyzer
from src.capture import PacedVideoSource, StaticSource
from src.privacy import ConsentToken, require_consent
from src.session import Phase, Session, wait_for

from .conftest import face_frame

pytestmark = pytest.mark.usefixtures("models_ready")


@pytest.fixture(scope="module")
def shared_analyzer(cfg):
    from dataclasses import replace

    return build_analyzer(replace(cfg, compare=replace(cfg.compare, mode="off")))


@pytest.fixture
def make_session(tmp_config, face_bgr, shared_analyzer, make_config, tmp_path):
    sessions = []

    def _make(cfg=None, source_factory=None, **kw):
        cfg = cfg or tmp_config
        if cfg.compare.mode != "off":
            from dataclasses import replace

            cfg = replace(cfg, compare=replace(cfg.compare, mode="off"))
        calls = []

        def default_factory(consent):
            calls.append(require_consent(consent))
            return StaticSource([face_frame(face_bgr)])

        s = Session(
            cfg,
            source_factory=source_factory or default_factory,
            analyzer_factory=lambda _cfg: shared_analyzer,
            tick_interval=0.05,
            **kw,
        )
        s.factory_calls = calls
        sessions.append(s)
        return s

    yield _make
    for s in sessions:
        s.shutdown()


def rows(cfg):
    out = []
    for p in sorted(cfg.logging.dir.glob("*.csv")):
        with p.open(encoding="utf-8", newline="") as fh:
            out += list(csv.DictReader(fh))
    return out


def test_camera_not_opened_before_consent(make_session):
    s = make_session()
    assert s.phase is Phase.CONSENT
    assert not s.camera_active
    assert s.factory_calls == []
    st = s.state()
    assert st["phase"] == "consent" and st["camera_active"] is False and st["result"] is None


def test_consent_starts_with_valid_token(make_session):
    s = make_session()
    changes = []
    s.on_change = lambda: changes.append(s.phase)
    s.accept_consent()
    assert wait_for(lambda: s.phase is Phase.RUNNING)
    assert isinstance(s.factory_calls[0], ConsentToken)
    assert s.camera_active
    assert changes[:2] == [Phase.LOADING, Phase.RUNNING]
    assert s._guards is not None and s._guards.network is not None and s._guards.network.active


def test_state_shows_smoothed_result(make_session):
    s = make_session()
    s.accept_consent()
    assert wait_for(lambda: (s.state()["result"] or {}).get("status") == "ok", timeout=8)
    st = s.state()
    assert st["result"]["top"] == "Happiness"
    assert st["result"]["top_label"] == "Xoşbəxtlik"
    assert len(st["result"]["probs"]) == 8
    assert st["face"]["ok"] is True
    assert st["latency_ms"] > 0
    # Anlıq: son kadrın öz nəticəsi (ortalamasız), həm əsas model, həm model kartı üçün
    assert st["instant"]["status"] == "ok" and st["instant"]["top"] == "Happiness"
    assert st["instant"]["top_label"] == "Xoşbəxtlik" and 0 < st["instant"]["top_prob"] <= 1
    primary = next(m for m in st["models"] if m["primary"])
    assert primary["instant"]["top"] == "Happiness"


def test_stop_closes_camera_and_keeps_csv(make_session, tmp_config):
    s = make_session()
    s.accept_consent()
    assert wait_for(lambda: s.state()["frames"] >= 2, timeout=8)
    s.stop()
    assert s.phase is Phase.STOPPED
    assert not s.camera_active
    assert s._source is None and s._pipeline is None
    assert len(rows(tmp_config)) >= 2


def test_restart_requires_new_consent(make_session):
    s = make_session()
    s.accept_consent()
    assert wait_for(lambda: s.phase is Phase.RUNNING)
    s.stop()
    s.restart()
    assert s.phase is Phase.CONSENT
    s.accept_consent()
    assert wait_for(lambda: s.phase is Phase.RUNNING)
    assert len(s.factory_calls) == 2


def test_self_report_answer_logged_with_snapshot(make_session, make_config, tmp_path):
    cfg = make_config(logging__dir=str(tmp_path / "logs"), self_report__interval_minutes=0.01)  # 0.6 s
    s = make_session(cfg)
    opened = []
    s.on_prompt = opened.append
    s.accept_consent()
    assert wait_for(lambda: any(p is not None for p in opened), timeout=10)
    prompt = next(p for p in opened if p is not None)
    assert s.prompt_state()["id"] == prompt.id
    assert s.answer(prompt.id, "happy")
    assert opened[-1] is None
    assert not s.answer(prompt.id, "happy")  # təkrar cavab yazılmır
    s.stop()
    reports = [r for r in rows(cfg) if r["record_type"] == "self_report"]
    assert len(reports) == 1
    assert reports[0]["self_report"] == "happy"
    assert reports[0]["status"] == prompt.snapshot[cfg.model.name].status.value


def test_self_report_timeout_logged(make_session, make_config, tmp_path):
    cfg = make_config(
        logging__dir=str(tmp_path / "logs"), self_report__interval_minutes=0.005, self_report__timeout_seconds=0.3
    )
    s = make_session(cfg)
    s.accept_consent()
    assert wait_for(lambda: s.self_reports >= 1, timeout=10)
    s.stop()
    reports = [r for r in rows(cfg) if r["record_type"] == "self_report"]
    assert reports and reports[0]["self_report"] == "timeout"


def test_blind_mode_hides_prediction(make_session, make_config, tmp_path):
    cfg = make_config(logging__dir=str(tmp_path / "logs"), ui__show_prediction=False)
    s = make_session(cfg)
    s.accept_consent()
    assert wait_for(lambda: (s.state()["result"] or {}).get("status") == "ok", timeout=8)
    result = s.state()["result"]
    assert "top" not in result and "probs" not in result
    assert "top" not in s.state()["instant"], "kor rejimdə anlıq nəticə də gizlidir"
    s.stop()
    assert any(r["top_emotion"] == "Happiness" for r in rows(cfg))  # CSV yenə tam yazılır


def test_preview_only_when_enabled(make_session):
    s = make_session()
    s.accept_consent()
    assert wait_for(lambda: s.state()["frames"] >= 1, timeout=8)
    s.preview_enabled = False
    assert s.preview_data_url() is None
    s.preview_enabled = True
    url = s.preview_data_url()
    assert url and url.startswith("data:image/jpeg;base64,")


def test_video_end_finishes_session(make_session, test_video, tmp_config, shared_analyzer):
    def factory(consent):
        src = PacedVideoSource(test_video)
        src._frame_interval = 0.002
        return src

    s = make_session(source_factory=factory)
    s.video = str(test_video)
    s.accept_consent()
    assert wait_for(lambda: s.phase is Phase.FINISHED, timeout=15)
    assert not s.camera_active


def test_camera_failure_reported(make_session):
    def broken(consent):
        raise RuntimeError("Kamera açılmadı (index 0)")

    s = make_session(source_factory=broken)
    s.accept_consent()
    assert wait_for(lambda: s.phase is Phase.ERROR)
    assert "Kamera" in s.state()["error"]
    s.restart()
    assert s.phase is Phase.CONSENT


def test_shutdown_removes_guards(make_session):
    s = make_session()
    s.accept_consent()
    assert wait_for(lambda: s.phase is Phase.RUNNING)
    guards = s._guards
    s.shutdown()
    assert not guards.disk.active and not guards.network.active
