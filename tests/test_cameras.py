"""Kamera seçimi: siyahı kameranı açmır, seçim yadda qalır, işləyərkən keçid, kamera sızmır."""

import sys
from dataclasses import replace

import pytest

from src import cameras as cams_mod
from src.cameras import CameraInfo, list_cameras, load_choice, resolve, save_choice
from src.capture import StaticSource, SwitchableSource
from src.privacy import require_consent
from src.session import Phase, Session, wait_for

from .conftest import face_frame

CAMS = [CameraInfo(0, "HP HD Camera", r"\\?\usb#vid_0408"), CameraInfo(1, "Logi C270 HD WebCam", r"\\?\usb#vid_046d")]


class FakeCam(StaticSource):
    def __init__(self, cam, frame):
        super().__init__([frame])
        self.cam = cam
        self.closed = False

    def close(self):
        self.closed = True
        super().close()


@pytest.fixture(autouse=True)
def state_file(tmp_path, monkeypatch):
    path = tmp_path / "local_state.json"
    monkeypatch.setattr(cams_mod, "STATE_PATH", path)
    return path


# --- saf hissələr -----------------------------------------------------------------


def test_switchable_source_switches_and_never_leaks():
    a, b, c = (FakeCam(None, None) for _ in range(3))
    src = SwitchableSource(a)
    assert src.switch(b) and a.closed and not b.closed
    src.close()
    assert b.closed
    # Sessiya bağlandıqdan sonra gələn keçid: yeni kamera dərhal bağlanır
    assert src.switch(c) is False and c.closed


def test_resolve_prefers_path_then_name_then_config_index():
    moved = [CameraInfo(0, "Logi C270 HD WebCam", r"\\?\usb#vid_046d"), CameraInfo(1, "HP HD Camera", r"\\?\usb#vid_0408")]
    assert resolve(moved, {"name": "x", "path": r"\\?\usb#vid_046d"}, 1).index == 0  # başqa porta taxılıb
    assert resolve(CAMS, {"name": "Logi C270 HD WebCam", "path": ""}, 0).index == 1
    assert resolve(CAMS, {"name": "Çıxarılıb", "path": "yoxdur"}, 1).index == 1
    assert resolve(CAMS, None, 7).index == 0
    assert resolve([], None, 0) is None


def test_choice_saved_and_corrupt_file_tolerated(state_file):
    assert load_choice() is None
    save_choice(CAMS[1])
    assert load_choice() == {"name": "Logi C270 HD WebCam", "path": r"\\?\usb#vid_046d"}
    state_file.write_text("{pozuq", encoding="utf-8")
    assert load_choice() is None
    save_choice(CAMS[0])
    assert load_choice()["name"] == "HP HD Camera"


def test_duplicate_names_numbered():
    out = cams_mod._number_duplicates([CameraInfo(0, "USB Camera"), CameraInfo(1, "USB Camera")])
    assert [c.name for c in out] == ["USB Camera", "USB Camera (2)"]


@pytest.mark.skipif(sys.platform != "win32", reason="DirectShow yalnız Windows-da")
def test_listing_does_not_open_any_camera(monkeypatch):
    import cv2

    def forbidden(*a, **k):
        raise AssertionError("siyahılama kameranı açmamalıdır")

    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    cams = list_cameras()
    assert [c.index for c in cams] == list(range(len(cams)))
    assert all(c.name for c in cams)


# --- sessiya ilə --------------------------------------------------------------------

pytest_models = pytest.mark.usefixtures("models_ready")


@pytest.fixture(scope="module")
def shared_analyzer(cfg):
    from src.app import build_analyzer

    return build_analyzer(replace(cfg, compare=replace(cfg.compare, mode="off")))


@pytest.fixture
def make_session(make_config, tmp_path, face_bgr, shared_analyzer):
    sessions = []

    def _make(fail=(), **cfg_kw):
        cfg = make_config(logging__dir=str(tmp_path / "logs"), compare__mode="off", **cfg_kw)
        opened = []

        def factory(consent, cam):
            require_consent(consent)
            if cam.index in fail:
                raise RuntimeError(f"Kamera açılmadı: {cam.name}")
            src = FakeCam(cam, face_frame(face_bgr))
            opened.append(src)
            return src

        s = Session(cfg, analyzer_factory=lambda _c: shared_analyzer, tick_interval=0.05,
                    camera_factory=factory, camera_lister=lambda: list(CAMS))
        s.opened = opened
        sessions.append(s)
        return s

    yield _make
    for s in sessions:
        s.shutdown()


@pytest_models
def test_choice_before_consent_does_not_open_camera(make_session):
    s = make_session()
    assert s.state()["cameras"] == [{"index": 0, "name": "HP HD Camera"}, {"index": 1, "name": "Logi C270 HD WebCam"}]
    assert s.select_camera(1) == (True, None)
    assert s.opened == [] and s.state()["camera"]["index"] == 1
    assert load_choice()["name"] == "Logi C270 HD WebCam"
    # Növbəti açılışda yadda qalan kamera seçilir
    assert make_session().camera.index == 1
    s.accept_consent()
    assert wait_for(lambda: s.phase is Phase.RUNNING)
    assert [o.cam.index for o in s.opened] == [1]


@pytest_models
def test_switch_while_running(make_session):
    s = make_session()
    s.accept_consent()
    assert wait_for(lambda: s.phase is Phase.RUNNING)
    first = s.opened[0]
    assert first.cam.index == 0
    frames = s.state()["frames"]
    assert s.select_camera(1) == (True, None)
    assert first.closed and not s.opened[1].closed and s.camera.index == 1
    assert wait_for(lambda: s.state()["frames"] > frames + 1), "keçiddən sonra analiz davam etməlidir"
    assert s.phase is Phase.RUNNING and s.camera_active
    s.stop()
    assert all(o.closed for o in s.opened), "dayandırandan sonra heç bir kamera açıq qalmamalıdır"


@pytest_models
def test_failed_switch_keeps_old_camera(make_session):
    s = make_session(fail=(1,))
    s.accept_consent()
    assert wait_for(lambda: s.phase is Phase.RUNNING)
    ok, err = s.select_camera(1)
    assert not ok and "Logi" in err
    assert s.camera.index == 0 and not s.opened[0].closed and s.phase is Phase.RUNNING
    assert load_choice() is None


@pytest_models
def test_no_switch_during_controlled_test(make_session):
    s = make_session(test__intro_seconds=5)
    s.accept_consent()
    assert wait_for(lambda: s.phase is Phase.RUNNING)
    assert s.start_test()
    ok, err = s.select_camera(1)
    assert not ok and "test" in err and s.camera.index == 0
