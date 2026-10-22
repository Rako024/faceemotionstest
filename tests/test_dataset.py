"""Araşdırma dataseti: şəkillər yalnız razılıqla, yalnız nəzarətli testdə, yalnız research.dir altına."""

import csv
import json
import os

import cv2
import pytest

from src.capture import StaticSource
from src.dataset import load_samples, main as dataset_cli, sessions
from src.session import Phase, Session, wait_for

from .conftest import face_frame

pytestmark = pytest.mark.usefixtures("models_ready")

MEDIA = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".mp4", ".avi"}


@pytest.fixture
def test_cfg(make_config, tmp_path):
    def make(**kw):
        base = dict(
            logging__dir=str(tmp_path / "logs"), research__dir=str(tmp_path / "dataset"), compare__mode="off",
            capture__analysis_fps=2.0, test__expressions=["Happiness", "Neutral"], test__rounds=1,
            test__intro_seconds=0.2, test__prepare_seconds=0.2, test__hold_seconds=2.5, test__rest_seconds=0.1,
            test__shuffle=False,
        )
        base.update(kw)
        return make_config(**base)

    return make


def run_test(cfg, face_bgr, **start_kw):
    s = Session(cfg, source_factory=lambda c: StaticSource([face_frame(face_bgr)]), tick_interval=0.05)
    try:
        s.accept_consent()
        assert wait_for(lambda: s.phase is Phase.RUNNING, timeout=20)
        ok = s.start_test(**start_kw)
        if ok:
            assert wait_for(lambda: s.state()["test"]["done"], timeout=40)
            state = s.state()["test"]
        else:
            state = None
        return s, ok, state
    finally:
        s.shutdown()


def media_files(root):
    found = []
    for dirpath, _, files in os.walk(root):
        found += [os.path.join(dirpath, f) for f in files if os.path.splitext(f)[1].lower() in MEDIA]
    return found


def test_frames_saved_only_with_consent(test_cfg, face_bgr, tmp_path):
    cfg = test_cfg()
    s, ok, st = run_test(cfg, face_bgr, tester_id="T-07", save_frames=True)
    assert ok and st["tester_id"] == "T-07" and st["saved_frames"] > 0
    (sdir,) = sessions(cfg.research.dir)
    assert sdir.name.endswith("_T-07")
    meta = json.loads((sdir / "meta.json").read_text(encoding="utf-8"))
    assert meta["closed"] and meta["frames"] == st["saved_frames"] and "razılıq" in meta["consent"]
    with (sdir / "frames.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == meta["frames"]
    assert {r["expected"] for r in rows} == {"Happiness", "Neutral"}
    img = cv2.imread(str(sdir / rows[0]["file"]))
    h, w = img.shape[:2]
    bx, by, bw, bh = (float(rows[0][k]) for k in ("box_x", "box_y", "box_w", "box_h"))
    assert 0 <= bx and bx + bw <= w + 1 and 0 <= by and by + bh <= h + 1
    assert w < 1280 and h < 720, "bütün kadr yox, üz ətrafı kəsik saxlanmalıdır"
    # Bütün şəkillər yalnız dataset qovluğundadır
    assert {os.path.dirname(p) for p in media_files(tmp_path)} == {str(sdir)}
    # CSV test sətirlərində tester ID
    with next((tmp_path / "logs").glob("*.csv")).open(encoding="utf-8", newline="") as fh:
        test_rows = [r for r in csv.DictReader(fh) if r["record_type"] == "test"]
    assert test_rows and {r["operator_id"] for r in test_rows} == {"T-07"}


def test_no_frames_without_consent(test_cfg, face_bgr, tmp_path):
    cfg = test_cfg()
    _, ok, st = run_test(cfg, face_bgr, tester_id="T-08", save_frames=False)
    assert ok and st["saved_frames"] is None
    assert not cfg.research.dir.exists()
    assert media_files(tmp_path) == []


def test_config_switch_overrides_consent(test_cfg, face_bgr, tmp_path):
    cfg = test_cfg(research__save_test_frames=False)
    _, ok, _ = run_test(cfg, face_bgr, tester_id="T-09", save_frames=True)
    assert ok and media_files(tmp_path) == []


def test_name_like_tester_id_rejected(test_cfg, face_bgr):
    s, ok, _ = run_test(test_cfg(), face_bgr, tester_id="Rauf", save_frames=True)
    assert not ok and "anonim" in s.test_error


def test_guard_still_blocks_other_writes_during_test(test_cfg, face_bgr, tmp_path):
    """Dataset yazılsa da, adi API-lər (cv2.imwrite) bağlı qalır."""
    import numpy as np

    cfg = test_cfg()
    s = Session(cfg, source_factory=lambda c: StaticSource([face_frame(face_bgr)]), tick_interval=0.05)
    try:
        s.accept_consent()
        assert wait_for(lambda: s.phase is Phase.RUNNING, timeout=20)
        assert s.start_test(tester_id="T-10", save_frames=True)
        with pytest.raises(PermissionError):
            cv2.imwrite(str(tmp_path / "leak.jpg"), np.zeros((4, 4, 3), np.uint8))
    finally:
        s.shutdown()


def test_loader_and_cli(test_cfg, face_bgr, tmp_path, capsys, monkeypatch):
    cfg = test_cfg()
    run_test(cfg, face_bgr, tester_id="T-11", save_frames=True)
    samples = load_samples(cfg.research.dir)
    assert samples and {s.tester_id for s in samples} == {"T-11"}
    assert samples[0].image().ndim == 3 and samples[0].landmarks.shape == (5, 2)
    assert load_samples(cfg.research.dir, testers={"T-99"}) == []

    import src.dataset as ds

    monkeypatch.setattr(ds, "load_config", lambda _p=None: cfg)
    assert dataset_cli([]) == 0 and "T-11" in capsys.readouterr().out
    (sdir,) = sessions(cfg.research.dir)
    assert dataset_cli(["--delete", sdir.name, "--yes"]) == 0
    assert sessions(cfg.research.dir) == []
