from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from src.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, AppConfig, load_config, parse_config

FIXTURES = Path(__file__).parent / "fixtures"
FACE_FIXTURE = FIXTURES / "face.png"


@pytest.fixture(scope="session")
def raw_config() -> dict:
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def make_config(raw_config):
    """Default config-in üzərinə dəyişiklik edib AppConfig qaytarır."""

    def _make(**overrides) -> AppConfig:
        raw = copy.deepcopy(raw_config)
        for dotted, value in overrides.items():
            node = raw
            *parents, leaf = dotted.split("__")
            for key in parents:
                node = node.setdefault(key, {})
            node[leaf] = value
        return parse_config(raw, base_dir=PROJECT_ROOT)

    return _make


@pytest.fixture(scope="session")
def cfg() -> AppConfig:
    return load_config()


@pytest.fixture(scope="session")
def face_bgr():
    import cv2

    if not FACE_FIXTURE.is_file():
        pytest.skip("tests/fixtures/face.png yoxdur — scripts/download_models.py --test-fixture")
    img = cv2.imread(str(FACE_FIXTURE))
    assert img is not None
    return img


@pytest.fixture(scope="session")
def yunet_path(cfg) -> Path:
    if not cfg.detector.model_path.is_file():
        pytest.skip("YuNet modeli yoxdur — scripts/download_models.py")
    return cfg.detector.model_path


@pytest.fixture(scope="session")
def models_ready(cfg, yunet_path):
    from src.recognizer import weights_path

    if not weights_path(cfg.model.name, cfg.model.engine).is_file():
        pytest.skip("Emosiya modeli yoxdur — scripts/download_models.py")


def face_frame(face_bgr, size=(720, 1280)):
    """Test şəklini daha böyük kadrın ortasına yerləşdirir."""
    import numpy as np

    h, w = size
    frame = np.full((h, w, 3), 110, np.uint8)
    fh, fw = face_bgr.shape[:2]
    y0, x0 = (h - fh) // 2, (w - fw) // 2
    frame[y0 : y0 + fh, x0 : x0 + fw] = face_bgr
    return frame


@pytest.fixture(scope="session")
def test_video(tmp_path_factory, face_bgr) -> Path:
    """10 fps, 8 s: 0–5 s üz, 5–8 s boş kadr. Testin GİRİŞ faylıdır (qoruyucular qurulmazdan əvvəl)."""
    import cv2
    import numpy as np

    path = tmp_path_factory.mktemp("video") / "input.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (1280, 720))
    assert writer.isOpened()
    face = face_frame(face_bgr)
    blank = np.full_like(face, 110)
    for i in range(80):
        writer.write(face if i < 50 else blank)
    writer.release()
    return path


@pytest.fixture
def tmp_config(make_config, tmp_path):
    """Log qovluğu tmp_path-də olan, tək modelli config (müqayisə testləri ayrıdır)."""
    return make_config(logging__dir=str(tmp_path / "logs"), compare__mode="off")
