import numpy as np
import pytest

from src.config import EMOTION_CLASSES, FER_KEYS, SUPPORTED_MODELS, ConfigError
from src.detector import FaceDetector
from src.model_zoo import FER_LABELS, SPECS, is_installed, make_recognizer, spec


def test_registry_matches_config():
    assert set(SPECS) == set(SUPPORTED_MODELS)
    assert set(FER_KEYS) == {k for k, s in SPECS.items() if s.family == "fer"}


def test_fer_models_have_no_contempt():
    for key in FER_KEYS:
        s = spec(key)
        assert "Contempt" not in s.classes
        assert s.classes == set(FER_LABELS)
        assert s.onnx_file and s.onnx_file.endswith(".onnx")


def test_unknown_model():
    with pytest.raises(KeyError):
        spec("resnet")


def test_fer_model_requires_onnx_cpu(make_config):
    with pytest.raises(ConfigError, match="onnx"):
        make_config(model__name="vit_fer2013", model__engine="torch", model__device="cuda")
    assert make_config(model__name="deepface_emotion").model.name == "deepface_emotion"


@pytest.fixture(scope="module")
def face_crop(cfg, yunet_path, face_bgr):
    det = FaceDetector(cfg.detector)
    d = det.detect(face_bgr)
    return det.crop(face_bgr, d.face)


@pytest.mark.models
@pytest.mark.parametrize("key", list(FER_KEYS))
def test_fer_model_predicts_smile(key, face_crop):
    if not is_installed(key):
        pytest.skip(f"{key} quraşdırılmayıb — scripts/download_models.py --extra")
    r = make_recognizer(key)
    p = r.predict(face_crop)
    assert p.probs.shape == (8,)
    assert p.probs.sum() == pytest.approx(1.0, abs=1e-4)
    assert p.probs[EMOTION_CLASSES.index("Contempt")] == 0.0
    assert EMOTION_CLASSES[int(np.argmax(p.probs))] == "Happiness"
    assert p.valence is None and p.latency_ms > 0
    assert (r.model_name, r.engine, r.device) == (key, "onnx", "cpu")


@pytest.mark.models
def test_fer_label_order_is_mapped(face_crop):
    """DeepFace çıxışı FER sırasındadır — bizim siniflərə düzgün köçürülməlidir."""
    if not is_installed("deepface_emotion"):
        pytest.skip("deepface_emotion yoxdur")
    r = make_recognizer("deepface_emotion")
    raw = r._session.run(None, {r._input_name: r._preprocess(face_crop)})[0][0]
    p = r.predict(face_crop).probs
    for i, label in enumerate(FER_LABELS):
        assert p[EMOTION_CLASSES.index(label)] == pytest.approx(raw[i], abs=1e-6)


def test_missing_fer_model_gives_hint(monkeypatch, tmp_path):
    import src.model_zoo as mz

    monkeypatch.setattr(mz, "MODELS_DIR", tmp_path)
    assert not is_installed("vit_fer2013")
    with pytest.raises(FileNotFoundError, match="--extra"):
        make_recognizer("vit_fer2013")


@pytest.mark.models
def test_emotiefflib_through_factory(face_crop, cfg):
    if not is_installed(cfg.model.name):
        pytest.skip("EmotiEffLib çəkisi yoxdur")
    r = make_recognizer(cfg.model.name)
    assert r.spec.family == "emotiefflib"
    assert EMOTION_CLASSES[int(np.argmax(r.predict(face_crop).probs))] == "Happiness"
