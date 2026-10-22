import sys
import types
from dataclasses import replace

import cv2
import numpy as np
import pytest

from src import recognizer as rec
from src.config import EMOTION_CLASSES, ModelConfig
from src.detector import FaceDetector
from src.recognizer import EmotionRecognizer, resolve_device


@pytest.fixture(scope="module")
def face_crop(cfg, yunet_path, face_bgr):
    det = FaceDetector(cfg.detector)
    d = det.detect(face_bgr)
    assert d.ok
    return det.crop(face_bgr, d.face)


def _need_weights(name, engine="onnx"):
    if not rec.weights_path(name, engine).is_file():
        pytest.skip(f"{name} çəkisi yoxdur — scripts/download_models.py")


@pytest.mark.models
@pytest.mark.parametrize("name", ["enet_b0_8_best_afew", "enet_b0_8_va_mtl", "enet_b2_8", "mbf_va_mtl"])
def test_prediction_shape_and_probabilities(cfg, face_crop, name):
    _need_weights(name)
    r = EmotionRecognizer(replace(cfg.model, name=name))
    p = r.predict(face_crop)
    assert p.probs.shape == (len(EMOTION_CLASSES),)
    assert np.all(p.probs >= 0) and np.all(p.probs <= 1)
    assert p.probs.sum() == pytest.approx(1.0, abs=1e-4)
    assert p.latency_ms > 0
    assert r.device == "cpu"
    if name.endswith("_mtl"):
        assert -1.5 <= p.valence <= 1.5 and -1.5 <= p.arousal <= 1.5
    else:
        assert p.valence is None and p.arousal is None


@pytest.mark.models
def test_smiling_portrait_is_happiness(cfg, face_crop):
    """Test şəkli gülümsəyən portretdir — default model bunu tanımalıdır."""
    _need_weights(cfg.model.name)
    p = EmotionRecognizer(cfg.model).predict(face_crop)
    assert EMOTION_CLASSES[int(np.argmax(p.probs))] == "Happiness"


@pytest.mark.models
def test_mtl_positive_valence_for_smile(cfg, face_crop):
    _need_weights("enet_b0_8_va_mtl")
    p = EmotionRecognizer(replace(cfg.model, name="enet_b0_8_va_mtl")).predict(face_crop)
    assert p.valence > 0.3


@pytest.mark.models
def test_input_converted_to_rgb(cfg, face_crop, monkeypatch):
    _need_weights(cfg.model.name)
    r = EmotionRecognizer(cfg.model)
    seen = {}
    original = r._fer.predict_emotions

    def spy(img, logits=True):
        seen["img"] = img.copy()
        return original(img, logits=logits)

    monkeypatch.setattr(r._fer, "predict_emotions", spy)
    r.predict(face_crop)
    assert np.array_equal(seen["img"], cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))


def test_missing_weights_do_not_trigger_download(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "WEIGHTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="download_models"):
        EmotionRecognizer(ModelConfig())


@pytest.mark.models
def test_onnx_engine_does_not_import_torch(cfg):
    """torch yalnız çevirmə üçün qurulsa belə, ONNX modeli yüklənəndə import olunmamalıdır (RAM/başlanğıc)."""
    import subprocess

    from src.config import PROJECT_ROOT

    _need_weights(cfg.model.name)
    code = (
        "import sys; from src.model_zoo import make_recognizer; "
        f"make_recognizer({cfg.model.name!r}); print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_onnx_always_cpu():
    assert resolve_device("onnx", "cpu") == "cpu"


def _fake_torch(cuda: bool):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    return torch


def test_torch_cuda_falls_back_to_cpu_when_unavailable(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    assert resolve_device("torch", "cuda") == "cpu"
    assert "CUDA" in caplog.text


def test_torch_cuda_used_when_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True))
    assert resolve_device("torch", "cuda") == "cuda"


def test_torch_missing_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="requirements-gpu"):
        resolve_device("torch", "cuda")
