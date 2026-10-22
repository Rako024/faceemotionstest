"""Keras .h5 → ONNX çeviricisinin sintetik modellərlə yoxlanması (Kaggle CNN üçün)."""

import os
import sys

import numpy as np
import pytest

from src.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
os.environ.setdefault("KERAS_BACKEND", "torch")
keras = pytest.importorskip("keras", reason="keras qurulmayıb (requirements-convert.txt)")
ort = pytest.importorskip("onnxruntime")

from keras_h5_to_onnx import UnsupportedLayer, convert  # noqa: E402


def _fer_like_cnn():
    """Tipik FER2013 CNN: 48×48×1 → 7 sinif."""
    L = keras.layers
    keras.utils.set_random_seed(1)
    model = keras.Sequential([
        keras.Input((48, 48, 1)),
        L.Conv2D(8, (3, 3), padding="same", activation="relu"),
        L.BatchNormalization(),
        L.Conv2D(8, (3, 3), activation="relu"),
        L.MaxPooling2D((2, 2)),
        L.Dropout(0.25),
        L.ZeroPadding2D(1),
        L.Conv2D(16, (3, 3), strides=2, padding="same"),
        L.LeakyReLU(negative_slope=0.1),
        L.AveragePooling2D((2, 2), padding="same"),
        L.Flatten(),
        L.Dense(32),
        L.BatchNormalization(),
        L.Activation("relu"),
        L.Dense(7, activation="softmax"),
    ])
    # BatchNorm-un hərəkətli statistikaları sıfırdan fərqli olsun
    for layer in model.layers:
        if isinstance(layer, L.BatchNormalization):
            n = layer.moving_mean.shape[0]
            rng = np.random.default_rng(n)
            layer.set_weights([rng.uniform(0.5, 1.5, n), rng.normal(0, 0.1, n), rng.normal(0, 0.2, n), rng.uniform(0.5, 2, n)])
    return model


def test_converted_matches_keras(tmp_path):
    model = _fer_like_cnn()
    h5 = tmp_path / "m.h5"
    model.save(str(h5))
    info = convert(h5, tmp_path / "m.onnx")
    assert info["input_shape"] == [48, 48, 1] and info["classes"] == 7

    batch = np.random.default_rng(0).random((5, 48, 48, 1), dtype=np.float32)
    ref = np.asarray(model.predict(batch, verbose=0))
    got = ort.InferenceSession(str(tmp_path / "m.onnx"), providers=["CPUExecutionProvider"]).run(None, {"input": batch})[0]
    np.testing.assert_allclose(got, ref, atol=1e-5)
    assert got.sum(axis=1) == pytest.approx(np.ones(5), abs=1e-5)


def test_global_pooling_rgb_model(tmp_path):
    L = keras.layers
    model = keras.Sequential([
        keras.Input((32, 32, 3)),
        L.Conv2D(4, 3, activation="relu"),
        L.GlobalAveragePooling2D(),
        L.Dense(7),
        L.Softmax(),
    ])
    model.save(str(tmp_path / "g.h5"))
    convert(tmp_path / "g.h5", tmp_path / "g.onnx")
    batch = np.random.default_rng(1).random((2, 32, 32, 3), dtype=np.float32)
    got = ort.InferenceSession(str(tmp_path / "g.onnx"), providers=["CPUExecutionProvider"]).run(None, {"input": batch})[0]
    np.testing.assert_allclose(got, np.asarray(model.predict(batch, verbose=0)), atol=1e-5)


def test_unsupported_layer_is_reported(tmp_path):
    L = keras.layers
    model = keras.Sequential([keras.Input((16, 16, 1)), L.Conv2D(2, 3), L.UpSampling2D(), L.Flatten(), L.Dense(7)])
    model.save(str(tmp_path / "u.h5"))
    with pytest.raises(UnsupportedLayer, match="UpSampling2D"):
        convert(tmp_path / "u.h5", tmp_path / "u.onnx")


def test_weights_only_file_rejected(tmp_path):
    model = _fer_like_cnn()
    model.save_weights(str(tmp_path / "w.weights.h5"))
    with pytest.raises(UnsupportedLayer, match="model_config"):
        convert(tmp_path / "w.weights.h5", tmp_path / "w.onnx")
