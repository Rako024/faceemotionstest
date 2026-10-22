"""Ön emal variantları: boşluq, düzləndirmə, flip TTA."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from src.recognizer import EmotionPrediction
from src.variants import Variant, alignment_matrix, face_crop, predict, roll_degrees


def test_variant_names():
    assert Variant(0.1).key == "m10"
    assert Variant(0.0, align=True, flip=True).key == "m0_al_fl"
    assert "düzləndirmə" in Variant(0.2, align=True).name


def test_margin_crop_size_and_clamp():
    img = np.zeros((400, 600, 3), np.uint8)
    assert face_crop(img, (200, 100, 100, 120), None, 0.0, False).shape[:2] == (120, 100)
    assert face_crop(img, (200, 100, 100, 120), None, 0.1, False).shape[:2] == (144, 120)
    assert face_crop(img, (-10, -10, 100, 100), None, 0.2, False).shape[:2] == (110, 110)


def test_alignment_makes_eyes_horizontal():
    lm = np.array([[100, 110], [160, 140], [130, 150], [110, 180], [150, 190]], np.float32)  # ~26.6° əyik
    assert roll_degrees(lm) == pytest.approx(26.565, abs=0.01)
    m = alignment_matrix(lm)
    eyes = np.hstack([lm[:2], np.ones((2, 1))]) @ m.T
    assert eyes[0, 1] == pytest.approx(eyes[1, 1], abs=1e-3)
    assert np.hypot(*(eyes[1] - eyes[0])) == pytest.approx(np.hypot(*(lm[1] - lm[0])), rel=1e-6)


def test_aligned_crop_keeps_face_size():
    img = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    lm = np.array([[170, 180], [230, 200], [200, 220], [180, 250], [220, 260]], np.float32)
    crop = face_crop(img, (150, 150, 100, 130), lm, 0.0, True)
    h, w = crop.shape[:2]  # fırlanmadan sonra koordinatlar tam ədəd deyil — ±1 px
    assert abs(h - 130) <= 1 and abs(w - 100) <= 1


def test_aligned_face_detected_upright(cfg, yunet_path, face_bgr):
    """Fırladılmış üz düzləndiriləndən sonra göz xətti üfüqi olmalıdır."""
    from src.detector import FaceDetector

    det = FaceDetector(cfg.detector)
    f = det.detect(face_bgr).face
    m = cv2.getRotationMatrix2D((f.x + f.w / 2, f.y + f.h / 2), -12, 1.0)
    rotated = cv2.warpAffine(face_bgr, m, (512, 512), borderMode=cv2.BORDER_REPLICATE)
    face = det.detect(rotated).face
    assert face is not None
    straight = face_crop(rotated, (face.x, face.y, face.w, face.h), face.landmarks, 0.3, True)
    padded = cv2.copyMakeBorder(straight, 100, 100, 100, 100, cv2.BORDER_REPLICATE)
    again = det.detect(padded).face
    assert again is not None
    assert abs(roll_degrees(again.landmarks)) < abs(roll_degrees(face.landmarks)) + 1.0


class Fake:
    def __init__(self):
        self.calls = []

    def predict(self, crop):
        self.calls.append(crop.copy())
        probs = np.zeros(8)
        probs[0 if len(self.calls) == 1 else 1] = 1.0
        return EmotionPrediction(probs, 0.2 * len(self.calls), -0.1, 5.0)


def test_flip_tta_averages_and_mirrors():
    fake = Fake()
    crop = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    p = predict(fake, crop, flip=True)
    assert len(fake.calls) == 2
    assert np.array_equal(fake.calls[1], cv2.flip(crop, 1))
    assert p.probs[0] == pytest.approx(0.5) and p.probs[1] == pytest.approx(0.5)
    assert p.valence == pytest.approx(0.3) and p.latency_ms >= 5.0


def test_no_flip_single_call():
    fake = Fake()
    predict(fake, np.zeros((4, 4, 3), np.uint8), flip=False)
    assert len(fake.calls) == 1


@pytest.mark.usefixtures("models_ready")
def test_live_pipeline_uses_config_variant(make_config, face_bgr):
    from src.app import build_analyzer

    from .conftest import face_frame

    cfg = make_config(compare__mode="off", detector__crop_margin=0.0, detector__align=True, model__flip_tta=True)
    a = build_analyzer(cfg)
    assert a.flip_tta is True
    r = a.process(face_frame(face_bgr), t=0.0)
    assert r.prediction is not None and r.model_ms > 0
    plain = build_analyzer(replace(cfg, model=replace(cfg.model, flip_tta=False)))
    assert plain.process(face_frame(face_bgr), t=0.0).model_ms < r.model_ms * 1.5 + 50
