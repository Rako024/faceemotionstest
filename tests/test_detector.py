from dataclasses import replace

import cv2
import numpy as np
import pytest

from src.config import DetectorConfig
from src.detector import Face, FaceDetector, FaceStatus, assess_face, region_detail


@pytest.fixture(scope="module")
def detector(cfg, yunet_path):
    return FaceDetector(cfg.detector)


@pytest.fixture(scope="module")
def ref_face(detector, face_bgr):
    det = detector.detect(face_bgr)
    assert det.ok
    return det.face


def _cover(img, face, y_from, y_to, color=(140, 170, 210)):
    out = img.copy()
    x, y, w, h = int(face.x), int(face.y), int(face.w), int(face.h)
    cv2.rectangle(out, (x - 5, y + int(h * y_from)), (x + w + 5, y + int(h * y_to)), color, -1)
    return out


# --- real YuNet ----------------------------------------------------------------


def test_detects_frontal_face(detector, face_bgr):
    det = detector.detect(face_bgr)
    assert det.status is FaceStatus.OK
    assert det.num_faces == 1
    f = det.face
    assert f.score >= 0.8
    assert f.landmarks.shape == (5, 2)
    h, w = face_bgr.shape[:2]
    assert 0 <= f.x and f.x + f.w <= w and 0 <= f.y and f.y + f.h <= h


def test_mirrored_face_still_ok(detector, face_bgr):
    assert detector.detect(cv2.flip(face_bgr, 1)).ok


@pytest.mark.parametrize(
    "frame",
    [
        np.full((480, 640, 3), 127, np.uint8),
        np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8),
    ],
    ids=["gray", "noise"],
)
def test_no_face(detector, frame):
    det = detector.detect(frame)
    assert det.status is FaceStatus.NOT_FOUND
    assert det.face is None and det.num_faces == 0


@pytest.mark.parametrize("level", [0, 14])
def test_closed_camera_reported_as_dark(detector, level):
    """Bağlı qapaq/kamera düyməsi: bircins qara kadr (ölçülmüş: orta ~14, std ~3)."""
    rng = np.random.default_rng(1)
    frame = np.clip(rng.normal(level, 3, (720, 1280, 3)), 0, 255).astype(np.uint8)
    assert detector.detect(frame).status is FaceStatus.CAMERA_DARK


def test_dim_room_with_face_is_not_dark(detector, face_bgr):
    dim = (face_bgr.astype(np.float32) * 0.35).astype(np.uint8)
    canvas = np.full((720, 1280, 3), 12, np.uint8)
    canvas[104:616, 384:896] = dim
    assert detector.detect(canvas).status is not FaceStatus.CAMERA_DARK


def test_picks_largest_face(detector, face_bgr):
    small = cv2.resize(face_bgr, None, fx=0.5, fy=0.5)
    canvas = np.full((600, 1000, 3), 120, np.uint8)
    canvas[40:552, 20:532] = face_bgr
    canvas[300 : 300 + small.shape[0], 700 : 700 + small.shape[1]] = small
    det = detector.detect(canvas)
    assert det.num_faces == 2
    assert det.ok
    assert det.face.x < 532, "kiçik (arxa plandakı) üz yox, böyük üz seçilməlidir"


def test_coordinates_mapped_back_after_downscale(detector, face_bgr, ref_face):
    big = cv2.resize(face_bgr, None, fx=2.5, fy=2.5)  # 1280px > max_input_side
    det = detector.detect(big)
    assert det.ok
    assert det.face.w == pytest.approx(ref_face.w * 2.5, rel=0.1)
    assert det.face.x == pytest.approx(ref_face.x * 2.5, rel=0.1)


def test_face_cut_by_frame_edge_is_partial(detector, face_bgr, ref_face):
    cut = face_bgr[:, int(ref_face.x + ref_face.w * 0.5) :].copy()
    det = detector.detect(cut)
    assert det.status in (FaceStatus.PARTIAL, FaceStatus.NOT_FOUND)
    assert not det.ok


def test_distant_face_too_small(detector, face_bgr):
    small = cv2.resize(face_bgr, None, fx=0.4, fy=0.4)
    canvas = np.full((720, 1280, 3), 128, np.uint8)
    canvas[200 : 200 + small.shape[0], 500 : 500 + small.shape[1]] = small
    assert detector.detect(canvas).status is FaceStatus.TOO_SMALL


@pytest.mark.parametrize(
    "y_from,y_to",
    [(0.5, 1.05), (0.6, 1.05), (0.25, 0.5)],
    ids=["hand-over-mouth", "lower-half", "eyes"],
)
def test_occluded_face_rejected(detector, face_bgr, ref_face, y_from, y_to):
    det = detector.detect(_cover(face_bgr, ref_face, y_from, y_to))
    assert not det.ok
    assert det.status in (FaceStatus.OCCLUDED, FaceStatus.NOT_FOUND)


def test_skin_toned_blurred_occluder_rejected(detector, face_bgr, ref_face):
    x, y, w, h = int(ref_face.x), int(ref_face.y), int(ref_face.w), int(ref_face.h)
    skin = face_bgr[y + int(h * 0.55) : y + int(h * 0.7), x + 5 : x + 25]
    skin = cv2.GaussianBlur(cv2.resize(skin, (w + 10, int(h * 0.45))), (0, 0), 3)
    img = face_bgr.copy()
    img[y + int(h * 0.55) : y + int(h * 0.55) + skin.shape[0], x - 5 : x - 5 + skin.shape[1]] = skin
    assert detector.detect(img).status is FaceStatus.OCCLUDED


def test_occlusion_check_can_be_disabled(cfg, yunet_path, face_bgr, ref_face):
    det = FaceDetector(replace(cfg.detector, min_region_detail=0.0))
    assert det.detect(_cover(face_bgr, ref_face, 0.5, 1.05)).ok


def test_region_detail_values(face_bgr, ref_face):
    detail = region_detail(face_bgr, ref_face)
    assert set(detail) == {"right_eye", "left_eye", "mouth"}
    assert min(detail.values()) > 0.8
    covered = region_detail(_cover(face_bgr, ref_face, 0.5, 1.05), ref_face)
    assert covered["mouth"] < 0.1


def test_low_confidence_largest_face_is_not_replaced(detector, face_bgr, ref_face):
    """Ən böyük üz pis görünürsə, arxadakı başqa üzə keçmirik."""
    with_bg = np.full((700, 1100, 3), 120, np.uint8)
    with_bg[100:612, 50:562] = face_bgr
    small = cv2.resize(face_bgr, None, fx=0.45, fy=0.45)
    with_bg[350 : 350 + small.shape[0], 800 : 800 + small.shape[1]] = small
    bad = with_bg.copy()
    x, y, w, h = int(ref_face.x) + 50, int(ref_face.y) + 100, int(ref_face.w), int(ref_face.h)
    cv2.rectangle(bad, (x - 5, y + int(h * 0.5)), (x + w + 5, y + h + 5), (140, 170, 210), -1)
    det = detector.detect(bad)
    if det.face is not None and det.face.x > 562:
        pytest.fail("böyük üz örtüləndə kiçik arxa üz seçildi")
    assert not det.ok


def test_crop_with_margin_and_clamp(detector, face_bgr, ref_face):
    crop = detector.crop(face_bgr, ref_face)
    m = detector.cfg.crop_margin
    assert crop.shape[1] == pytest.approx(ref_face.w * (1 + 2 * m), abs=3)
    assert crop.shape[0] == pytest.approx(ref_face.h * (1 + 2 * m), abs=3)
    edge = Face(-10, -10, 100, 100, 0.9, np.zeros((5, 2), np.float32))
    assert detector.crop(face_bgr, edge).shape[:2] == (100, 100)


def test_missing_model_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_models"):
        FaceDetector(DetectorConfig(model_path=tmp_path / "nope.onnx"))


# --- həndəsi yoxlamalar (sintetik landmark-lar) ---------------------------------

FRAME = (720, 1280, 3)


def _face(right_eye=(560, 330), left_eye=(640, 330), nose=(600, 370), score=0.95, box=(520, 270, 160, 200)):
    lm = np.array([right_eye, left_eye, nose, (570, 420), (630, 420)], np.float32)
    return Face(*box, score, lm)


def test_geometry_frontal_ok():
    assert assess_face(_face(), FRAME, DetectorConfig()) is FaceStatus.OK


def test_geometry_low_confidence():
    assert assess_face(_face(score=0.79), FRAME, DetectorConfig()) is FaceStatus.LOW_CONFIDENCE


def test_geometry_profile_by_nose_offset():
    assert assess_face(_face(nose=(640, 370)), FRAME, DetectorConfig()) is FaceStatus.PROFILE


def test_geometry_tilted_head():
    # göz xətti ~40° əyik
    f = _face(right_eye=(560, 310), left_eye=(621, 361), nose=(585, 360))
    assert assess_face(f, FRAME, DetectorConfig()) is FaceStatus.TILTED


def test_geometry_box_touching_edge_is_partial():
    f = _face(box=(1, 270, 160, 200))
    assert assess_face(f, FRAME, DetectorConfig()) is FaceStatus.PARTIAL


def test_geometry_landmarks_outside_box_is_partial():
    f = _face(nose=(600, 600))
    assert assess_face(f, FRAME, DetectorConfig()) is FaceStatus.PARTIAL


def test_geometry_too_small():
    scale = 0.3  # eyni üz, 160×200 → 48×60 px
    lm = (_face().landmarks - np.array([520, 270])) * scale + np.array([600, 300])
    f = Face(600, 300, 160 * scale, 200 * scale, 0.95, lm.astype(np.float32))
    assert assess_face(f, FRAME, DetectorConfig()) is FaceStatus.TOO_SMALL
