"""OpenCV YuNet ilə üz aşkarlama və keyfiyyət filtri.

Qayda: yalnız ən böyük üz götürülür. Üz tam görünmürsə (aşağı confidence,
kadrın kənarında kəsik, çox kiçik, yanakı, əyik, göz/ağız örtülü) nəticə
"üz yoxdur" sayılır və model işə salınmır — təxmin etmirik.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .config import DetectorConfig

# Aşkarlama həddi: 0.8-dən aşağı olanları da tapırıq ki, "aşağı confidence"
# səbəbini "üz tapılmadı"dan ayıra bilək. Qəbul həddi isə config-dəkidir.
_CANDIDATE_THRESHOLD = 0.5


class FaceStatus(str, Enum):
    OK = "ok"
    NOT_FOUND = "not_found"
    LOW_CONFIDENCE = "low_confidence"
    PARTIAL = "partial"
    TOO_SMALL = "too_small"
    PROFILE = "profile"
    TILTED = "tilted"
    OCCLUDED = "occluded"
    CAMERA_DARK = "camera_dark"  # kadr tam qaradır: qapaq/məxfilik düyməsi bağlıdır


@dataclass(frozen=True)
class Face:
    """Orijinal kadr koordinatlarında üz."""

    x: float
    y: float
    w: float
    h: float
    score: float
    # YuNet sırası: sağ göz, sol göz, burun ucu, ağzın sağ küncü, ağzın sol küncü
    landmarks: np.ndarray  # (5, 2)

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass(frozen=True)
class Detection:
    status: FaceStatus
    face: Face | None
    num_faces: int

    @property
    def ok(self) -> bool:
        return self.status is FaceStatus.OK


def assess_face(face: Face, frame_shape: tuple[int, ...], cfg: DetectorConfig) -> FaceStatus:
    """Seçilmiş üzün modelə verilə biləcək keyfiyyətdə olub-olmadığını yoxlayır."""
    if face.score < cfg.score_threshold:
        return FaceStatus.LOW_CONFIDENCE

    height, width = frame_shape[:2]
    m = cfg.edge_margin_px
    if (
        face.x < m
        or face.y < m
        or face.x + face.w > width - m
        or face.y + face.h > height - m
    ):
        return FaceStatus.PARTIAL

    lm = face.landmarks
    pad_x, pad_y = 0.1 * face.w, 0.1 * face.h
    inside = (
        (lm[:, 0] >= face.x - pad_x)
        & (lm[:, 0] <= face.x + face.w + pad_x)
        & (lm[:, 1] >= face.y - pad_y)
        & (lm[:, 1] <= face.y + face.h + pad_y)
    )
    if not bool(inside.all()):
        return FaceStatus.PARTIAL

    if min(face.w, face.h) < cfg.min_face_px:
        return FaceStatus.TOO_SMALL

    right_eye, left_eye, nose = lm[0], lm[1], lm[2]
    eye_vec = left_eye - right_eye
    interocular = float(np.hypot(*eye_vec))
    if interocular < 1e-3:
        return FaceStatus.PROFILE

    # Burnun göz ortasından göz xətti boyunca sapması: düz baxanda ~0, yanakı ~0.5+
    axis = eye_vec / interocular
    yaw_ratio = abs(float(np.dot(nose - (right_eye + left_eye) / 2.0, axis))) / interocular
    if yaw_ratio > cfg.max_yaw_ratio:
        return FaceStatus.PROFILE

    roll = math.degrees(math.atan2(float(eye_vec[1]), abs(float(eye_vec[0]))))
    if abs(roll) > cfg.max_roll_deg:
        return FaceStatus.TILTED

    return FaceStatus.OK


# Bağlı kamera (məxfilik qapağı, klaviaturadakı kamera düyməsi) demək olar ki bircins qara kadr verir:
# ölçülmüş HP HD Camera: orta parlaqlıq ~14/255, std ~3. Qaranlıq otaqda belə üzlü kadrın std-si daha böyükdür.
DARK_MEAN_MAX = 25.0
DARK_STD_MAX = 6.0


def is_dark_frame(frame_bgr: np.ndarray) -> bool:
    small = cv2.resize(frame_bgr, (64, 36), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < DARK_MEAN_MAX and float(gray.std()) < DARK_STD_MAX


# Landmark ətrafındakı sahələr: (yarım en, yarım hündürlük) göz arası məsafəyə nisbətən
_DETAIL_REGIONS = {
    "right_eye": (0.30, 0.22),
    "left_eye": (0.30, 0.22),
    "mouth": (0.60, 0.25),
}


def region_detail(frame_bgr: np.ndarray, face: Face) -> dict[str, float]:
    """Göz və ağız ətrafında gradient sıxlığının bütün üzün ortalamasına nisbəti.

    Açıq üzdə göz ~1.5, ağız ~1.1 olur. Əl, mikrofon və ya əşya ilə örtüləndə
    sahə hamarlaşır və nisbət kəskin düşür (~0–0.4). YuNet belə üzləri hələ də
    yüksək confidence ilə tapdığı üçün örtülmə yalnız bu yolla tutulur.
    """
    height, width = frame_bgr.shape[:2]
    x0, y0 = max(0, int(face.x)), max(0, int(face.y))
    x1, y1 = min(width, int(face.x + face.w)), min(height, int(face.y + face.h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return {name: 0.0 for name in _DETAIL_REGIONS}
    gray = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    base = float(mag.mean())
    if base < 1e-6:
        return {name: 0.0 for name in _DETAIL_REGIONS}

    lm = face.landmarks
    interocular = float(np.hypot(*(lm[1] - lm[0])))
    centers = {"right_eye": lm[0], "left_eye": lm[1], "mouth": (lm[3] + lm[4]) / 2.0}
    out = {}
    for name, (half_w, half_h) in _DETAIL_REGIONS.items():
        cx, cy = float(centers[name][0]) - x0, float(centers[name][1]) - y0
        a0 = max(0, int(cx - half_w * interocular))
        a1 = min(mag.shape[1], int(cx + half_w * interocular) + 1)
        b0 = max(0, int(cy - half_h * interocular))
        b1 = min(mag.shape[0], int(cy + half_h * interocular) + 1)
        patch = mag[b0:b1, a0:a1]
        out[name] = float(patch.mean()) / base if patch.size else 0.0
    return out


class FaceDetector:
    def __init__(self, cfg: DetectorConfig) -> None:
        if not cfg.model_path.is_file():
            raise FileNotFoundError(
                f"YuNet modeli tapılmadı: {cfg.model_path}. "
                "Əvvəlcə 'python scripts/download_models.py' işə salın."
            )
        self.cfg = cfg
        self._net = cv2.FaceDetectorYN.create(
            str(cfg.model_path),
            "",
            (320, 320),
            min(_CANDIDATE_THRESHOLD, cfg.score_threshold),
            cfg.nms_threshold,
            50,
        )
        self._input_size: tuple[int, int] | None = None

    def _faces(self, frame_bgr: np.ndarray) -> list[Face]:
        height, width = frame_bgr.shape[:2]
        scale = min(1.0, self.cfg.max_input_side / max(height, width))
        if scale < 1.0:
            small = cv2.resize(
                frame_bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
            )
        else:
            small = frame_bgr
        size = (small.shape[1], small.shape[0])
        if size != self._input_size:
            self._net.setInputSize(size)
            self._input_size = size

        _, raw = self._net.detect(small)
        if raw is None:
            return []
        faces = []
        for row in raw:
            x, y, w, h = (float(v) / scale for v in row[:4])
            landmarks = (row[4:14].reshape(5, 2) / scale).astype(np.float32)
            faces.append(Face(x, y, w, h, float(row[14]), landmarks))
        return faces

    def detect(self, frame_bgr: np.ndarray) -> Detection:
        faces = self._faces(frame_bgr)
        if not faces:
            status = FaceStatus.CAMERA_DARK if is_dark_frame(frame_bgr) else FaceStatus.NOT_FOUND
            return Detection(status, None, 0)
        # Ən böyük üz — operatorun özü. Arxadakı kiçik üzlərə keçmirik, hətta
        # ən böyük üz keyfiyyət yoxlamasından keçməsə belə.
        largest = max(faces, key=lambda f: f.area)
        status = assess_face(largest, frame_bgr.shape, self.cfg)
        if status is FaceStatus.OK and self.cfg.min_region_detail > 0:
            detail = region_detail(frame_bgr, largest)
            if min(detail.values()) < self.cfg.min_region_detail:
                status = FaceStatus.OCCLUDED
        return Detection(status, largest, len(faces))

    def crop(self, frame_bgr: np.ndarray, face: Face) -> np.ndarray:
        """Üzü crop_margin qədər genişləndirilmiş, kadr sərhədinə sıxılmış (istəyə görə düzləndirilmiş) kəsik."""
        from .variants import face_crop

        return face_crop(frame_bgr, (face.x, face.y, face.w, face.h), face.landmarks,
                         self.cfg.crop_margin, self.cfg.align)
