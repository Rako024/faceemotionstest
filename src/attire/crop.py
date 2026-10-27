"""Gövdə kəsiyi, keyfiyyət filtri və pencəyin "iki qat" əlaməti.

Üz modelə verilmir — kəsik çənədən aşağı başlayır, yalnız geyim görünür. Ölçülər üz qutusuna
nisbətəndir, ona görə kameradan məsafə fərqli olsa da kəsik bədənin eyni hissəsini tutur.
Gövdə kadra düşmürsə, kiçik, qaranlıq və ya bulanıqdırsa, nəticə "qiymətləndirilə bilmir" olur —
model təxmin etmir. İşıq və kəskinlik ÜZDƏ ölçülür: qara və ya düz rəngli paltar "qaranlıq" və
"bulanıq" sayılmasın.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from ..config import AttireConfig
from ..detector import FaceDetector, FaceStatus

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 — kadr koordinatlarında
SHARPNESS_WIDTH = 128  # kəskinlik bu enə gətirilmiş üzdə ölçülür — ayırdetmədən asılı olmasın

# Pencək = iki qat: döşün ortasında (yaxaların arası) köynək, yanlarda pencəyin parçası.
# Zonalar üz ölçüsünə nisbətəndir (en → üzün eni, hündürlük → üzün hündürlüyü, çənədən aşağı).
LAYER_CENTER = 0.25  # orta zolağın yarım-eni
LAYER_SIDES = (0.7, 1.25)  # yan zonalar: mərkəzdən bu məsafələr arasında
LAYER_ROWS = (0.35, 1.1)
LAYER_DELTA_E = 20.0  # orta zolaqda pikselin yan rəngdən "fərqli" sayılması (CIE76 ΔE)


class TorsoStatus(str, Enum):
    OK = "ok"
    NO_FACE = "no_face"
    TORSO_HIDDEN = "torso_hidden"
    TOO_SMALL = "too_small"
    DARK = "dark"
    BLURRY = "blurry"


STATUS_TEXT = {
    TorsoStatus.OK: "",
    TorsoStatus.NO_FACE: "üz tapılmadı — gövdənin yeri bilinmir",
    TorsoStatus.TORSO_HIDDEN: "gövdə kadra kifayət qədər düşmür",
    TorsoStatus.TOO_SMALL: "gövdə kəsiyi çox kiçikdir",
    TorsoStatus.DARK: "işıq azdır — üz qaranlıq görünür",
    TorsoStatus.BLURRY: "şəkil bulanıqdır",
}


@dataclass
class Torso:
    status: TorsoStatus
    crop: np.ndarray | None = None  # modelə veriləcək kvadrat kəsik (boz kənarlı)
    collar: np.ndarray | None = None  # rəngi ölçülən yaxa-döş zonası
    box: Box | None = None
    visible: float = 0.0  # çənədən aşağı kadra düşən hündürlük, üzün hündürlüyü ilə
    brightness: float = 0.0  # üzün
    sharpness: float = 0.0  # üzün
    layering: float | None = None  # pencəyin "iki qat" əlaməti, 0–1; None → yan zonalar kadrda deyil

    @property
    def ok(self) -> bool:
        return self.status is TorsoStatus.OK

    @property
    def reason(self) -> str:
        return STATUS_TEXT[self.status]


def _clip(x0: float, y0: float, x1: float, y1: float, shape: tuple[int, ...]) -> Box:
    height, width = shape[:2]
    return (max(0, math.floor(x0)), max(0, math.floor(y0)), min(width, math.ceil(x1)), min(height, math.ceil(y1)))


def torso_box(face: tuple[float, float, float, float], shape: tuple[int, ...], cfg: AttireConfig) -> tuple[Box, float]:
    """Gövdə qutusu (kadra sıxılmış) və çənədən aşağı görünən hündürlük (üz hündürlüyü ilə)."""
    x, y, w, h = face
    cx, chin = x + w / 2.0, y + h
    box = _clip(cx - cfg.torso_width * w / 2.0, chin, cx + cfg.torso_width * w / 2.0, chin + cfg.torso_height * h, shape)
    visible = max(0.0, (box[3] - chin) / h) if h > 0 else 0.0
    return box, visible


def collar_box(face: tuple[float, float, float, float], shape: tuple[int, ...], cfg: AttireConfig) -> Box:
    """Yaxa-döş zonası: çənənin düz altı, köynəyin pencək və qalstukla belə görünən hissəsi."""
    x, y, w, h = face
    cx, top = x + w / 2.0, y + h + cfg.collar_top * h
    return _clip(cx - cfg.collar_width * w / 2.0, top, cx + cfg.collar_width * w / 2.0, top + cfg.collar_height * h,
                 shape)


def letterbox(image: np.ndarray, fill: int = 128) -> np.ndarray:
    """Kvadrata tamamlayır — model girişi kvadratdır, sıxılma geyimin formasını təhrif etməsin."""
    h, w = image.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(image, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT,
                              value=(fill, fill, fill))


def brightness(image: np.ndarray) -> float:
    return float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())


def sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = SHARPNESS_WIDTH / gray.shape[1]
    gray = cv2.resize(gray, (SHARPNESS_WIDTH, max(1, round(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _lab(image: np.ndarray) -> np.ndarray:
    """OpenCV 8-bit Lab → L* 0–100, a*/b* sıfır ətrafında (ΔE üçün)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def layering(frame: np.ndarray, face: tuple[float, float, float, float]) -> float | None:
    """Pencəyin "iki qat" əlaməti: orta zolağın neçə payı HƏR İKİ yan zonanın rəngindən fərqlənir.

    Pencək + köynək: ortada köynək (və qalstuk), yanlarda pencək → yüksək. Pencəksiz köynək (qara da
    olsa): orta və yanlar eyni parça → ~0. Kiçik naxışlar bulanıqlaşdırma ilə ortalanır. Yan zonalar
    kadra düşmürsə — None (yoxlamaq olmur, model ehtimalına etibar edilir).
    """
    x, y, w, h = face
    cx, chin = x + w / 2.0, y + h
    height, width = frame.shape[:2]
    y0, y1 = round(chin + LAYER_ROWS[0] * h), min(height, round(chin + LAYER_ROWS[1] * h))
    left, right = round(cx - LAYER_SIDES[1] * w), round(cx + LAYER_SIDES[1] * w)
    if y1 - y0 < 0.3 * h or left < 0 or right > width:
        return None
    k = max(3, int(w / 8) | 1)
    band = _lab(cv2.blur(frame[y0:y1, left:right], (k, k)))
    off = lambda d: round((LAYER_SIDES[1] - d) * w)  # noqa: E731 - mərkəzdən d qədər soldakı sütun (lentdə)
    centre = band[:, off(LAYER_CENTER) : off(-LAYER_CENTER)].reshape(-1, 3)
    shares = []
    for side in (band[:, : off(LAYER_SIDES[0])], band[:, off(-LAYER_SIDES[0]) :]):
        colour = np.median(side.reshape(-1, 3), axis=0)
        shares.append(float((np.linalg.norm(centre - colour, axis=1) > LAYER_DELTA_E).mean()))
    return min(shares)


def extract_torso(frame: np.ndarray, face: tuple[float, float, float, float], cfg: AttireConfig) -> Torso:
    box, visible = torso_box(face, frame.shape, cfg)
    if visible < cfg.min_visible:
        return Torso(TorsoStatus.TORSO_HIDDEN, box=box, visible=visible)
    x0, y0, x1, y1 = box
    region = frame[y0:y1, x0:x1]
    if min(region.shape[:2]) == 0 or region.shape[1] < cfg.min_torso_px:
        return Torso(TorsoStatus.TOO_SMALL, box=box, visible=visible)
    fx0, fy0, fx1, fy1 = _clip(face[0], face[1], face[0] + face[2], face[1] + face[3], frame.shape)
    face_img = frame[fy0:fy1, fx0:fx1]
    bright, sharp = brightness(face_img), sharpness(face_img)
    status = TorsoStatus.OK
    if bright < cfg.min_brightness:
        status = TorsoStatus.DARK
    elif sharp < cfg.min_sharpness:
        status = TorsoStatus.BLURRY
    cx0, cy0, cx1, cy1 = collar_box(face, frame.shape, cfg)
    return Torso(status, letterbox(region), frame[cy0:cy1, cx0:cx1], box, visible, bright, sharp,
                 layering(frame, face))


def find_torso(frame: np.ndarray, detector: FaceDetector, cfg: AttireConfig) -> Torso:
    """Ən böyük üzün altındakı gövdə. Üzün keyfiyyəti (bucaq, örtülmə) burada vacib deyil — yalnız yeri."""
    detection = detector.detect(frame)
    if detection.face is None:
        status = TorsoStatus.DARK if detection.status is FaceStatus.CAMERA_DARK else TorsoStatus.NO_FACE
        return Torso(status)
    f = detection.face
    return extract_torso(frame, (f.x, f.y, f.w, f.h), cfg)
