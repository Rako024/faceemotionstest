"""Qərar qaydası: rəsmi = düyməli köynək (attire.formal_garments) VƏ YA pencək (attire.jacket_is_formal).

Rəng standart olaraq şərt deyil — boz, mavi, naxışlı düyməli köynək də rəsmidir. attire.require_white
yandırılsa, yalnız ağ rəsmi sayılır; onda rəng iki müstəqil siqnalla yoxlanılır: SigLIP-in "white"
cavabı və yaxa-döş zonasında ağ piksellərin payı (isti lampa altında ağ köynək sarımtıl görünə
bilər — ona görə hər iki siqnal ayrıca yazılır). Qalstuk yalnız məlumat üçündür.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import AttireConfig
from ..labels import ATTIRE_LABELS
from .crop import Torso

FORMAL, NOT_FORMAL, UNKNOWN = "formal", "not_formal", "unknown"

Scores = dict[str, dict[str, float]]  # sual → cavab → ehtimal (bir şəkil üçün)


def label(key: str) -> str:
    return ATTIRE_LABELS.get(key, key)


def white_share(region_bgr: np.ndarray, min_lightness: float, max_chroma: float) -> float:
    """Ağ piksellərin payı: açıq (L*) və az doyğun (a*, b* sıfıra yaxın)."""
    if region_bgr is None or region_bgr.size == 0:
        return 0.0
    lab = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = lab[..., 0] * (100.0 / 255.0)  # OpenCV 8-bit: L 0–255, a/b 128 ətrafında
    chroma = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
    return float(((lightness >= min_lightness) & (chroma <= max_chroma)).mean())


def color_ok(model_white: bool, pixel_white: bool, rule: str) -> bool:
    """color_rule: both → iki siqnal da ağ, model / pixels → yalnız biri."""
    return (model_white or rule == "pixels") and (pixel_white or rule == "model")


def split_scores(probs: dict[str, np.ndarray], labels: dict[str, list[str]]) -> list[Scores]:
    """ZeroShot nəticəsi (sual → (N, L)) → hər şəkil üçün sual → cavab → ehtimal."""
    n = len(next(iter(probs.values())))
    return [{q: dict(zip(labels[q], map(float, p[i]))) for q, p in probs.items()} for i in range(n)]


@dataclass
class Signals:
    shirt: bool  # rəsmi geyim növü (düyməli köynək)
    jacket_seen: bool  # pencək görünür: model "var" deyir VƏ iki qat görünür
    jacket: bool  # pencək rəsmi sayılır (jacket_is_formal və jacket_seen)
    model_white: bool
    pixel_white: bool
    formal: bool


def jacket_seen(jacket_p: float, layering: float | None, cfg: AttireConfig) -> bool:
    """Qara/tünd köynək modelə tünd pencək kimi görünə bilər — pencək üçün iki qat da görünməlidir."""
    layered = layering is None or cfg.jacket_min_layering <= 0 or layering >= cfg.jacket_min_layering
    return jacket_p >= cfg.jacket_threshold and layered


def signals(formal_p: float, jacket_p: float, white_p: float, share: float, cfg: AttireConfig,
            layering: float | None = None) -> Signals:
    """Qərarın yeganə yeri — həm canlı/oflayn nəticə, həm də hədlərin kalibrlənməsi bunu çağırır."""
    shirt = formal_p >= cfg.formal_threshold
    seen = jacket_seen(jacket_p, layering, cfg)
    jacket = cfg.jacket_is_formal and seen
    model_white, pixel_white = white_p >= cfg.white_threshold, share >= cfg.white_min_share
    white = not cfg.require_white or color_ok(model_white, pixel_white, cfg.color_rule)
    return Signals(shirt, seen, jacket, model_white, pixel_white, (shirt or jacket) and white)


@dataclass
class Verdict:
    result: str  # formal | not_formal | unknown
    reason: str = ""  # qeyri-rəsmi / qiymətləndirilə bilmir — niyə
    torso: str = "ok"
    garment: str = ""  # ən ehtimallı geyim növü
    formal_p: float | None = None  # rəsmi növlərin cəmi ehtimalı
    color: str = ""  # ən ehtimallı rəng
    white_p: float | None = None
    white_share: float | None = None
    tie_p: float | None = None
    jacket_p: float | None = None
    garment_ok: bool | None = None
    model_white: bool | None = None
    pixel_white: bool | None = None
    jacket_ok: bool | None = None  # pencək rəsmi sayıldı
    jacket_seen: bool | None = None  # pencək görünür (model + iki qat)
    layering: float | None = None
    basis: str = ""  # rəsmi — nəyə görə: "düyməli köynək", "pencək", "düyməli köynək + pencək"
    garment_probs: dict[str, float] | None = None
    color_probs: dict[str, float] | None = None

    @property
    def text(self) -> str:
        return label(self.result)


def is_formal(v: Verdict, cfg: AttireConfig) -> bool:
    """Saxlanılmış ballardan qərarı başqa hədlərlə yenidən hesablayır (kalibrləmə)."""
    return signals(v.formal_p, v.jacket_p, v.white_p, v.white_share, cfg, v.layering).formal


def explain(s: Signals, top_garment: str, top_color: str, formal_p: float, share: float,
            cfg: AttireConfig) -> tuple[str, str]:
    """(basis, reason): rəsmidirsə nəyə görə, deyilsə niyə."""
    basis = " + ".join(part for ok, part in ((s.shirt, label("shirt")), (s.jacket, "pencək")) if ok)
    reasons = []
    if not (s.shirt or s.jacket):
        what = "düyməli köynək və ya pencək" if cfg.jacket_is_formal else "düyməli köynək"
        if top_garment in cfg.formal_garments:
            reasons.append(f"{label(top_garment)} olduğu dəqiq deyil ({formal_p:.0%})")
        else:
            reasons.append(f"{what} görünmür — {label(top_garment)}")
    if cfg.require_white:
        if cfg.color_rule != "pixels" and not s.model_white:
            reasons.append(f"ağ deyil — {label(top_color)}")
        if cfg.color_rule != "model" and not s.pixel_white:
            reasons.append(f"yaxa zonasında ağ az görünür ({share:.0%})")
    return basis, "; ".join(reasons)


def decide(torso: Torso, scores: Scores | None, cfg: AttireConfig) -> Verdict:
    if not torso.ok or scores is None:
        return Verdict(UNKNOWN, torso.reason, torso.status.value)
    garment, color = scores["garment"], scores["color"]
    top_garment = max(garment, key=garment.get)
    top_color = max(color, key=color.get)
    formal_p = sum(garment[g] for g in cfg.formal_garments)
    jacket_p = scores["jacket"]["present"]
    share = white_share(torso.collar, cfg.white_min_lightness, cfg.white_max_chroma)
    s = signals(formal_p, jacket_p, color["white"], share, cfg, torso.layering)
    basis, reason = explain(s, top_garment, top_color, formal_p, share, cfg)
    return Verdict(
        result=FORMAL if s.formal else NOT_FORMAL, reason=reason, torso=torso.status.value,
        garment=top_garment, formal_p=formal_p, color=top_color, white_p=color["white"], white_share=share,
        tie_p=scores["tie"]["present"], jacket_p=jacket_p, garment_ok=s.shirt, model_white=s.model_white,
        pixel_white=s.pixel_white, jacket_ok=s.jacket, jacket_seen=s.jacket_seen, layering=torso.layering,
        basis=basis if s.formal else "", garment_probs=dict(garment), color_probs=dict(color),
    )
