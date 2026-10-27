"""Paneldə canlı geyim yoxlaması: kamera axınından bir neçə kadr → səs çoxluğu → "Geyim" kartı.

Emosiya pipeline-ına toxunmur: kadrlar mənbənin yaddaşdakı son kadrından (surəti) götürülür,
öz üz detektoru ilə işlənir. Model ilk yoxlamada fonda yüklənir — torch/transformers yalnız
onda import olunur. Nəticə yalnız ekranda göstərilir: kadr, kəsik və nəticə diskə yazılmır.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import Counter
from typing import Any, Callable

import numpy as np

from ..config import AppConfig
from .crop import find_torso
from .encoder import Models
from .verdict import FORMAL, NOT_FORMAL, UNKNOWN, Verdict, decide, label, split_scores

log = logging.getLogger(__name__)

FIRST_FRAME_WAIT_S = 5.0  # kamera yenicə açılıbsa, ilk kadrı bu qədər gözlə
ACCESSORY_P = 0.5

# Qiymətləndirilə bilməyəndə operatora nə etməli olduğunu deyən ipucu
HINTS = {
    "torso_hidden": "Kameranı bir az aşağı əyin və ya bir az geri oturun — yaxa görünməlidir",
    "no_face": "Üzünüz kadrda olmalıdır",
    "too_small": "Kameraya bir az yaxınlaşın",
    "dark": "İşığı artırın",
    "blurry": "Bir neçə saniyə tərpənmədən oturun",
    "no_frame": "Kameradan görüntü gəlmir",
}


def load_live_models(cfg: AppConfig) -> Models:
    """SigLIP 2 + mətn vektorları; sonra mətn hissəsi yaddaşdan çıxarılır."""
    from ..detector import FaceDetector
    from .encoder import SiglipEncoder, ZeroShot

    encoder = SiglipEncoder(cfg.attire.model_dir, threads=cfg.attire.live.threads)
    zero_shot = ZeroShot(encoder, cfg.attire.prompts)
    encoder.drop_text_model()
    return Models(FaceDetector(cfg.detector), encoder, zero_shot)


def aggregate(verdicts: list[Verdict], total: int) -> tuple[Verdict, int]:
    """Səs çoxluğu. Kadrların yarısından azı qiymətləndirilə bilirsə — "qiymətləndirilə bilmir".

    Qaytarır: (ümumi nəticə, onu dəstəkləyən kadr sayı).
    """
    ok = [v for v in verdicts if v.result != UNKNOWN]
    if total == 0 or len(ok) < math.ceil(total / 2):
        unknown = [v for v in verdicts if v.result == UNKNOWN]
        if not unknown:
            return Verdict(UNKNOWN, "kameradan görüntü gəlmir", "no_frame"), 0
        torso, n = Counter(v.torso for v in unknown).most_common(1)[0]
        reason = next(v.reason for v in unknown if v.torso == torso)
        return Verdict(UNKNOWN, reason, torso), n
    formal = [v for v in ok if v.result == FORMAL]
    winners = formal if len(formal) > len(ok) / 2 else [v for v in ok if v.result == NOT_FORMAL]
    mean = lambda attr: float(np.mean([getattr(v, attr) for v in ok]))  # noqa: E731
    garments, colors = _mean_probs([v.garment_probs for v in ok]), _mean_probs([v.color_probs for v in ok])
    most = lambda attr, vs: Counter(getattr(v, attr) for v in vs).most_common(1)[0][0]  # noqa: E731
    return Verdict(
        result=winners[0].result, reason=most("reason", winners), torso="ok",
        garment=max(garments, key=garments.get) if garments else most("garment", ok), formal_p=mean("formal_p"),
        color=max(colors, key=colors.get) if colors else most("color", ok), white_p=mean("white_p"),
        white_share=mean("white_share"), tie_p=mean("tie_p"), jacket_p=mean("jacket_p"),
        garment_ok=sum(bool(v.garment_ok) for v in ok) > len(ok) / 2,
        jacket_ok=sum(bool(v.jacket_ok) for v in ok) > len(ok) / 2,
        jacket_seen=sum(bool(v.jacket_seen) for v in ok) > len(ok) / 2,
        layering=_mean_opt([v.layering for v in ok]),
        basis=most("basis", winners), garment_probs=garments or None, color_probs=colors or None,
    ), len(winners)


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def _mean_opt(values: list[float | None]) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _mean_probs(dists: list[dict[str, float] | None]) -> dict[str, float]:
    dists = [d for d in dists if d]
    if not dists:
        return {}
    return {k: float(np.mean([d.get(k, 0.0) for d in dists])) for k in dists[0]}


def _ranked(probs: dict[str, float] | None, top: int | None = None) -> list[dict[str, Any]]:
    items = sorted((probs or {}).items(), key=lambda kv: -kv[1])[:top]
    return [{"key": k, "label": label(k), "p": round(p, 3)} for k, p in items]


class AttireChecker:
    """Bir yoxlama = config-dəki sayda kadr. Eyni anda yalnız bir yoxlama gedir."""

    def __init__(self, cfg: AppConfig, frame_source: Callable[[], np.ndarray | None],
                 on_change: Callable[[], None] = lambda: None,
                 loader: Callable[[AppConfig], Models] = load_live_models) -> None:
        self.cfg = cfg
        self._frame = frame_source
        self._on_change = on_change
        self._loader = loader
        self._models: Models | None = None
        self._labels: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self.status = "idle"  # idle | loading | checking | done | error
        self.progress = 0
        self.verdict: Verdict | None = None
        self.votes = 0
        self.error: str | None = None
        self.checked_at: float | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        with self._lock:
            if self.busy:
                return False
            self._cancel.clear()
            self._thread = threading.Thread(target=self._run, name="AttireCheck", daemon=True)
            self._thread.start()
        return True

    def cancel(self) -> None:
        """Gedən yoxlamanı dayandırır (kamera bağlananda). Əvvəlki nəticə qalır."""
        self._cancel.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def _set(self, status: str, progress: int = 0) -> None:
        self.status, self.progress = status, progress
        try:
            self._on_change()
        except Exception:  # noqa: BLE001
            log.exception("on_change callback xətası")

    def _analyze(self, frame: np.ndarray) -> Verdict:
        m = self._models
        torso = find_torso(frame, m.detector, self.cfg.attire)
        scores = None
        if torso.ok:
            emb = m.encoder.encode_images([torso.crop])
            scores = split_scores(m.zero_shot(emb), self._labels)[0]
        return decide(torso, scores, self.cfg.attire)

    def _wait_first_frame(self) -> None:
        deadline = time.monotonic() + FIRST_FRAME_WAIT_S
        while self._frame() is None and time.monotonic() < deadline and not self._cancel.wait(0.1):
            pass

    def _run(self) -> None:
        live = self.cfg.attire.live
        try:
            if self._models is None:
                self._set("loading")
                started = time.monotonic()
                self._models = self._loader(self.cfg)
                self._labels = {q: zq.labels for q, zq in self._models.zero_shot.questions.items()}
                log.info("Geyim modeli yükləndi: %.1f s", time.monotonic() - started)
            if self._cancel.is_set():
                self._set("done" if self.verdict else "idle")
                return
            self._set("checking", 0)
            self._wait_first_frame()
            verdicts = []
            for i in range(live.frames):
                if self._cancel.is_set():
                    self._set("done" if self.verdict else "idle")
                    return
                frame = self._frame()
                verdicts.append(self._analyze(frame) if frame is not None
                                else Verdict(UNKNOWN, "kameradan görüntü gəlmir", "no_frame"))
                self._set("checking", i + 1)
                if i + 1 < live.frames:
                    self._cancel.wait(live.interval_s)
            self.verdict, self.votes = aggregate(verdicts, live.frames)
            self.checked_at = time.time()
            self.error = None
            v = self.verdict
            log.info("Geyim: %s (%d/%d) %s | növ=%s köynək=%s pencək=%s qat=%s rəng=%s", v.result, self.votes,
                     live.frames, v.basis or v.reason, v.garment or "-", _fmt(v.formal_p), _fmt(v.jacket_p),
                     _fmt(v.layering), v.color or "-")
            self._set("done")
        except Exception as exc:  # noqa: BLE001 - paneldə göstərilir
            log.exception("Geyim yoxlaması alınmadı")
            self.error = str(exc)
            self._set("error")

    def state(self) -> dict[str, Any]:
        live = self.cfg.attire.live
        out: dict[str, Any] = {"status": self.status, "progress": self.progress, "frames": live.frames,
                               "loaded": self._models is not None, "error": self.error, "result": None}
        v = self.verdict
        if v is not None:
            r2 = lambda x: None if x is None else round(x, 3)  # noqa: E731
            out["result"] = {
                "result": v.result,
                "label": label(v.result),
                "reason": v.reason,
                "basis": v.basis,
                "hint": HINTS.get(v.torso) if v.result == UNKNOWN else None,
                "garment": label(v.garment) if v.garment else None,
                "color": label(v.color) if v.color else None,
                "garments": [{**g, "formal": g["key"] in self.cfg.attire.formal_garments}
                             for g in _ranked(v.garment_probs)],  # bütün geyim növləri, ehtimala görə
                "colors": _ranked(v.color_probs, 3),
                "tie": v.tie_p is not None and v.tie_p >= ACCESSORY_P,
                "jacket": bool(v.jacket_seen),
                # model "pencək" dedi, amma iki qat görünmədi (məs. qara köynək) — kartda izah olunur
                "jacket_rejected": bool(v.jacket_p is not None and v.jacket_p >= self.cfg.attire.jacket_threshold
                                        and not v.jacket_seen),
                "layering": r2(v.layering),
                "tie_p": r2(v.tie_p),
                "jacket_p": r2(v.jacket_p),
                "formal_p": r2(v.formal_p),
                "white_p": r2(v.white_p),
                "votes": self.votes,
                "age_s": None if self.checked_at is None else round(time.time() - self.checked_at),
            }
        return out
