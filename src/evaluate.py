"""Oflayn qiymətləndirici: saxlanılmış nəzarətli test şəkillərində model × ayar müqayisəsi.

    python -m src.evaluate                                   # quraşdırılmış bütün modellər, standart ayar şəbəkəsi
    python -m src.evaluate --models enet_b0_8_best_afew,enet_b0_8_va_mtl --margins 0,0.1,0.2
    python -m src.evaluate --align off --flip both --testers T-01,T-02 --out reports/x.html

Ayarlar (bax src/variants.py — canlı rejimdə eyni funksiyalar işləyir):
    boşluq (detector.crop_margin), düzləndirmə (detector.align), flip TTA (model.flip_tta)

Üz yenidən aşkarlanmır: test zamanı YuNet-in tapdığı qutu və 5 nöqtə frames.csv-dədir, kəsik
onlardan qurulur — fərq yalnız model və ayardan gəlir. Proqnozlar sessiya qovluğunda keşlənir
(.cache/, yalnız rəqəmlər); yeni tester əlavə olunanda yalnız onun şəkilləri hesablanır.

Hesabatda tester ID-ləri və fərdi göstəricilər yoxdur — yalnız testerlər arası yayılma.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import EMOTION_CLASSES, AppConfig, load_config
from .dataset import Sample, load_cache, load_samples, save_cache
from .labels import EMOTION_LABELS
from .report import PAGE, details, esc, num, pct, seq_bin, tile, tip, wilson, write_report
from .variants import Variant, face_crop

DEFAULT_MARGINS = (0.0, 0.1, 0.2, 0.3)
EXTRA_THRESHOLDS = (0.5, 0.6, 0.7)
MIN_TESTERS = 3  # testerlər üzrə çarpaz yoxlama (LOTO) üçün
TOP_ROWS = 25  # hesabatda açıq görünən sıra sayı
CLASS_INDEX = {c: i for i, c in enumerate(EMOTION_CLASSES)}


# --- proqnozlar -------------------------------------------------------------------


@dataclass
class Preds:
    probs: np.ndarray  # (N, 8)
    ms: np.ndarray  # (N,)
    flip_probs: np.ndarray | None = None
    flip_ms: np.ndarray | None = None

    def get(self, flip: bool) -> tuple[np.ndarray, np.ndarray]:
        if not flip:
            return self.probs, self.ms
        return (self.probs + self.flip_probs) / 2.0, self.ms + self.flip_ms


def crop_key(margin: float, align: bool) -> str:
    return Variant(margin, align, False).key


def build_variants(margins, aligns, flips, baseline: Variant) -> list[Variant]:
    out = [Variant(float(m), a, f) for m in margins for a in aligns for f in flips]
    if baseline not in out:
        out.insert(0, baseline)
    return out


def _predict_one(recognizer, image: np.ndarray, sample: Sample, margin: float, align: bool,
                 flip: bool) -> tuple[np.ndarray, float, np.ndarray | None, float]:
    import cv2

    crop = face_crop(image, sample.box, sample.landmarks, margin, align)
    if crop.size == 0:
        crop = image
    p = recognizer.predict(crop)
    if not flip:
        return p.probs, p.latency_ms, None, 0.0
    q = recognizer.predict(cv2.flip(crop, 1))
    return p.probs, p.latency_ms, q.probs, q.latency_ms


def predict_all(
    samples: list[Sample],
    models: list[str],
    variants: list[Variant],
    use_cache: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
    recognizer_factory: Callable[[str], Any] | None = None,
) -> dict[tuple[str, str], Preds]:
    """(model, crop_key) → bütün nümunələr üçün proqnozlar (samples sırası ilə)."""
    if recognizer_factory is None:
        from .model_zoo import make_recognizer as recognizer_factory  # noqa: N813

    crops = list(dict.fromkeys((v.margin, v.align) for v in variants))
    need_flip = any(v.flip for v in variants)
    by_session: dict[Path, list[int]] = {}
    for i, s in enumerate(samples):
        by_session.setdefault(s.path.parent, []).append(i)

    n = len(samples)
    out: dict[tuple[str, str], Preds] = {}
    for key in models:
        arrays = {
            crop_key(m, a): {
                "probs": np.zeros((n, len(EMOTION_CLASSES))), "ms": np.zeros(n),
                "flip_probs": np.zeros((n, len(EMOTION_CLASSES))) if need_flip else None,
                "flip_ms": np.zeros(n) if need_flip else None,
            }
            for m, a in crops
        }
        todo: dict[Path, list[tuple[float, bool]]] = {}
        for sdir, idx in by_session.items():
            files = np.array([samples[i].path.name for i in idx])
            for m, a in crops:
                ck = crop_key(m, a)
                cached = load_cache(sdir, f"{key}__{ck}") if use_cache else None
                rows = _match_cache(cached, files, need_flip)
                if rows is None:
                    todo.setdefault(sdir, []).append((m, a))
                    continue
                dst = arrays[ck]
                dst["probs"][idx] = cached["probs"][rows]
                dst["ms"][idx] = cached["ms"][rows]
                if need_flip:
                    dst["flip_probs"][idx] = cached["flip_probs"][rows]
                    dst["flip_ms"][idx] = cached["flip_ms"][rows]

        total = sum(len(by_session[sdir]) * len(cs) for sdir, cs in todo.items())
        if total:
            recognizer = recognizer_factory(key)
            done, last = 0, 0.0
            for sdir, cs in todo.items():
                idx = by_session[sdir]
                for i in idx:
                    image = samples[i].image()
                    for m, a in cs:
                        dst = arrays[crop_key(m, a)]
                        p, ms, fp, fms = _predict_one(recognizer, image, samples[i], m, a, need_flip)
                        dst["probs"][i], dst["ms"][i] = p, ms
                        if need_flip:
                            dst["flip_probs"][i], dst["flip_ms"][i] = fp, fms
                        done += 1
                    if progress and (time.monotonic() - last > 1.0 or done == total):
                        last = time.monotonic()
                        progress(key, done, total)
                if use_cache:
                    files = np.array([samples[i].path.name for i in idx])
                    for m, a in cs:
                        src = arrays[crop_key(m, a)]
                        payload = {"files": files, "probs": src["probs"][idx], "ms": src["ms"][idx]}
                        if need_flip:
                            payload.update(flip_probs=src["flip_probs"][idx], flip_ms=src["flip_ms"][idx])
                        save_cache(sdir, f"{key}__{crop_key(m, a)}", payload)
            del recognizer

        for ck, a in arrays.items():
            out[(key, ck)] = Preds(a["probs"], a["ms"], a["flip_probs"], a["flip_ms"])
    return out


def _match_cache(cached: dict | None, files: np.ndarray, need_flip: bool) -> np.ndarray | None:
    """Keşdəki sətirlərin indeksləri (files sırası ilə) və ya None (keş yararsızdır)."""
    if cached is None or "files" not in cached or (need_flip and "flip_probs" not in cached):
        return None
    pos = {str(f): i for i, f in enumerate(cached["files"])}
    try:
        return np.array([pos[str(f)] for f in files], dtype=int)
    except KeyError:
        return None


# --- metrikalar -------------------------------------------------------------------


@dataclass
class Meta:
    expected: np.ndarray  # (N,) sinif indeksi
    step: np.ndarray  # (N,) unikal addım (sessiya + addım)
    tester: np.ndarray  # (N,) tester indeksi (ID hesabata çıxmır)

    @classmethod
    def from_samples(cls, samples: list[Sample]) -> "Meta":
        steps: dict[tuple[str, int], int] = {}
        testers: dict[str, int] = {}
        return cls(
            np.array([CLASS_INDEX[s.expected] for s in samples], dtype=int),
            np.array([steps.setdefault((s.session_id, s.step), len(steps)) for s in samples], dtype=int),
            np.array([testers.setdefault(s.tester_id, len(testers)) for s in samples], dtype=int),
        )


def _balanced(correct: np.ndarray, y: np.ndarray) -> float | None:
    accs = [float(correct[y == e].mean()) for e in np.unique(y)]
    return float(np.mean(accs)) if accs else None


def score(probs: np.ndarray, ms: np.ndarray, meta: Meta, supported: frozenset[str], cfg: AppConfig,
          thresholds: tuple[float, ...]) -> dict[str, Any] | None:
    """Bir model + ayar üçün kadr və addım səviyyəsində göstəricilər."""
    mask = np.array([EMOTION_CLASSES[e] in supported for e in meta.expected], dtype=bool)
    if not mask.any():
        return None
    p, y, steps, testers = probs[mask], meta.expected[mask], meta.step[mask], meta.tester[mask]
    pred, top = p.argmax(1), p.max(1)
    correct = pred == y
    n, k = len(y), int(correct.sum())
    cats = np.array([cfg.mapping[c] for c in EMOTION_CLASSES])

    per_expr = {EMOTION_CLASSES[e]: (float(correct[y == e].mean()), int((y == e).sum())) for e in np.unique(y)}
    step_frame_acc: dict[int, float] = {}
    step_hits = []
    for s in np.unique(steps):
        m = steps == s
        step_frame_acc[int(s)] = float(correct[m].mean())
        step_hits.append(p[m].mean(0).argmax() == y[m][0])  # canlı 10 s ortalamaya yaxın
    per_tester = {int(t): _balanced(correct[testers == t], y[testers == t]) for t in np.unique(testers)}
    conf = {}
    for t in thresholds:
        c = top >= t
        conf[t] = (float(c.mean()), float(correct[c].mean()) if c.any() else None)
    tester_vals = [v for v in per_tester.values() if v is not None]
    return {
        "n": n, "k": k, "acc": k / n, "ci": wilson(k, n), "balanced": _balanced(correct, y),
        "category": float((cats[pred] == cats[y]).mean()),
        "step_acc": float(np.mean(step_hits)), "n_steps": len(step_hits),
        "per_expr": per_expr, "conf": conf, "ms": float(ms[mask].mean()),
        "spread": (min(tester_vals), max(tester_vals)) if len(tester_vals) >= 2 else None,
        "per_tester": per_tester, "step_frame_acc": step_frame_acc,
    }


def sign_test(a: dict[int, float], b: dict[int, float]) -> dict[str, Any]:
    """Addım səviyyəsində işarə testi: a b-dən neçə addımda yaxşı/pisdir, ikitərəfli p."""
    common = sorted(a.keys() & b.keys())
    better = sum(a[s] > b[s] + 1e-9 for s in common)
    worse = sum(b[s] > a[s] + 1e-9 for s in common)
    m = better + worse
    p = None
    if m:
        k = min(better, worse)
        p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2**m)
    return {"better": better, "worse": worse, "ties": len(common) - m, "p": p}


def loto(candidates: dict[Any, dict[int, float | None]], baseline: Any) -> dict[str, Any] | None:
    """Testerlər üzrə çarpaz yoxlama: hər tester üçün ən yaxşı variant QALANLARDA seçilir,
    sonra həmin testerdə canlı ayarla müqayisə olunur. "Ən yaxşını seçmək" optimizmini aradan qaldırır."""
    if baseline not in candidates:
        return None
    testers = set.intersection(*(
        {t for t, v in per.items() if v is not None} for per in candidates.values()
    ))
    if len(testers) < MIN_TESTERS:
        return None
    gains, picks = [], Counter()
    for t in sorted(testers):
        others = testers - {t}
        best = max(candidates, key=lambda c: (np.mean([candidates[c][o] for o in others]), c == baseline))
        picks[best] += 1
        gains.append(candidates[best][t] - candidates[baseline][t])
    pick, count = picks.most_common(1)[0]
    return {
        "gain": float(np.mean(gains)), "n": len(gains),
        "improved": sum(g > 1e-9 for g in gains), "worse": sum(g < -1e-9 for g in gains),
        "pick": pick, "pick_share": count / len(gains),
    }


def setting_effects(rows: dict[Variant, dict], base_margin: float) -> dict[str, dict[str, Any]]:
    """Hər ayarın təsiri: digər ayarlar sabit saxlanaraq balanslı düzgünlük fərqi (orta)."""
    pairs: dict[str, list[tuple[Variant, Variant]]] = {"align": [], "flip": []}
    for v in rows:
        if not v.align and (w := Variant(v.margin, True, v.flip)) in rows:
            pairs["align"].append((v, w))
        if not v.flip and (w := Variant(v.margin, v.align, True)) in rows:
            pairs["flip"].append((v, w))
    for m in sorted({v.margin for v in rows} - {base_margin}):
        pairs[f"m{round(m * 100)}"] = [
            (v, w) for v in rows if v.margin == base_margin
            if (w := Variant(m, v.align, v.flip)) in rows
        ]
    out = {}
    for name, ps in pairs.items():
        deltas = [rows[w]["balanced"] - rows[v]["balanced"] for v, w in ps]
        if deltas:
            out[name] = {"delta": float(np.mean(deltas)), "n": len(deltas), "better": sum(d > 1e-9 for d in deltas)}
    return out


def run_evaluation(
    cfg: AppConfig,
    samples: list[Sample],
    models: list[str],
    variants: list[Variant],
    baseline: Variant,
    use_cache: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
    recognizer_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    from .model_zoo import spec

    if not samples:
        raise ValueError("Qiymətləndirmək üçün şəkil yoxdur")
    if baseline not in variants:
        variants = [baseline, *variants]
    threshold = cfg.smoothing.uncertain_threshold
    thresholds = tuple(sorted({threshold, *EXTRA_THRESHOLDS}))
    meta = Meta.from_samples(samples)
    preds = predict_all(samples, models, variants, use_cache, progress, recognizer_factory)

    rows: list[dict] = []
    by_model: dict[str, dict[str, Any]] = {}
    for key in models:
        s = spec(key)
        per_variant: dict[Variant, dict] = {}
        for v in variants:
            probs, ms = preds[(key, crop_key(v.margin, v.align))].get(v.flip)
            r = score(probs, ms, meta, s.classes, cfg, thresholds)
            if r is None:
                continue
            r.update(model=key, label=s.label, dataset=s.dataset, variant=v)
            per_variant[v] = r
            rows.append(r)
        if baseline not in per_variant:
            continue
        base = per_variant[baseline]
        for r in per_variant.values():
            r["vs_base"] = {"delta": r["balanced"] - base["balanced"],
                            **sign_test(r["step_frame_acc"], base["step_frame_acc"])}
        best = max(per_variant.values(), key=lambda r: (r["balanced"], r["variant"] == baseline))
        by_model[key] = {
            "baseline": base, "best": best,
            "effects": setting_effects(per_variant, baseline.margin),
            "loto": loto({v: r["per_tester"] for v, r in per_variant.items()}, baseline),
        }

    rows.sort(key=lambda r: (-r["balanced"], -r["step_acc"], r["ms"]))
    live_key = (cfg.model.name, baseline)
    live = next((r for r in rows if (r["model"], r["variant"]) == live_key), None)
    best = rows[0] if rows else None
    glob: dict[str, Any] = {"best": best, "live": live, "sign": None, "loto": None}
    if best is not None and live is not None:
        glob["sign"] = sign_test(best["step_frame_acc"], live["step_frame_acc"])
        glob["loto"] = loto({(r["model"], r["variant"]): r["per_tester"] for r in rows}, live_key)

    present = {EMOTION_CLASSES[e] for e in np.unique(meta.expected)}
    expressions = [e for e in cfg.test.expressions if e in present] + sorted(present - set(cfg.test.expressions))
    return {
        "n_samples": len(samples), "n_sessions": len({s.session_id for s in samples}),
        "n_testers": len(np.unique(meta.tester)), "n_steps": len(np.unique(meta.step)),
        "threshold": threshold, "thresholds": thresholds, "models": models, "variants": variants,
        "baseline": baseline, "rows": rows, "by_model": by_model, "global": glob, "expressions": expressions,
    }


# --- formatlama -------------------------------------------------------------------

FACTOR_LABELS = {"align": "Düzləndirmə", "flip": "Flip TTA"}


def factor_label(name: str, base_margin: float) -> str:
    if name in FACTOR_LABELS:
        return FACTOR_LABELS[name]
    return f"Boşluq {name[1:]}% (canlı: {round(base_margin * 100)}%)"


def pp(d: float | None) -> str:
    if d is None:
        return "—"
    sign = "+" if d > 5e-4 else "−" if d < -5e-4 else "±"
    return f"{sign}{num(abs(d) * 100, 1)} pp"


def fmt_p(p: float | None) -> str:
    if p is None:
        return "—"
    return "p<0,001" if p < 0.001 else f"p={num(p, 3 if p < 0.01 else 2)}"


def variant_label(v: Variant, baseline: Variant) -> str:
    return v.name + (" (canlı)" if v == baseline else "")


def format_console(res: dict[str, Any], limit: int = 12) -> str:
    base = res["baseline"]
    lines = [
        f"{res['n_samples']} şəkil · {res['n_testers']} tester · {res['n_steps']} addım · "
        f"{len(res['models'])} model × {len(res['variants'])} ayar",
        "",
        f"{'#':>2}  {'Model':<16} {'Ayar':<36} {'Balanslı':>8} {'Kadr':>6} {'Addım':>6} {'Kat.':>6} {'ms':>6}",
    ]
    for i, r in enumerate(res["rows"][:limit], 1):
        lines.append(
            f"{i:>2}  {r['label']:<16} {variant_label(r['variant'], base):<36} {pct(r['balanced'], 1):>8} "
            f"{pct(r['acc']):>6} {pct(r['step_acc']):>6} {pct(r['category']):>6} {num(r['ms'], 1):>6}"
        )
    lines += ["", "Hər model üçün ən yaxşı ayar (canlı ayara nisbətən):"]
    for key, bm in res["by_model"].items():
        b, best = bm["baseline"], bm["best"]
        vs = best["vs_base"]
        lo = bm["loto"]
        lo_txt = f" · LOTO {pp(lo['gain'])}" if lo else ""
        lines.append(
            f"  {b['label']:<16} {pct(b['balanced'], 1):>6} → {pct(best['balanced'], 1):>6} "
            f"({pp(vs['delta'])}, addım {vs['better']}↑/{vs['worse']}↓ {fmt_p(vs['p'])}{lo_txt})  {best['variant'].name}"
        )
    g = res["global"]
    if g["best"] is not None:
        b = g["best"]
        lines += ["", f"Ən yaxşı: {b['label']} · {b['variant'].name} — balanslı {pct(b['balanced'], 1)}"]
        if g["live"] is not None:
            s = g["sign"]
            lines.append(f"Canlı ayar ({g['live']['label']} · {base.name}): {pct(g['live']['balanced'], 1)} · "
                         f"fərq {pp(b['balanced'] - g['live']['balanced'])}, addım {s['better']}↑/{s['worse']}↓ {fmt_p(s['p'])}")
    if res["n_testers"] < MIN_TESTERS:
        lines.append(f"\nDiqqət: {res['n_testers']} tester — nəticə az adamın üzünə uyğunlaşa bilər; "
                     f"LOTO yoxlaması üçün ən azı {MIN_TESTERS} tester lazımdır.")
    return "\n".join(lines)


def write_csv(res: dict[str, Any], path: Path) -> Path:
    exprs = res["expressions"]
    ts = res["thresholds"]
    head = ["rank", "model", "dataset", "variant", "margin", "align", "flip", "frames", "accuracy", "ci_low", "ci_high",
            "balanced", "category_acc", "step_acc", "steps", "ms_per_frame", "delta_vs_live_variant", "steps_better",
            "steps_worse", "sign_p", "tester_min", "tester_max",
            *[f"coverage@{t:g}" for t in ts], *[f"acc_confident@{t:g}" for t in ts], *[f"acc_{e}" for e in exprs]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for i, r in enumerate(res["rows"], 1):
            v, vs = r["variant"], r.get("vs_base", {})
            ci = r["ci"] or (None, None)
            spread = r["spread"] or (None, None)
            w.writerow([
                i, r["model"], r["dataset"], v.key, v.margin, int(v.align), int(v.flip), r["n"],
                _f(r["acc"]), _f(ci[0]), _f(ci[1]), _f(r["balanced"]), _f(r["category"]), _f(r["step_acc"]),
                r["n_steps"], _f(r["ms"], 2), _f(vs.get("delta")), vs.get("better", ""), vs.get("worse", ""),
                _f(vs.get("p")), _f(spread[0]), _f(spread[1]),
                *[_f(r["conf"][t][0]) for t in ts], *[_f(r["conf"][t][1]) for t in ts],
                *[_f(r["per_expr"][e][0]) if e in r["per_expr"] else "" for e in exprs],
            ])
    return path


def _f(x: float | None, digits: int = 4) -> str:
    return "" if x is None else f"{x:.{digits}f}"


# --- HTML -------------------------------------------------------------------------


def _acc_bar(r: dict, label: str) -> str:
    acc, ci = r["balanced"], r["ci"]
    lo, hi = (pct(ci[0]), pct(ci[1])) if ci else ("—", "—")
    attrs = tip(label, f"Balanslı {pct(acc, 1)} · kadr {pct(r['acc'], 1)}", f"Kadr düzgünlüyünün 95% intervalı {lo}–{hi}",
                f"{r['n']} kadr")
    whisker = f'<span class="ci" style="left:{ci[0] * 100:.1f}%;width:{(ci[1] - ci[0]) * 100:.1f}%"></span>' if ci else ""
    return (f'<div class="agree-row"><span class="num">{pct(acc)}</span><div class="ab" {attrs}>'
            f'<span class="ab-fill" style="width:{acc * 100:.1f}%"></span>{whisker}</div></div>')


def _sig(vs: dict) -> str:
    if vs.get("p") is None:
        return '<span class="muted">—</span>'
    return f'{vs["better"]}↑ / {vs["worse"]}↓ <span class="muted">{fmt_p(vs["p"])}</span>'


def render_html(res: dict[str, Any], cfg: AppConfig, generated: datetime | None = None) -> str:
    generated = generated or datetime.now()
    base, g, exprs = res["baseline"], res["global"], res["expressions"]
    t0 = res["threshold"]

    banners = []
    if res["n_testers"] < MIN_TESTERS:
        banners.append(
            f'<div class="banner warn"><strong>Az tester ({res["n_testers"]}).</strong> Ən yaxşı ayar bir-iki adamın üzünə '
            f'uyğunlaşa bilər. Testerlər üzrə çarpaz yoxlama (LOTO) ən azı {MIN_TESTERS} tester tələb edir.</div>')

    hero = ""
    best, live = g["best"], g["live"]
    if best is not None:
        live_tiles = ""
        if live is not None:
            s = g["sign"]
            lo = g["loto"]
            live_tiles = (
                tile("Canlı ayar", pct(live["balanced"], 1), f"{esc(live['label'])} · {esc(base.name)}")
                + tile("Fərq (addım üzrə)", pp(best["balanced"] - live["balanced"]),
                       f"{s['better']}↑ / {s['worse']}↓ addım · {fmt_p(s['p'])}")
                + tile("LOTO — yeni adamda gözlənilən", pp(lo["gain"]) if lo else "—",
                       f"{lo['improved']}/{lo['n']} testerdə yaxşı" if lo else f"≥{MIN_TESTERS} tester lazımdır")
            )
        hero = f"""
<section class="hero-card">
  <div>
    <p class="eyebrow">Ən yaxşı kombinasiya</p>
    <p class="hero-figure num">{pct(best['balanced'])}</p>
    <p class="sub"><strong>{esc(best['label'])}</strong> · {esc(variant_label(best['variant'], base))}<br>
    balanslı düzgünlük · addım üzrə {pct(best['step_acc'])} · {num(best['ms'], 1)} ms/kadr</p>
  </div>
  <div class="hero-side">{live_tiles}</div>
</section>"""

    # Hər model üçün ən yaxşı ayar
    mrows = []
    for key, bm in sorted(res["by_model"].items(), key=lambda kv: -kv[1]["best"]["balanced"]):
        b, bb = bm["baseline"], bm["best"]
        lo = bm["loto"]
        lo_txt = (f'{pp(lo["gain"])} <span class="muted">{lo["improved"]}/{lo["n"]}</span>' if lo else '<span class="muted">—</span>')
        mrows.append(
            f'<tr><th>{esc(b["label"])}<br><code class="sub-code">{esc(key)}</code></th>'
            f'<td class="num">{pct(b["balanced"], 1)}</td><td>{esc(variant_label(bb["variant"], base))}</td>'
            f'<td class="agree-cell">{_acc_bar(bb, b["label"])}</td><td class="num">{pp(bb["vs_base"]["delta"])}</td>'
            f'<td>{_sig(bb["vs_base"])}</td><td>{lo_txt}</td><td class="num">{num(bb["ms"], 1)}</td></tr>'
        )
    per_model = (
        '<div class="table-wrap"><table class="data models-table"><thead><tr><th>Model</th>'
        f'<th class="num">Canlı ayarla</th><th>Ən yaxşı ayar</th><th>Balanslı (ən yaxşı)</th><th class="num">Fərq</th>'
        '<th>Addım üzrə ↑/↓</th><th>LOTO</th><th class="num">ms/kadr</th>'
        f'</tr></thead><tbody>{"".join(mrows)}</tbody></table></div>'
    )

    # Ayarların təsiri
    factors = list(dict.fromkeys(f for bm in res["by_model"].values() for f in bm["effects"]))
    erows = []
    for key, bm in res["by_model"].items():
        cells = []
        for f in factors:
            e = bm["effects"].get(f)
            if e is None:
                cells.append('<td class="num muted">—</td>')
                continue
            strong = " strong" if abs(e["delta"]) >= 0.02 and e["better"] in (0, e["n"]) else ""
            attrs = tip(f"{bm['baseline']['label']} · {factor_label(f, base.margin)}",
                        f"{e['better']}/{e['n']} cütdə yaxşılaşdırdı", f"orta {pp(e['delta'])}")
            cells.append(f'<td class="num eff{strong}" {attrs}>{pp(e["delta"])}</td>')
        erows.append(f'<tr><th>{esc(bm["baseline"]["label"])}</th>{"".join(cells)}</tr>')
    effects = (
        '<div class="table-wrap"><table class="data"><thead><tr><th>Model</th>'
        + "".join(f'<th class="num">{esc(factor_label(f, base.margin))}</th>' for f in factors)
        + f'</tr></thead><tbody>{"".join(erows)}</tbody></table></div>'
    ) if factors else '<p class="empty">Müqayisə üçün ayar cütü yoxdur</p>'

    # Bütün kombinasiyalar
    def all_row(i: int, r: dict) -> str:
        cov, acc_c = r["conf"][t0]
        return (
            f'<tr><td class="num muted">{i}</td><th>{esc(r["label"])}</th><td>{esc(variant_label(r["variant"], base))}</td>'
            f'<td class="agree-cell">{_acc_bar(r, r["label"] + " · " + r["variant"].name)}</td>'
            f'<td class="num">{pct(r["acc"])}</td><td class="num">{pct(r["step_acc"])}</td><td class="num">{pct(r["category"])}</td>'
            f'<td class="num">{pct(cov)} / {pct(acc_c)}</td><td class="num">{num(r["ms"], 1)}</td></tr>'
        )

    head = ('<thead><tr><th class="num">#</th><th>Model</th><th>Ayar</th><th>Balanslı</th><th class="num">Kadr</th>'
            f'<th class="num">Addım</th><th class="num">Kateqoriya</th><th class="num">≥{num(t0 * 100)}%: əhatə / düz.</th>'
            '<th class="num">ms</th></tr></thead>')
    top_rows = "".join(all_row(i, r) for i, r in enumerate(res["rows"][:TOP_ROWS], 1))
    rest_rows = "".join(all_row(i, r) for i, r in enumerate(res["rows"][TOP_ROWS:], TOP_ROWS + 1))
    all_table = f'<div class="table-wrap"><table class="data models-table">{head}<tbody>{top_rows}</tbody></table></div>'
    if rest_rows:
        all_table += details(f'<table class="data models-table">{head}<tbody>{rest_rows}</tbody></table>').replace(
            "Cədvəl görünüşü", f"Qalan {len(res['rows']) - TOP_ROWS} kombinasiya")

    # İfadələr üzrə: hər modelin ən yaxşı ayarı + canlı ayar
    heat_rows = []
    shown = []
    if live is not None:
        shown.append((f"{live['label']} · canlı ayar", live))
    for bm in sorted(res["by_model"].values(), key=lambda b: -b["best"]["balanced"]):
        r = bm["best"]
        if r is not live:
            shown.append((f"{r['label']} · {r['variant'].name}", r))
    for name, r in shown:
        cells = []
        for e in exprs:
            v = r["per_expr"].get(e)
            if v is None:
                cells.append('<td class="cell">—</td>')
                continue
            attrs = tip(f"{name} · {EMOTION_LABELS[e]}", f"{pct(v[0], 1)} düzgün", f"{v[1]} kadr")
            cells.append(f'<td class="cell b{seq_bin(v[0])}" {attrs}>{pct(v[0])}</td>')
        heat_rows.append(f'<tr><th scope="row">{esc(name)}</th>{"".join(cells)}</tr>')
    heat = (
        '<div class="table-wrap"><table class="confusion pairwise"><thead><tr><th class="corner">Model · ayar ↓ · ifadə →</th>'
        + "".join(f'<th scope="col">{esc(EMOTION_LABELS[e])}</th>' for e in exprs)
        + f'</tr></thead><tbody>{"".join(heat_rows)}</tbody></table></div>'
    )
    scale = "".join(f'<span class="b{i}"></span>' for i in range(7))

    # Hədd cədvəli (ən yaxşı kombinasiya)
    thr = ""
    if best is not None:
        trs = "".join(
            f'<tr><td class="num">{num(t * 100)}%</td><td class="num">{pct(best["conf"][t][0])}</td>'
            f'<td class="num">{pct(best["conf"][t][1], 1)}</td></tr>' for t in res["thresholds"]
        )
        thr = f"""
<section class="card">
  <h2>Əminlik həddi — {esc(best['label'])} · {esc(best['variant'].name)}</h2>
  <p class="lead">Həddi qaldırdıqca model daha az kadrda nəticə verir (qalanı “qeyri-müəyyən”), amma verdiyi nəticə daha
  düzgün olur. Canlı hədd: <code>smoothing.uncertain_threshold = {num(t0, 2)}</code>.</p>
  <div class="table-wrap"><table class="data" style="width:auto"><thead><tr><th class="num">Hədd</th>
  <th class="num">Əhatə (nəticə verilən kadr)</th><th class="num">Düzgünlük (əmin kadrlarda)</th></tr></thead>
  <tbody>{trs}</tbody></table></div>
</section>"""

    rec = ""
    if best is not None:
        rec = (f"<li><strong>Tövsiyə (avtomatik tətbiq olunmur):</strong> <code>model.name: {esc(best['model'])}</code>, "
               f"<code>detector.crop_margin: {best['variant'].margin:g}</code>, "
               f"<code>detector.align: {str(best['variant'].align).lower()}</code>, "
               f"<code>model.flip_tta: {str(best['variant'].flip).lower()}</code>. "
               "Dəyişməzdən əvvəl fərqin addım üzrə əhəmiyyətli olduğuna və LOTO-nun müsbət olduğuna baxın.</li>")

    chips = "".join(f'<span class="chip">{esc(c)}</span>' for c in (
        f"{res['n_testers']} tester", f"{res['n_sessions']} sessiya", f"{num(res['n_samples'])} şəkil",
        f"{res['n_steps']} addım", f"{len(res['models'])} model", f"{len(res['variants'])} ayar",
    ))
    body = f"""
<header class="page-head">
  <p class="eyebrow">Emosiya Pilotu · oflayn qiymətləndirmə</p>
  <h1>Model × ayar</h1>
  <p class="sub">Nəzarətli testdə saxlanılmış şəkillər üzərində hər model və ön emal ayarı (boşluq, düzləndirmə, flip)
  eyni kadrlarda müqayisə olunur. Canlı ayar: {esc(base.name)}.</p>
  <div class="chips">{chips}</div>
</header>
{"".join(banners)}
{hero}
<section class="card">
  <h2>Hər model üçün ən yaxşı ayar</h2>
  <p class="lead">“Fərq” — eyni modelin canlı ayarına nisbətən balanslı düzgünlük. “Addım üzrə” — hər tapşırılan ifadə
  addımında kadrların düzgünlük payı müqayisə olunur (işarə testi); kadrlar bir-birinə bənzədiyi üçün bu, kadr sayından
  daha dürüst göstəricidir. LOTO — ayar bir testerdən başqa hamısında seçilib, kənarda qalan testerdə yoxlanır.</p>
  {per_model}
</section>
<section class="card">
  <h2>Ayarların təsiri</h2>
  <p class="lead">Hər ayar dəyişdirilərkən digərləri sabit saxlanır; orta fərq göstərilir. Qalın — bütün cütlərdə eyni
  istiqamətdə və ən azı 2 pp.</p>
  {effects}
</section>
<section class="card">
  <h2>İfadələr üzrə</h2>
  <p class="lead">Canlı ayar və hər modelin ən yaxşı ayarı.</p>
  {heat}
  <div class="scale"><span>0%</span><div class="ramp">{scale}</div><span>100%</span></div>
</section>
{thr}
<section class="card">
  <h2>Bütün kombinasiyalar</h2>
  <p class="lead">Balanslı düzgünlüyə görə sıralanıb. “Addım” — addımdakı kadrların ehtimal ortalaması (canlı 10 s
  ortalamaya yaxın). “Kateqoriya” — operator kateqoriyasına (mapping) çevriləndən sonra.</p>
  {all_table}
</section>
<section class="card notes">
  <h2>Metodologiya və məhdudiyyətlər</h2>
  <ul>
    {rec}
    <li><strong>Pozalı ifadələr təbii ifadələrdən asandır</strong> — bu, yuxarı həddir; canlı sorğu cavabları ilə birlikdə baxın.</li>
    <li>Ən yaxşını {len(res['rows'])} kombinasiya arasından seçmək nəticəni şişirdir. Buna görə LOTO və addım üzrə işarə testi
      göstərilir; yeni testerlərlə təsdiqlənməyən fərqə etibar etməyin.</li>
    <li>Kadr düzgünlüyünün etibar intervalı (Wilson) kadrları müstəqil sayır — real qeyri-müəyyənlik daha böyükdür.</li>
    <li>FER2013 modellərində “Nifrət” (Contempt) yoxdur; həmin ifadə onların hesabından çıxarılır.</li>
    <li>Üz qutusu və nöqtələr test vaxtı YuNet-dən götürülüb; yalnız tam görünən üzlər (status “ok”) sayılır.</li>
    <li>Şəkillər yalnız razılıqla, <code>{esc(str(cfg.research.dir))}</code> qovluğundadır; silmək:
      <code>python -m src.dataset --delete &lt;sessiya&gt;</code>.</li>
  </ul>
  <p class="foot">Yaradılıb: {generated.strftime("%Y-%m-%d %H:%M")}. Bu hesabatda tester ID-ləri və fərdi göstəricilər yoxdur.</p>
</section>"""
    extra_css = """<style>
.muted { color: var(--muted); }
td.eff.strong { font-weight: 700; color: var(--ink); }
td.eff { color: var(--ink-2); }
</style>
</head>"""
    return (PAGE.replace("<title>Pilot hesabatı</title>", "<title>Model × ayar</title>")
            .replace("</head>", extra_css, 1).replace("{{BODY}}", body))


# --- CLI --------------------------------------------------------------------------


def _parse_bool_set(value: str) -> tuple[bool, ...]:
    return {"off": (False,), "on": (True,), "both": (False, True)}[value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.evaluate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--dataset", help="dataset qovluğu (default: research.dir)")
    parser.add_argument("--models", help="vergüllə; default: quraşdırılmış bütün modellər")
    parser.add_argument("--margins", default=",".join(f"{m:g}" for m in DEFAULT_MARGINS),
                        help="kəsik boşluqları, vergüllə (0 = AffectNet kimi boşluqsuz)")
    parser.add_argument("--align", choices=["off", "on", "both"], default="both")
    parser.add_argument("--flip", choices=["off", "on", "both"], default="both")
    parser.add_argument("--testers", help="yalnız bu tester ID-ləri (vergüllə)")
    parser.add_argument("--all-faces", action="store_true", help="qismən/profil üzləri də say (default: yalnız ok)")
    parser.add_argument("--no-cache", action="store_true", help="proqnozları yenidən hesabla")
    parser.add_argument("--out", help="HTML yolu (CSV yanında .csv ilə yazılır)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    from .model_zoo import SPECS, installed_models
    from .privacy import install_runtime_guards

    cfg = load_config(args.config)
    root = Path(args.dataset) if args.dataset else cfg.research.dir
    testers = {t.strip() for t in args.testers.split(",")} if args.testers else None
    samples = load_samples(root, testers=testers, only_ok=not args.all_faces)
    if not samples:
        print(f"{root} — qiymətləndiriləcək şəkil yoxdur. Nəzarətli testi “Şəkilləri saxla” ilə keçirin.")
        return 1

    models = [m.strip() for m in args.models.split(",")] if args.models else installed_models()
    unknown = [m for m in models if m not in SPECS]
    if unknown:
        print(f"Naməlum model: {', '.join(unknown)}")
        return 2
    missing = [m for m in models if m not in installed_models()]
    if missing:
        print(f"Quraşdırılmayıb, keçilir: {', '.join(missing)}")
        models = [m for m in models if m not in missing]
    if not models:
        return 2
    try:
        margins = sorted({float(x) for x in args.margins.split(",")})
    except ValueError:
        print("--margins: rəqəmlər vergüllə, məs. 0,0.1,0.2")
        return 2
    baseline = Variant(cfg.detector.crop_margin, cfg.detector.align, cfg.model.flip_tta)
    variants = build_variants(margins, _parse_bool_set(args.align), _parse_bool_set(args.flip), baseline)

    print(f"{len(samples)} şəkil · {len(models)} model × {len(variants)} ayar — proqnozlar hesablanır…", file=sys.stderr)
    guards = install_runtime_guards(cfg.privacy.block_network)
    started = time.monotonic()

    def progress(model: str, done: int, total: int) -> None:
        print(f"\r  {model:<24} {done * 100 // total:>3}%  ({done}/{total})", end="" if done < total else "\n",
              file=sys.stderr, flush=True)

    try:
        res = run_evaluation(cfg, samples, models, variants, baseline, use_cache=not args.no_cache, progress=progress)
    finally:
        guards.uninstall()
    print(f"Hazırdır: {time.monotonic() - started:.0f} s\n", file=sys.stderr)
    print(format_console(res))

    out = Path(args.out).resolve() if args.out else cfg.report.dir / f"qiymetlendirme_{datetime.now():%Y-%m-%d_%H%M}.html"
    write_report(render_html(res, cfg), out)
    csv_path = write_csv(res, out.with_suffix(".csv"))
    print(f"\nHesabat: {out}\nCSV:     {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
