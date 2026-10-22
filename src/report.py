"""CSV loglarından HTML hesabat.

    python -m src.report                       # logs/ → reports/hesabat_YYYY-MM-DD.html
    python -m src.report --logs logs --out r.html --from 2026-10-01 --to 2026-10-31

Hesabat YALNIZ qrup səviyyəsindədir: fərdi operator reytinqi, sıralaması və ya
operator üzrə bölgü yaradılmır. Fayl özünə-kifayətdir (xarici resurs yoxdur).
"""

from __future__ import annotations

import argparse
import html
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DECLINED, EMOTION_CLASSES, TIMEOUT, AppConfig, load_config
from .csv_logger import COLUMNS, OPTIONAL_COLUMNS, PROB_COLUMNS, SAMPLE, SELF_REPORT, TEST
from .labels import EMOTION_LABELS, FACE_REASON_LABELS

UNCERTAIN = "_uncertain"
NO_PREDICTION = "_no_prediction"
MIN_HOUR_SAMPLES = 20
NUMERIC = ["top_prob", *PROB_COLUMNS, "valence", "arousal", "n_frames", "latency_ms", "detector_ms", "model_ms", "cpu_percent"]

# Saatlıq paylanma üçün qatlanmış seriyalar (≤ 7: 6 kateqorial slot + boz)
HOURLY_SERIES = [
    ("Neutral", "Neytral", ["Neutral"]),
    ("Happiness", "Xoşbəxtlik", ["Happiness"]),
    ("Sadness", "Kədər", ["Sadness"]),
    ("Anger", "Qəzəb", ["Anger"]),
    ("Surprise", "Təəccüb", ["Surprise"]),
    ("Other", "Digər (ikrah, həqarət, qorxu)", ["Disgust", "Contempt", "Fear"]),
    ("uncertain", "Qeyri-müəyyən", ["uncertain"]),
]
STATUS_SERIES = [
    ("ok", "Müəyyən nəticə"),
    ("uncertain", "Qeyri-müəyyən (<50%)"),
    ("no_face", "Üz yoxdur"),
    ("warming_up", "Toplanır (<3 kadr)"),
]


class ReportError(Exception):
    pass


# --- yükləmə --------------------------------------------------------------------


def find_logs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.csv"))


def load_logs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8")
        for col in OPTIONAL_COLUMNS - set(df.columns):  # köhnə sxemli fayllar
            df[col] = ""
        missing = set(COLUMNS) - set(df.columns)
        if missing:
            raise ReportError(f"{p.name}: çatışmayan sütunlar {sorted(missing)}")
        frames.append(df[COLUMNS])
    if not frames:
        raise ReportError("Heç bir CSV faylı tapılmadı")
    df = pd.concat(frames, ignore_index=True)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Vaxt qeyd olunduğu yerin yerli vaxtıdır (ISO 8601, offset ilə) — saatı birbaşa götürürük.
    df["date"] = df["timestamp"].str.slice(0, 10)
    df["hour"] = pd.to_numeric(df["timestamp"].str.slice(11, 13), errors="coerce")
    return df


def filter_logs(df: pd.DataFrame, date_from: str | None, date_to: str | None, model: str | None) -> pd.DataFrame:
    if date_from:
        df = df[df["date"] >= date_from]
    if date_to:
        df = df[df["date"] <= date_to]
    if model:
        df = df[df["model_name"] == model]
    return df


# --- statistika -----------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def cohen_kappa(a: list[str], b: list[str], labels: list[str]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    idx = {label: i for i, label in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    pe = float(m.sum(axis=1) @ m.sum(axis=0)) / (n * n)
    if pe >= 1:
        return None
    return float((po - pe) / (1 - pe))


def model_category(df: pd.DataFrame, cfg: AppConfig) -> pd.Series:
    """Modelin smoothed nəticəsini operator kateqoriyasına çevirir (config mapping)."""
    out = pd.Series(NO_PREDICTION, index=df.index, dtype=object)
    ok = df["status"] == "ok"
    out[ok] = df.loc[ok, "top_emotion"].map(cfg.mapping).fillna(NO_PREDICTION)
    out[df["status"] == "uncertain"] = UNCERTAIN
    rule = cfg.report.va_tired_rule
    if rule.enabled and "tired" in cfg.self_report.categories:
        tired = ok & df["arousal"].notna() & (df["arousal"] <= rule.arousal_max)
        out[tired] = "tired"
    return out


def accuracy(df: pd.DataFrame, cfg: AppConfig) -> dict[str, Any]:
    cats = list(cfg.self_report.categories)
    reports = df[df["record_type"] == SELF_REPORT].copy()
    answered = reports[reports["self_report"].isin(cats)].copy()
    answered["pred"] = model_category(answered, cfg)

    mappable = set(cfg.mapping.values())
    if cfg.report.va_tired_rule.enabled:
        mappable.add("tired")
    unmappable = [c for c in cats if c not in mappable]

    confident = answered[answered["pred"].isin(cats)]
    comparable = confident[confident["self_report"].isin(mappable)]
    k_all = int((answered["pred"] == answered["self_report"]).sum())
    k_conf = int((confident["pred"] == confident["self_report"]).sum())
    k_comp = int((comparable["pred"] == comparable["self_report"]).sum())

    columns = cats + [UNCERTAIN, NO_PREDICTION]
    matrix = [[int(((answered["self_report"] == r) & (answered["pred"] == c)).sum()) for c in columns] for r in cats]

    per_category = []
    for c in cats:
        as_answer = confident[confident["self_report"] == c]
        as_pred = confident[confident["pred"] == c]
        per_category.append({
            "code": c,
            "label": cfg.self_report.categories[c],
            "reports": int((answered["self_report"] == c).sum()),
            "recall": (as_answer["pred"] == c).mean() if len(as_answer) else None,
            "precision": (as_pred["self_report"] == c).mean() if len(as_pred) else None,
            "mappable": c in mappable,
        })

    baseline = None
    if len(comparable):
        baseline = float(comparable["self_report"].value_counts(normalize=True).iloc[0])

    return {
        "categories": cats,
        "columns": columns,
        "matrix": matrix,
        "n_prompts": len(reports),
        "n_answered": len(answered),
        "n_declined": int((reports["self_report"] == DECLINED).sum()),
        "n_timeout": int((reports["self_report"] == TIMEOUT).sum()),
        "overall": (k_all, len(answered)),
        "confident": (k_conf, len(confident)),
        "comparable": (k_comp, len(comparable)),
        "confident_ci": wilson(k_conf, len(confident)),
        "comparable_ci": wilson(k_comp, len(comparable)),
        "coverage": len(confident) / len(answered) if len(answered) else None,
        "kappa": cohen_kappa(list(comparable["self_report"]), list(comparable["pred"]), cats),
        "baseline": baseline,
        "per_category": per_category,
        "unmappable": unmappable,
    }


def quality(samples: pd.DataFrame) -> dict[str, Any]:
    n = len(samples)
    counts = {key: int((samples["status"] == key).sum()) for key, _ in STATUS_SERIES}
    no_face = samples[samples["face_detected"] == "0"]
    reasons = no_face["face_reason"].replace("", "not_found").value_counts()
    return {
        "n": n,
        "counts": counts,
        "share": {k: (v / n if n else 0.0) for k, v in counts.items()},
        "reasons": [(r, int(c)) for r, c in reasons.items()],
        "n_no_face": len(no_face),
    }


def hourly(samples: pd.DataFrame) -> dict[str, Any]:
    est = samples[samples["status"].isin(["ok", "uncertain"])]
    hours = sorted(int(h) for h in samples["hour"].dropna().unique())
    rows = []
    for h in range(min(hours), max(hours) + 1) if hours else []:
        sub = est[est["hour"] == h]
        n = len(sub)
        shares = {}
        for key, _, members in HOURLY_SERIES:
            k = int(sub["top_emotion"].isin(members).sum())
            shares[key] = (k, k / n if n else 0.0)
        rows.append({"hour": h, "n": n, "enough": n >= MIN_HOUR_SAMPLES, "shares": shares})
    return {"rows": rows}


def performance(samples: pd.DataFrame) -> dict[str, Any]:
    def stats(series: pd.Series) -> dict[str, float | None]:
        s = series.dropna()
        if s.empty:
            return {"mean": None, "p50": None, "p95": None, "n": 0}
        return {"mean": float(s.mean()), "p50": float(s.quantile(0.5)), "p95": float(s.quantile(0.95)), "n": int(len(s))}

    by_model = []
    for (model, engine, device), sub in samples.groupby(["model_name", "engine", "device"], sort=True):
        by_model.append({
            "model": model, "engine": engine, "device": device,
            "latency": stats(sub["latency_ms"]), "model_ms": stats(sub["model_ms"]),
            "detector": stats(sub["detector_ms"]), "cpu": stats(sub["cpu_percent"]),
        })
    return {
        "latency": stats(samples["latency_ms"]),
        "detector": stats(samples["detector_ms"]),
        "model_ms": stats(samples["model_ms"]),
        "cpu": stats(samples["cpu_percent"]),
        "by_model": by_model,
    }


def valence_arousal(df: pd.DataFrame, cfg: AppConfig) -> list[dict[str, Any]] | None:
    reports = df[(df["record_type"] == SELF_REPORT) & df["self_report"].isin(list(cfg.self_report.categories))]
    reports = reports[reports["valence"].notna()]
    if reports.empty:
        return None
    out = []
    for code, label in cfg.self_report.categories.items():
        sub = reports[reports["self_report"] == code]
        out.append({
            "label": label, "n": len(sub),
            "valence": float(sub["valence"].mean()) if len(sub) else None,
            "arousal": float(sub["arousal"].mean()) if len(sub) else None,
        })
    return out


def models_breakdown(df: pd.DataFrame, cfg: AppConfig) -> list[dict[str, Any]]:
    """Hər model üçün operatorla uyğunluq — eyni sorğular üzrə (sorğu anında hər model yazılır)."""
    from .model_zoo import SPECS

    out = []
    for model, sub in df.groupby("model_name", sort=False):
        acc = accuracy(sub, cfg)
        k, n = acc["comparable"]
        samples = sub[sub["record_type"] == SAMPLE]
        est = samples[samples["status"].isin(["ok", "uncertain"])]
        spec = SPECS.get(model)
        out.append({
            "model": model,
            "label": spec.label if spec else model,
            "dataset": spec.dataset if spec else "—",
            "answered": acc["n_answered"],
            "coverage": acc["coverage"],
            "agreement": k / n if n else None,
            "ci": acc["comparable_ci"],
            "n": n,
            "kappa": acc["kappa"],
            "uncertain_share": (est["status"] == "uncertain").mean() if len(est) else None,
            "model_ms": float(samples["model_ms"].mean()) if samples["model_ms"].notna().any() else None,
            "model_ms_p95": float(samples["model_ms"].quantile(0.95)) if samples["model_ms"].notna().any() else None,
        })
    out.sort(key=lambda m: (m["agreement"] is None, -(m["agreement"] or 0)))
    return out


def model_agreement(samples: pd.DataFrame, cfg: AppConfig, order: list[str] | None = None) -> dict[str, Any] | None:
    """Modellər arası razılaşma: eyni kadrda (eyni timestamp) hər iki model müəyyən nəticə
    verdikdə, operator kateqoriyası səviyyəsində uyğunluq faizi və Cohen kappa."""
    ok = samples[samples["status"] == "ok"][["timestamp", "model_name", "top_emotion"]].copy()
    if ok["model_name"].nunique() < 2:
        return None
    ok["cat"] = ok["top_emotion"].map(cfg.mapping)
    wide = ok.pivot_table(index="timestamp", columns="model_name", values="cat", aggfunc="first")
    models = [m for m in (order or []) if m in wide.columns] + [m for m in wide.columns if m not in (order or [])]
    cats = list(cfg.self_report.categories)
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for a in models:
        for b in models:
            if a == b:
                n = int(wide[a].notna().sum())
                cells[(a, b)] = {"n": n, "agree": 1.0 if n else None, "kappa": None}
                continue
            both = wide[[a, b]].dropna()
            n = len(both)
            agree = float((both[a] == both[b]).mean()) if n else None
            kappa = cohen_kappa(list(both[a]), list(both[b]), cats) if n else None
            cells[(a, b)] = {"n": n, "agree": agree, "kappa": kappa}
    return {"models": models, "cells": cells}


def calibration(df: pd.DataFrame) -> dict[str, Any] | None:
    """Nəzarətli test (record_type=test): hər kadrın nəticəsi tapşırılan ifadə ilə müqayisə olunur.

    Qeyri-müəyyən kadrlarda CSV-də top_emotion="uncertain" yazılır — modelin ən güclü
    təxmini ehtimal sütunlarından (argmax) bərpa olunur. Üz olmayan kadrlar məxrəcə daxil deyil.
    """
    from .model_zoo import SPECS

    t = df[(df["record_type"] == TEST) & (df["expected"] != "")].copy()
    if t.empty:
        return None
    face = t[t["status"] != "no_face"].copy()
    probs = face[PROB_COLUMNS].to_numpy(dtype=float)
    probs = np.where(np.isnan(probs), -1.0, probs)
    face["pred"] = [EMOTION_CLASSES[i] for i in probs.argmax(axis=1)] if len(face) else []
    expressions = [e for e in EMOTION_CLASSES if e in set(t["expected"])]

    models = []
    for model, sub in t.groupby("model_name", sort=False):
        fs = face[face["model_name"] == model]
        k, n = int((fs["pred"] == fs["expected"]).sum()), len(fs)
        per = {}
        for e in expressions:
            fe = fs[fs["expected"] == e]
            per[e] = {"acc": float((fe["pred"] == e).mean()) if len(fe) else None, "n": len(fe)}
        known = [v["acc"] for v in per.values() if v["acc"] is not None]
        spec = SPECS.get(model)
        models.append({
            "model": model,
            "label": spec.label if spec else model,
            "dataset": spec.dataset if spec else "—",
            "frames": len(sub),
            "face": n,
            "accuracy": k / n if n else None,
            "ci": wilson(k, n),
            "balanced": float(np.mean(known)) if known else None,
            "uncertain": float((fs["status"] == "uncertain").mean()) if n else None,
            "per_expression": per,
            "confusion": {e: fs[fs["expected"] == e]["pred"].value_counts().to_dict() for e in expressions},
        })
    models.sort(key=lambda m: (m["balanced"] is None, -(m["balanced"] or 0)))
    return {"expressions": expressions, "models": models, "frames": len(t) // max(1, t["model_name"].nunique())}


def primary_model(df: pd.DataFrame, cfg: AppConfig) -> str:
    counts = df["model_name"].value_counts()
    return cfg.model.name if cfg.model.name in counts.index else str(counts.index[0])


def compute(df: pd.DataFrame, cfg: AppConfig) -> dict[str, Any]:
    all_samples = df[df["record_type"] == SAMPLE]
    multi = df["model_name"].nunique() > 1
    primary = primary_model(df, cfg)
    # Əsas göstəricilər, keyfiyyət və saatlıq paylanma ƏSAS model üzrə — modellər qarışmasın.
    dp = df[df["model_name"] == primary] if multi else df
    samples = dp[dp["record_type"] == SAMPLE]
    n_operators = int(df["operator_id"].nunique())
    fps = cfg.capture.analysis_fps
    by_model = models_breakdown(df, cfg) if multi else []
    order = [primary] + [m["model"] for m in by_model if m["model"] != primary]
    return {
        "period": (df["date"].min(), df["date"].max()) if len(df) else (None, None),
        "days": int(df["date"].nunique()),
        "n_operators": n_operators,
        "small_group": n_operators < cfg.report.min_group_size,
        "n_samples": len(samples),
        "hours_monitored": len(samples) / fps / 3600 if fps else None,
        "models": sorted(df["model_name"].unique()),
        "primary": primary,
        "accuracy": accuracy(dp, cfg),
        "quality": quality(samples),
        "hourly": hourly(samples),
        "performance": performance(all_samples),
        "va": valence_arousal(dp, cfg),
        "by_model": by_model,
        "pairwise": model_agreement(all_samples, cfg, order) if multi else None,
        "calibration": calibration(df),
    }


# --- formatlama -----------------------------------------------------------------

esc = html.escape


def num(x: float | None, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    text = f"{x:,.{digits}f}"
    return text.replace(",", " ").replace(".", ",")


def pct(x: float | None, digits: int = 0) -> str:
    return "—" if x is None else f"{num(x * 100, digits)}%"


def ratio(k: int, n: int) -> float | None:
    return k / n if n else None


def category_label(code: str, cfg: AppConfig) -> str:
    if code == UNCERTAIN:
        return "Qeyri-müəyyən"
    if code == NO_PREDICTION:
        return "Üz yoxdur / toplanır"
    return cfg.self_report.categories.get(code, code)


def tip(*lines: str) -> str:
    return f'data-tip="{esc(chr(10).join(lines))}" tabindex="0"'


# --- qrafiklər (HTML/SVG, xarici kitabxana yoxdur) ------------------------------


def seq_bin(share: float) -> int:
    """Sıra payını 0–6 ardıcıl rəng pilləsinə çevirir (0 = boş)."""
    if share <= 0:
        return 0
    return min(6, 1 + int(share * 6))


def render_confusion(acc: dict[str, Any], cfg: AppConfig) -> str:
    cats, cols, matrix = acc["categories"], acc["columns"], acc["matrix"]
    head = "".join(f'<th scope="col">{esc(category_label(c, cfg))}</th>' for c in cols)
    body = []
    for r, row in zip(cats, matrix):
        total = sum(row)
        cells = []
        for c, k in zip(cols, row):
            share = k / total if total else 0.0
            diag = " diag" if r == c else ""
            muted = " side" if c in (UNCERTAIN, NO_PREDICTION) else ""
            label = pct(share) if total else "—"
            cells.append(
                f'<td class="cell b{seq_bin(share)}{diag}{muted}" '
                f'{tip(f"Operator: {category_label(r, cfg)}", f"Model: {category_label(c, cfg)}", f"{k} / {total} cavab · {pct(share)}")}>'
                f"{label}</td>"
            )
        unm = "" if r not in acc["unmappable"] else ' <span class="tag">modeldə yoxdur</span>'
        body.append(
            f'<tr><th scope="row">{esc(category_label(r, cfg))}{unm}</th>{"".join(cells)}'
            f'<td class="n num">{total}</td></tr>'
        )
    scale = "".join(f'<span class="b{i}"></span>' for i in range(7))
    return f"""
<div class="table-wrap">
<table class="confusion">
  <caption class="sr-only">Sətirlər — operatorun öz cavabı, sütunlar — modelin həmin andakı nəticəsi. Xanalarda sətir faizi.</caption>
  <thead><tr><th class="corner">Operator ↓ · Model →</th>{head}<th scope="col">n</th></tr></thead>
  <tbody>{"".join(body)}</tbody>
</table>
</div>
<div class="scale"><span>0%</span><div class="ramp">{scale}</div><span>100% — sətir payı</span></div>"""


def render_stacked_bar(items: list[tuple[str, str, int, float]], series_class: str = "s") -> str:
    """Bir üfüqi 100% zolaq. items: (açar, ad, say, pay)."""
    segs, legend = [], []
    for i, (key, label, k, share) in enumerate(items, start=1):
        if share > 0:
            inline = f'<span class="seg-label">{pct(share)}</span>' if share >= 0.08 else ""
            segs.append(
                f'<div class="seg {series_class}{i}" style="flex-grow:{share:.6f}" '
                f'{tip(label, f"{pct(share, 1)} · {num(k)} kadr")}>{inline}</div>'
            )
        legend.append(
            f'<li><span class="key {series_class}{i}"></span>{esc(label)}'
            f'<span class="lv num">{pct(share, 1)}</span></li>'
        )
    return f'<div class="stack">{"".join(segs)}</div><ul class="legend">{"".join(legend)}</ul>'


def render_hbars(items: list[tuple[str, int]], total: int) -> str:
    if not items:
        return '<p class="empty">Məlumat yoxdur</p>'
    top = max(k for _, k in items)
    rows = []
    for label, k in items:
        w = k / top if top else 0
        rows.append(
            f'<div class="hb-label">{esc(label)}</div>'
            f'<div class="hb-track"><div class="hb-bar" style="width:{w * 100:.2f}%" '
            f'{tip(label, f"{num(k)} kadr · {pct(ratio(k, total), 1)}")}></div></div>'
            f'<div class="hb-val num">{pct(ratio(k, total))}</div>'
        )
    return f'<div class="hbars">{"".join(rows)}</div>'


def render_hourly(hourly_data: dict[str, Any]) -> str:
    rows = hourly_data["rows"]
    if not rows:
        return '<p class="empty">Məlumat yoxdur</p>'
    cols = []
    for row in rows:
        h = row["hour"]
        if not row["enough"]:
            n = row["n"]
            cols.append(
                f'<div class="col thin" {tip(f"{h:02d}:00–{h:02d}:59", f"Az məlumat: {n} kadr (< {MIN_HOUR_SAMPLES})")}>'
                f'<div class="bars"></div><span class="x num">{h:02d}</span></div>'
            )
            continue
        segs = []
        for i, (key, label, _) in enumerate(HOURLY_SERIES, start=1):
            k, share = row["shares"][key]
            if share > 0:
                cls = "sg" if key == "uncertain" else f"s{i}"
                segs.append(
                    f'<div class="seg {cls}" style="flex-grow:{share:.6f}" '
                    f'{tip(f"{h:02d}:00 · {label}", f"{pct(share, 1)} · {num(k)} kadr")}></div>'
                )
        cols.append(
            f'<div class="col"><div class="bars">{"".join(segs)}</div>'
            f'<span class="x num">{h:02d}</span></div>'
        )
    legend = "".join(
        f'<li><span class="key {"sg" if key == "uncertain" else f"s{i}"}"></span>{esc(label)}</li>'
        for i, (key, label, _) in enumerate(HOURLY_SERIES, start=1)
    )
    grid = "".join(f'<div class="gl" style="bottom:{p}%"><span>{p}%</span></div>' for p in (0, 50, 100))
    return f"""
<ul class="legend">{legend}</ul>
<div class="hourly"><div class="grid">{grid}</div><div class="cols">{"".join(cols)}</div></div>"""


def hourly_table(hourly_data: dict[str, Any]) -> str:
    head = "".join(f"<th>{esc(label)}</th>" for _, label, _ in HOURLY_SERIES)
    body = []
    for row in hourly_data["rows"]:
        cells = "".join(f'<td class="num">{pct(row["shares"][key][1], 1)}</td>' for key, _, _ in HOURLY_SERIES)
        body.append(f'<tr><th>{row["hour"]:02d}:00</th><td class="num">{num(row["n"])}</td>{cells}</tr>')
    return f'<table class="data"><thead><tr><th>Saat</th><th>Kadr</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def render_model_comparison(data: dict[str, Any], cfg: AppConfig) -> str:
    """Modellərin operatorla uyğunluğu (eyni sorğular üzrə) + modellər arası razılaşma."""
    rows = []
    for m in data["by_model"]:
        agree = m["agreement"]
        ci = m["ci"]
        bar = ""
        if agree is not None:
            whisker = ""
            if ci:
                whisker = (f'<span class="ci" style="left:{ci[0] * 100:.1f}%;width:{(ci[1] - ci[0]) * 100:.1f}%"></span>')
            lo, hi = (pct(ci[0]), pct(ci[1])) if ci else ("—", "—")
            n_answers = m["n"]
            attrs = tip(m["label"], f"Uyğunluq {pct(agree, 1)}", f"95% interval {lo}–{hi}", f"{n_answers} cavab")
            bar = (f'<div class="ab" {attrs}>'
                   f'<span class="ab-fill" style="width:{agree * 100:.1f}%"></span>{whisker}</div>')
        primary = ' <span class="tag">əsas</span>' if m["model"] == data["primary"] else ""
        rows.append(
            f'<tr><th>{esc(m["label"])}{primary}<br><code class="sub-code">{esc(m["model"])}</code></th>'
            f'<td>{esc(m["dataset"])}</td>'
            f'<td class="agree-cell"><div class="agree-row"><span class="num">{pct(agree)}</span>{bar}</div></td>'
            f'<td class="num">{num(m["kappa"], 2) if m["kappa"] is not None else "—"}</td>'
            f'<td class="num">{pct(m["coverage"])}</td>'
            f'<td class="num">{pct(m["uncertain_share"])}</td>'
            f'<td class="num">{num(m["model_ms"], 1)} / {num(m["model_ms_p95"], 1)}</td>'
            f'<td class="num">{m["n"]}</td></tr>'
        )
    table = (
        '<div class="table-wrap"><table class="data models-table"><thead><tr><th>Model</th><th>Dataset</th>'
        "<th>Operatorla uyğunluq</th><th>Kappa</th><th>Əminlik</th><th>Qeyri-müəyyən</th><th>Model ms orta / p95</th>"
        f'<th>n</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )

    pair_html = ""
    pw = data.get("pairwise")
    if pw:
        labels = {m["model"]: m["label"] for m in data["by_model"]}
        head = "".join(f'<th scope="col">{esc(labels.get(b, b))}</th>' for b in pw["models"])
        body = []
        for a in pw["models"]:
            cells = []
            for b in pw["models"]:
                c = pw["cells"][(a, b)]
                if a == b:
                    cells.append('<td class="cell self">—</td>')
                    continue
                share = c["agree"] or 0.0
                kappa = num(c["kappa"], 2) if c["kappa"] is not None else "—"
                frames = num(c["n"])
                attrs = tip(f"{labels.get(a, a)} ↔ {labels.get(b, b)}", f"kappa {kappa} · {frames} kadr",
                            f"Eyni kateqoriya: {pct(share, 1)}")
                value = pct(c["agree"]) if c["n"] else "—"
                cells.append(f'<td class="cell b{seq_bin(share)}" {attrs}>{value}</td>')
            body.append(f'<tr><th scope="row">{esc(labels.get(a, a))}</th>{"".join(cells)}</tr>')
        scale = "".join(f'<span class="b{i}"></span>' for i in range(7))
        pair_html = f"""
  <h3>Modellər arası razılaşma</h3>
  <p class="lead">Eyni kadrda hər iki model müəyyən nəticə verdikdə, eyni operator kateqoriyasını deməsi faizi.
  Yüksək razılaşma modellərin doğru olduğunu yox, eyni şeyi gördüyünü göstərir.</p>
  <div class="table-wrap"><table class="confusion pairwise">
    <thead><tr><th class="corner"></th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>
  <div class="scale"><span>0%</span><div class="ramp">{scale}</div><span>100% — eyni kateqoriya payı</span></div>"""

    return f"""
<section class="card">
  <h2>Modellərin müqayisəsi</h2>
  <p class="lead">Bütün modellər eyni üz kəsiyini alıb və sorğu anında hamısının nəticəsi yazılıb — müqayisə eyni cavablar üzrədir.
  FER2013 modellərində “Həqarət” sinfi yoxdur. Uyğunluq faizinə interval (açıq zolaq) ilə birlikdə baxın.</p>
  {table}
  {pair_html}
</section>"""


def render_calibration(cal: dict[str, Any]) -> str:
    exprs = cal["expressions"]
    rows = []
    for i, m in enumerate(cal["models"]):
        acc, ci = m["accuracy"], m["ci"]
        bar = ""
        if acc is not None:
            whisker = f'<span class="ci" style="left:{ci[0] * 100:.1f}%;width:{(ci[1] - ci[0]) * 100:.1f}%"></span>' if ci else ""
            lo, hi = (pct(ci[0]), pct(ci[1])) if ci else ("—", "—")
            n_face = m["face"]
            attrs = tip(m["label"], f"Düzgünlük {pct(acc, 1)}", f"95% interval {lo}–{hi}", f"{n_face} üzlü kadr")
            bar = f'<div class="ab" {attrs}><span class="ab-fill" style="width:{acc * 100:.1f}%"></span>{whisker}</div>'
        ms = cal["models"]
        lead = len(ms) > 1 and (ms[0]["balanced"] or 0) > 0 and (ms[0]["balanced"] or 0) > (ms[1]["balanced"] or 0)
        best = ' <span class="tag">ən yaxşı</span>' if i == 0 and lead else ""
        rows.append(
            f'<tr><th>{esc(m["label"])}{best}<br><code class="sub-code">{esc(m["model"])}</code></th>'
            f'<td>{esc(m["dataset"])}</td>'
            f'<td class="agree-cell"><div class="agree-row"><span class="num">{pct(acc)}</span>{bar}</div></td>'
            f'<td class="num">{pct(m["balanced"])}</td><td class="num">{pct(m["uncertain"])}</td>'
            f'<td class="num">{m["face"]} / {m["frames"]}</td></tr>'
        )
    table = (
        '<div class="table-wrap"><table class="data models-table"><thead><tr><th>Model</th><th>Dataset</th>'
        "<th>Düzgünlük (bütün kadrlar)</th><th>Balanslı düzgünlük</th><th>Qeyri-müəyyən</th><th>Üzlü / cəmi kadr</th>"
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )

    head = "".join(f'<th scope="col">{esc(EMOTION_LABELS[e])}</th>' for e in exprs)
    body = []
    for m in cal["models"]:
        cells = []
        for e in exprs:
            v = m["per_expression"][e]
            if v["acc"] is None:
                cells.append('<td class="cell">—</td>')
                continue
            attrs = tip(f"{m['label']} · {EMOTION_LABELS[e]}", f"{pct(v['acc'], 1)} düzgün", f"{v['n']} kadr")
            cells.append(f'<td class="cell b{seq_bin(v["acc"])}" {attrs}>{pct(v["acc"])}</td>')
        body.append(f'<tr><th scope="row">{esc(m["label"])}</th>{"".join(cells)}</tr>')
    heat = (
        '<div class="table-wrap"><table class="confusion pairwise">'
        f'<thead><tr><th class="corner">Model ↓ · ifadə →</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    )

    best = cal["models"][0]
    pred_cols = [c for c in EMOTION_CLASSES if any(best["confusion"][e].get(c) for e in exprs)]
    chead = "".join(f'<th scope="col">{esc(EMOTION_LABELS[c])}</th>' for c in pred_cols)
    cbody = []
    for e in exprs:
        counts = best["confusion"][e]
        total = sum(counts.values())
        cells = []
        for c in pred_cols:
            share = counts.get(c, 0) / total if total else 0.0
            attrs = tip(f"Göstərilən: {EMOTION_LABELS[e]}", f"Model: {EMOTION_LABELS[c]}", f"{counts.get(c, 0)} / {total} kadr")
            diag = " diag" if c == e else ""
            cells.append(f'<td class="cell b{seq_bin(share)}{diag}" {attrs}>{pct(share) if total else "—"}</td>')
        cbody.append(f'<tr><th scope="row">{esc(EMOTION_LABELS[e])}</th>{"".join(cells)}<td class="n num">{total}</td></tr>')
    confusion = (
        f'<h3>Səhvlərin quruluşu — {esc(best["label"])}</h3>'
        '<div class="table-wrap"><table class="confusion">'
        f'<thead><tr><th class="corner">Göstərilən ↓ · Model →</th>{chead}<th scope="col">n</th></tr></thead>'
        f'<tbody>{"".join(cbody)}</tbody></table></div>'
    )
    scale = "".join(f'<span class="b{i}"></span>' for i in range(7))
    return f"""
<section class="card">
  <h2>Nəzarətli test — göstərilən ifadələr</h2>
  <p class="lead">Tester ekrandakı göstərişlə ifadələri saxlayıb; tapşırılan ifadə “düzgün cavab”dır. Hər kadrın öz
  nəticəsi qiymətləndirilir (10 s ortalama yox), yalnız üz görünən kadrlar sayılır. <strong>Balanslı düzgünlük</strong> —
  ifadələr üzrə düzgünlüklərin ortalaması (hər ifadə bərabər çəkidə). Pozalı ifadələr təbii ifadələrdən asandır —
  bu, yuxarı həddir; model seçimində sahədəki sorğu cavabları ilə birlikdə baxın.</p>
  {table}
  <h3>İfadələr üzrə</h3>
  {heat}
  {confusion}
  <div class="scale"><span>0%</span><div class="ramp">{scale}</div><span>100%</span></div>
</section>"""


def details(table_html: str) -> str:
    return f'<details class="tv"><summary>Cədvəl görünüşü</summary><div class="table-wrap">{table_html}</div></details>'


# --- səhifə ---------------------------------------------------------------------


def tile(label: str, value: str, note: str = "") -> str:
    return f'<div class="tile"><p class="t-label">{esc(label)}</p><p class="t-value">{value}</p><p class="t-note">{note}</p></div>'


def render_html(data: dict[str, Any], cfg: AppConfig, generated: datetime | None = None) -> str:
    generated = generated or datetime.now()
    acc, q, perf = data["accuracy"], data["quality"], data["performance"]
    k_comp, n_comp = acc["comparable"]
    k_conf, n_conf = acc["confident"]
    k_all, n_all = acc["overall"]
    agreement = ratio(k_comp, n_comp)
    ci = acc["comparable_ci"]
    p0, p1 = data["period"]
    period = f"{p0} — {p1}" if p0 and p0 != p1 else (p0 or "—")

    banners = []
    if data["small_group"]:
        banners.append(
            f'<div class="banner warn"><strong>Kiçik qrup.</strong> Məlumat {data["n_operators"]} operatordan gəlir '
            f'(hədd: {cfg.report.min_group_size}). Qrup göstəriciləri fərdi nəticəyə çox yaxın ola bilər — '
            "hesabatı bu səbəbdən geniş paylaşmayın.</div>"
        )
    if n_comp < 30:
        banners.append(
            f'<div class="banner info"><strong>Az cavab.</strong> Müqayisə üçün cəmi {n_comp} cavab var. '
            "Uyğunluq faizinin etibar intervalı genişdir; ən azı 100 cavab tövsiyə olunur.</div>"
        )

    primary_note = f" · əsas model <code>{esc(data['primary'])}</code>" if len(data["models"]) > 1 else ""
    hero = f"""
<section class="hero-card">
  <div>
    <p class="t-label">Model ilə operatorun öz cavabının uyğunluğu{primary_note}</p>
    <p class="hero-figure">{pct(agreement)}</p>
    <p class="t-note">{k_comp} / {n_comp} müqayisə oluna bilən cavab · 95% etibar intervalı
      {pct(ci[0]) if ci else "—"}–{pct(ci[1]) if ci else "—"}</p>
  </div>
  <div class="hero-side">
    {tile("Cohen kappa", num(acc["kappa"], 2) if acc["kappa"] is not None else "—", "0 = təsadüfi, 1 = tam")}
    {tile("Sadə bazis", pct(acc["baseline"]), "həmişə ən çox seçilən cavab")}
    {tile("Bütün cavablar üzrə", pct(ratio(k_all, n_all)), "qeyri-müəyyən və üz yoxdur = uyğunsuz")}
  </div>
</section>"""

    kpis = "".join([
        tile("Sorğu cavabları", num(acc["n_answered"]),
             f'{acc["n_prompts"]} sorğu · {acc["n_declined"]} imtina · {acc["n_timeout"]} cavabsız'),
        tile("Modelin əminliyi", pct(acc["coverage"]), "cavab anlarında müəyyən nəticə payı"),
        tile("“Üz yoxdur” payı", pct(q["share"]["no_face"], 1), f'{num(q["counts"]["no_face"])} kadr'),
        tile("“Qeyri-müəyyən” payı", pct(q["share"]["uncertain"], 1), f'{num(q["counts"]["uncertain"])} kadr'),
        tile("Gecikmə (orta / p95)", f'{num(perf["latency"]["mean"])} / {num(perf["latency"]["p95"])} ms', "detektor + model"),
        tile("CPU (orta / p95)", f'{num(perf["cpu"]["mean"], 1)} / {num(perf["cpu"]["p95"], 1)}%', "prosesin ümumi CPU payı"),
    ])

    per_cat_rows = "".join(
        f'<tr><th>{esc(c["label"])}{"" if c["mappable"] else " <span class=tag>modeldə yoxdur</span>"}</th>'
        f'<td class="num">{c["reports"]}</td><td class="num">{pct(c["recall"])}</td><td class="num">{pct(c["precision"])}</td></tr>'
        for c in acc["per_category"]
    )
    per_cat = (
        '<table class="data"><thead><tr><th>Kateqoriya</th><th>Cavab</th>'
        "<th>Tapılma (recall)</th><th>Dəqiqlik (precision)</th></tr></thead>"
        f"<tbody>{per_cat_rows}</tbody></table>"
    )

    models_html = render_model_comparison(data, cfg) if data["by_model"] else ""

    status_items = [(k, label, q["counts"][k], q["share"][k]) for k, label in STATUS_SERIES]
    status_table = (
        '<table class="data"><thead><tr><th>Vəziyyət</th><th>Kadr</th><th>Pay</th></tr></thead><tbody>'
        + "".join(f'<tr><th>{esc(lb)}</th><td class="num">{num(k)}</td><td class="num">{pct(s, 1)}</td></tr>' for _, lb, k, s in status_items)
        + "</tbody></table>"
    )
    reason_items = [(FACE_REASON_LABELS.get(r, r), k) for r, k in q["reasons"]]
    reason_table = (
        '<table class="data"><thead><tr><th>Səbəb</th><th>Kadr</th></tr></thead><tbody>'
        + "".join(f'<tr><th>{esc(lb)}</th><td class="num">{num(k)}</td></tr>' for lb, k in reason_items)
        + "</tbody></table>"
    )

    perf_rows = "".join(
        f'<tr><th><code>{esc(m["model"])}</code></th><td>{esc(m["engine"])} · {esc(m["device"])}</td>'
        f'<td class="num">{num(m["detector"]["mean"], 1)}</td><td class="num">{num(m["model_ms"]["mean"], 1)}</td>'
        f'<td class="num">{num(m["latency"]["mean"], 1)}</td><td class="num">{num(m["latency"]["p95"], 1)}</td>'
        f'<td class="num">{num(m["cpu"]["mean"], 1)}%</td><td class="num">{num(m["latency"]["n"])}</td></tr>'
        for m in perf["by_model"]
    )
    perf_table = (
        '<table class="data"><thead><tr><th>Model</th><th>Mühərrik</th><th>Detektor, ms</th><th>Model, ms</th>'
        "<th>Cəmi orta, ms</th><th>Cəmi p95, ms</th><th>CPU orta</th><th>Kadr</th></tr></thead>"
        f"<tbody>{perf_rows}</tbody></table>"
    )

    va_html = ""
    if data["va"]:
        rows = "".join(
            f'<tr><th>{esc(v["label"])}</th><td class="num">{v["n"]}</td>'
            f'<td class="num">{num(v["valence"], 2)}</td><td class="num">{num(v["arousal"], 2)}</td></tr>'
            for v in data["va"]
        )
        va_html = f"""
<section class="card">
  <h2>Valentlik və oyanma — operatorun cavabına görə</h2>
  <p class="lead">Yalnız <code>_va_mtl</code> modelləri. Hər kateqoriya üçün cavab anındakı orta dəyərlər
  (−1…+1). “Yorğun” cavablarında oyanmanın aşağı olması gözlənilir — bu, config-dəki ixtiyari VA qaydası üçün əsas verir.</p>
  <div class="table-wrap"><table class="data"><thead><tr><th>Operatorun cavabı</th><th>n</th><th>Valentlik</th><th>Oyanma</th></tr></thead>
  <tbody>{rows}</tbody></table></div>
</section>"""

    mapping_rows = "".join(
        f"<li>{esc(EMOTION_LABELS[c])} → {esc(cfg.self_report.categories[cfg.mapping[c]])}</li>" for c in EMOTION_CLASSES
    )
    va_note = (
        f"Aktivdir: oyanma ≤ {num(cfg.report.va_tired_rule.arousal_max, 2)} olduqda “Yorğun” sayılır."
        if cfg.report.va_tired_rule.enabled
        else "Söndürülüb."
    )

    body = f"""
<header class="page-head">
  <p class="eyebrow">Emosiya Pilotu · qiymətləndirmə hesabatı</p>
  <h1>Pilot hesabatı</h1>
  <p class="sub">{esc(period)} · {data["days"]} gün · {data["n_operators"]} operator ·
    ≈{num(data["hours_monitored"], 1)} saat müşahidə · {esc(", ".join(data["models"]))}</p>
  <p class="chips"><span class="chip">Yalnız qrup səviyyəsi</span><span class="chip">Fərdi reytinq yoxdur</span>
    <span class="chip">Video/şəkil saxlanılmayıb</span></p>
</header>
{"".join(banners)}
{hero}
<section class="tiles">{kpis}</section>
{models_html}
{render_calibration(data["calibration"]) if data.get("calibration") else ""}
<section class="card">
  <h2>Model nəticəsi və operatorun öz cavabı{f" — {esc(data['primary'])}" if len(data["models"]) > 1 else ""}</h2>
  <p class="lead">Sətirlər operatorun seçdiyi əhvaldır, sütunlar modelin həmin anda (sorğu göstəriləndə) verdiyi smoothed nəticədir.
  Diaqonal — uyğunluq. Modelin 8 sinfi config-dəki mapping ilə operator kateqoriyalarına çevrilir.</p>
  {render_confusion(acc, cfg)}
  <h3>Kateqoriyalar üzrə</h3>
  <div class="table-wrap">{per_cat}</div>
</section>

<section class="card">
  <h2>Məlumat keyfiyyəti</h2>
  <p class="lead">Analiz olunan bütün kadrların vəziyyəti. “Üz yoxdur” olanda model işə salınmır və təxmin edilmir.</p>
  {render_stacked_bar([(k, lb, n, s) for k, lb, n, s in status_items])}
  {details(status_table)}
  <h3>“Üz yoxdur” səbəbləri</h3>
  {render_hbars(reason_items, q["n_no_face"])}
  {details(reason_table)}
</section>

<section class="card">
  <h2>Gün ərzində emosiya paylanması</h2>
  <p class="lead">Bütün operatorlar birlikdə, saat üzrə modelin smoothed nəticələrinin payı (üz olan kadrlar).
  {MIN_HOUR_SAMPLES} kadrdan az olan saatlar boş göstərilir.</p>
  {render_hourly(data["hourly"])}
  {details(hourly_table(data["hourly"]))}
</section>

<section class="card">
  <h2>Performans</h2>
  <p class="lead">Bir analiz kadrının emal vaxtı (üz aşkarlama + model) və prosesin CPU istifadəsi.</p>
  <div class="tiles inner">
    {tile("Detektor (orta / p95)", f'{num(perf["detector"]["mean"], 1)} / {num(perf["detector"]["p95"], 1)} ms')}
    {tile("Model (orta / p95)", f'{num(perf["model_ms"]["mean"], 1)} / {num(perf["model_ms"]["p95"], 1)} ms')}
    {tile("Cəmi (p50)", f'{num(perf["latency"]["p50"], 1)} ms')}
  </div>
  <div class="table-wrap">{perf_table}</div>
</section>
{va_html}
<section class="card notes">
  <h2>Metodologiya və məhdudiyyətlər</h2>
  <ul>
    <li><strong>Model daxili hissi yox, üz ifadəsini ölçür.</strong> Operatorun öz cavabı ilə uyğunsuzluq həmişə modelin səhvi demək deyil.</li>
    <li>Müqayisə yalnız modelin <em>müəyyən</em> nəticə verdiyi və kateqoriyanın modeldə qarşılığı olan cavablar üzrədir.
      {f"Qarşılığı olmayan: {esc(', '.join(cfg.self_report.categories[c] for c in acc['unmappable']))}." if acc["unmappable"] else ""}</li>
    <li>Operator ekranda modelin nəticəsini görürsə, cavabı ona “uyğunlaşa” bilər. Qərəzsiz ölçü üçün <code>ui.show_prediction: false</code> (kor rejim) tövsiyə olunur.</li>
    <li>Etibar intervalı Wilson üsulu ilədir. Kappa təsadüfi uyğunluğu çıxır; 0,4-dən aşağı dəyər zəif razılıq sayılır.</li>
    <li>VA “Yorğun” qaydası: {va_note}</li>
    <li>Mapping: <ul class="mapping">{mapping_rows}</ul></li>
    <li>Model çəkiləri AffectNet (qeyri-kommersiya akademik lisenziya) üzərində öyrədilib — kommersiya istifadəsindən əvvəl hüquqi yoxlama lazımdır.</li>
  </ul>
  <p class="foot">Yaradılıb: {generated.strftime("%Y-%m-%d %H:%M")} · {num(data["n_samples"])} analiz kadrı.
  Bu hesabatda operator ID-ləri, fərdi göstəricilər və sıralama yoxdur.</p>
</section>"""
    return PAGE.replace("{{BODY}}", body)


PAGE = """<!doctype html>
<html lang="az">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:">
<title>Pilot hesabatı</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --sg: #c3c2b7;
  --b0: transparent; --b1: #cde2fb; --b2: #9ec5f4; --b3: #6da7ec; --b4: #3987e5; --b5: #256abf; --b6: #184f95;
  --b-ink-dark: #0b0b0b; --b-ink-light: #ffffff;
  --warn-bg: #fff4dc; --warn-ink: #6b4a00; --info-bg: #e8f1fc; --info-ink: #184f95;
  --font: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; --sg: #5c5b56;
    --b1: #0d366b; --b2: #104281; --b3: #184f95; --b4: #256abf; --b5: #3987e5; --b6: #6da7ec;
    --warn-bg: #3a2c0a; --warn-ink: #f5d48a; --info-bg: #0f2540; --info-ink: #9ec5f4;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 var(--font); -webkit-font-smoothing: antialiased; }
main { max-width: 1040px; margin: 0 auto; padding: 40px 16px 64px; }
h1 { font-size: 34px; line-height: 1.1; letter-spacing: -0.025em; margin: 4px 0 8px; }
h2 { font-size: 20px; letter-spacing: -0.015em; margin: 0 0 6px; }
h3 { font-size: 15px; margin: 24px 0 10px; }
code { font: 13px ui-monospace, "Cascadia Mono", Consolas, monospace; }
.eyebrow { margin: 0; color: var(--muted); font-size: 13px; font-weight: 600; letter-spacing: .02em; text-transform: uppercase; }
.sub, .lead { color: var(--ink-2); margin: 0 0 14px; }
.lead { max-width: 72ch; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 0; }
.chip { font-size: 12px; padding: 4px 10px; border-radius: 999px; background: var(--surface); box-shadow: inset 0 0 0 1px var(--ring); color: var(--ink-2); }
.page-head { margin-bottom: 20px; }
.card, .hero-card, .tile { background: var(--surface); border-radius: 18px; box-shadow: inset 0 0 0 1px var(--ring), 0 1px 2px rgba(0,0,0,.03); }
.card { padding: 22px; margin: 16px 0; }
.banner { padding: 12px 16px; border-radius: 14px; margin: 10px 0; font-size: 14px; }
.banner.warn { background: var(--warn-bg); color: var(--warn-ink); }
.banner.info { background: var(--info-bg); color: var(--info-ink); }
.hero-card { padding: 24px; display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 2fr); gap: 20px; align-items: center; margin: 16px 0; }
.hero-figure { font-size: 64px; font-weight: 700; letter-spacing: -0.03em; line-height: 1; margin: 6px 0; }
.hero-side { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.hero-side .tile { box-shadow: none; background: var(--page); }
.tiles { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.tiles.inner { margin-bottom: 16px; }
.tiles.inner .tile { background: var(--page); box-shadow: none; }
.tile { padding: 14px 16px; }
.t-label { margin: 0; font-size: 13px; color: var(--ink-2); }
.t-value { margin: 4px 0 2px; font-size: 24px; font-weight: 650; letter-spacing: -0.02em; }
.t-note { margin: 0; font-size: 12px; color: var(--muted); }
.num { font-variant-numeric: tabular-nums; }
.table-wrap { overflow-x: auto; }
table { border-collapse: separate; border-spacing: 0; width: 100%; font-size: 13px; }
table.data th, table.data td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--grid); }
table.data td.num, table.data th.num { text-align: right; }
table.data thead th { color: var(--ink-2); font-weight: 600; }
.tag { font-size: 11px; font-weight: 500; color: var(--ink-2); background: var(--page); border-radius: 6px; padding: 1px 6px; margin-left: 4px; white-space: nowrap; }
/* confusion heatmap */
table.confusion { border-spacing: 2px; min-width: 720px; }
.confusion th { font-weight: 500; color: var(--ink-2); padding: 6px 8px; font-size: 12px; text-align: center; }
.confusion th[scope=row] { text-align: left; color: var(--ink); font-size: 13px; white-space: nowrap; }
.confusion .corner { text-align: left; color: var(--muted); }
.confusion td.cell { height: 44px; text-align: center; border-radius: 6px; font-variant-numeric: tabular-nums; background: var(--page); color: var(--ink-2); outline: none; }
.confusion td.cell:hover, .confusion td.cell:focus { box-shadow: inset 0 0 0 2px var(--ink); }
.confusion td.diag { font-weight: 700; }
.confusion td.n { color: var(--muted); text-align: center; }
.confusion td.cell.b1 { background: var(--b1); color: var(--b-ink-dark); }
.confusion td.cell.b2 { background: var(--b2); color: var(--b-ink-dark); }
.confusion td.cell.b3 { background: var(--b3); color: var(--b-ink-dark); }
.confusion td.cell.b4 { background: var(--b4); color: var(--b-ink-light); }
.confusion td.cell.b5 { background: var(--b5); color: var(--b-ink-light); }
.confusion td.cell.b6 { background: var(--b6); color: var(--b-ink-light); }
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .confusion td.cell:is(.b1, .b2, .b3, .b4) { color: #ffffff; }
  :root:where(:not([data-theme="light"])) .confusion td.cell:is(.b5, .b6) { color: #0b0b0b; }
}
.scale { display: flex; align-items: center; gap: 10px; font-size: 12px; color: var(--muted); margin-top: 10px; }
.ramp { display: flex; gap: 2px; }
.ramp span { width: 28px; height: 10px; border-radius: 3px; background: var(--page); }
.ramp .b1 { background: var(--b1); } .ramp .b2 { background: var(--b2); } .ramp .b3 { background: var(--b3); }
.ramp .b4 { background: var(--b4); } .ramp .b5 { background: var(--b5); } .ramp .b6 { background: var(--b6); }
/* stacked bar */
.stack { display: flex; gap: 2px; height: 28px; margin: 6px 0 12px; }
.stack .seg { min-width: 2px; display: flex; align-items: center; justify-content: center; outline: none; }
.stack .seg:first-child { border-radius: 4px 0 0 4px; } .stack .seg:last-child { border-radius: 0 4px 4px 0; }
.stack .seg:only-child { border-radius: 4px; }
.seg-label { font-size: 12px; font-weight: 600; color: #fff; white-space: nowrap; }
.s3 .seg-label, .s4 .seg-label, .s5 .seg-label { color: #0b0b0b; }
.s1 { background: var(--s1); } .s2 { background: var(--s2); } .s3 { background: var(--s3); }
.s4 { background: var(--s4); } .s5 { background: var(--s5); } .s6 { background: var(--s6); } .sg { background: var(--sg); }
.seg:hover, .seg:focus { filter: brightness(1.12); }
.legend { list-style: none; padding: 0; margin: 0 0 8px; display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--ink-2); }
.legend li { display: flex; align-items: center; gap: 6px; }
.legend .key { width: 10px; height: 10px; border-radius: 3px; }
.legend .lv { color: var(--ink); font-weight: 600; }
/* horizontal bars */
.hbars { display: grid; grid-template-columns: minmax(120px, 220px) 1fr 52px; gap: 8px 12px; align-items: center; }
.hb-label { font-size: 13px; color: var(--ink-2); }
.hb-track { height: 14px; }
.hb-bar { height: 14px; background: var(--s1); border-radius: 0 4px 4px 0; min-width: 2px; outline: none; }
.hb-bar:hover, .hb-bar:focus { filter: brightness(1.12); }
.hb-val { font-size: 13px; text-align: right; }
/* hourly */
.hourly { position: relative; height: 240px; margin: 12px 0 8px 36px; }
.hourly .grid { position: absolute; inset: 0 0 22px 0; }
.gl { position: absolute; left: 0; right: 0; height: 1px; background: var(--grid); }
.gl span { position: absolute; left: -36px; top: -8px; font-size: 11px; color: var(--muted); width: 30px; text-align: right; font-variant-numeric: tabular-nums; }
.hourly .cols { position: absolute; inset: 0; display: flex; justify-content: space-around; align-items: stretch; }
.col { display: flex; flex-direction: column; align-items: center; flex: 1; max-width: 40px; }
.col .bars { flex: 1; width: min(22px, 70%); display: flex; flex-direction: column-reverse; gap: 2px; margin-bottom: 4px; }
.col .bars .seg { min-height: 1px; outline: none; }
.col .bars .seg:last-child { border-radius: 4px 4px 0 0; }
.col .x { font-size: 11px; color: var(--muted); height: 18px; }
.col.thin .x { opacity: .45; }
/* modellərin müqayisəsi */
.models-table td, .models-table th { vertical-align: middle; }
.sub-code { font-size: 11px; color: var(--muted); }
.agree-row { display: flex; align-items: center; gap: 10px; min-width: 180px; }
.agree-row .num { width: 38px; text-align: right; font-weight: 600; }
.ab { position: relative; flex: 1; height: 14px; outline: none; }
.ab::before { content: ""; position: absolute; inset: 5px 0; background: var(--grid); border-radius: 2px; }
.ab-fill { position: absolute; left: 0; top: 0; height: 14px; background: var(--s1); border-radius: 0 4px 4px 0; }
.ab .ci { position: absolute; top: 5px; height: 4px; background: var(--ink); opacity: .35; border-radius: 2px; }
.ab:hover .ab-fill, .ab:focus .ab-fill { filter: brightness(1.12); }
table.pairwise { min-width: 0; width: auto; }
.pairwise td.cell { min-width: 84px; }
.pairwise td.self { color: var(--muted); background: transparent; }
details.tv { margin: 6px 0 4px; }
details.tv summary { cursor: pointer; font-size: 13px; color: var(--s1); width: fit-content; }
details.tv[open] summary { margin-bottom: 8px; }
.empty { color: var(--muted); }
.notes ul { padding-left: 20px; margin: 8px 0; }
.notes li { margin: 6px 0; color: var(--ink-2); }
.notes li strong { color: var(--ink); }
.mapping { columns: 2; padding-left: 18px; }
.foot { font-size: 12px; color: var(--muted); margin-top: 16px; }
.sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
#tip { position: fixed; pointer-events: none; z-index: 10; background: var(--surface); color: var(--ink); font-size: 12px; line-height: 1.45;
  padding: 8px 10px; border-radius: 10px; box-shadow: 0 6px 24px rgba(0,0,0,.18), inset 0 0 0 1px var(--ring); max-width: 260px; opacity: 0; transition: opacity .12s; }
#tip.show { opacity: 1; }
#tip b { display: block; font-size: 13px; }
@media (max-width: 760px) {
  .hero-card { grid-template-columns: 1fr; }
  .hero-side, .tiles { grid-template-columns: 1fr 1fr; }
  .hero-figure { font-size: 52px; }
  .hbars { grid-template-columns: 110px 1fr 44px; }
}
@media print { #tip { display: none; } details.tv { display: none; } .card, .hero-card { break-inside: avoid; } }
</style>
</head>
<body>
<main>
{{BODY}}
</main>
<div id="tip" role="tooltip"></div>
<script>
(() => {
  const tipEl = document.getElementById("tip");
  function show(el, x, y) {
    const lines = el.getAttribute("data-tip").split("\\n");
    tipEl.replaceChildren();
    lines.forEach((line, i) => {
      const node = document.createElement(i === lines.length - 1 && lines.length > 1 ? "b" : "span");
      node.textContent = line;
      tipEl.appendChild(node);
      if (node.tagName === "SPAN") tipEl.appendChild(document.createElement("br"));
    });
    const r = tipEl.getBoundingClientRect();
    tipEl.style.left = Math.min(window.innerWidth - r.width - 8, x + 14) + "px";
    tipEl.style.top = Math.max(8, y - r.height - 10) + "px";
    tipEl.classList.add("show");
  }
  document.addEventListener("pointermove", (e) => {
    const el = e.target.closest("[data-tip]");
    if (el) show(el, e.clientX, e.clientY); else tipEl.classList.remove("show");
  });
  document.addEventListener("focusin", (e) => {
    const el = e.target.closest("[data-tip]");
    if (!el) return;
    const r = el.getBoundingClientRect();
    show(el, r.left + r.width / 2, r.top);
  });
  document.addEventListener("focusout", () => tipEl.classList.remove("show"));
})();
</script>
</body>
</html>
"""


def write_report(html_text: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.report", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="config.yaml yolu")
    parser.add_argument("--logs", help="CSV qovluğu və ya faylı (default: logging.dir)")
    parser.add_argument("--out", help="HTML faylı (default: report.dir/hesabat_<tarix>.html)")
    parser.add_argument("--from", dest="date_from", help="başlanğıc tarix YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", help="son tarix YYYY-MM-DD")
    parser.add_argument("--model", help="yalnız bu modelin nəticələri")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config(args.config)
    logs = Path(args.logs) if args.logs else cfg.logging.dir
    try:
        df = filter_logs(load_logs(find_logs(logs)), args.date_from, args.date_to, args.model)
        if df.empty:
            raise ReportError("Seçilmiş filtrə uyğun sətir yoxdur")
    except ReportError as exc:
        print(f"Xəta: {exc}", file=sys.stderr)
        return 1

    data = compute(df, cfg)
    out = Path(args.out) if args.out else cfg.report.dir / f"hesabat_{datetime.now():%Y-%m-%d}.html"
    write_report(render_html(data, cfg), out)
    acc = data["accuracy"]
    k, n = acc["comparable"]
    print(f"Hesabat: {out}")
    print(f"  {data['n_samples']} kadr, {acc['n_answered']} sorğu cavabı, uyğunluq {pct(ratio(k, n))} ({k}/{n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
