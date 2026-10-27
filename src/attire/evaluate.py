"""Geyim yoxlaması: şəkil qovluğu → hər şəklin nəticəsi, dəqiqlik, CSV.

    python -m src.attire geyim_test/                    # resmi/ və qeyri-resmi/ varsa — dəqiqlik ölçülür
    python -m src.attire kadrlar/                       # etiketsiz — yalnız nəticələr
    python -m src.attire geyim_test/ --out reports/x.csv --folds 0

Etiketli qovluq:
    geyim_test/
      resmi/         (və ya rəsmi/)        düyməli köynək / bluzka (istənilən rəng) və ya pencək
      qeyri-resmi/   (və ya qeyri-rəsmi/)  futbolka, polo, sviter, hudi…

Etiketli qovluqda üç şey ölçülür: indiki hədlərlə dəqiqlik, eyni şəkillərdə ən yaxşı hədlər və
SigLIP vektorları üzərində öyrədilmiş kiçik təsnifatçının çarpaz yoxlaması (zero-shot çatmasa,
öz data ilə öyrətməyin nə qazandıracağını göstərir). Fayl adı "P01_1.jpg" kimidirsə, "_"-dan
əvvəlki hissə şəxs sayılır: eyni şəxsin şəkilləri ya yalnız öyrətməyə, ya yalnız testə düşür —
əks halda dəqiqlik şişirdilmiş çıxar.

Şəkillər yalnız oxunur. Diskə yalnız CSV yazılır (fayl adı, nəticə, ballar); şəbəkə bağlıdır.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ..config import AppConfig, AttireConfig, load_config
from ..report import wilson
from .crop import find_torso
from .encoder import Models
from .verdict import FORMAL, NOT_FORMAL, UNKNOWN, Verdict, decide, is_formal, label, split_scores

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_DIRS = {"resmi": FORMAL, "rəsmi": FORMAL, "qeyri-resmi": NOT_FORMAL, "qeyri-rəsmi": NOT_FORMAL}
UNREADABLE = "unreadable"
CLASSES = (FORMAL, NOT_FORMAL)

# Kalibrləmə şəbəkəsi
FORMAL_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
JACKET_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
LAYERING_GRID = (0.0, 0.15, 0.25, 0.35, 0.5)
WHITE_GRID = (0.2, 0.3, 0.4, 0.5, 0.6)
SHARE_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
COLOR_RULES = ("both", "model", "pixels")

RIDGE_LAMBDA_PER_DIM = 1.0  # standartlaşdırılmış vektorlarda; az data üçün güclü requlyarizasiya
MIN_CV_IMAGES = 10
MAX_ERRORS_SHOWN = 30
CSV_COLUMNS = ["file", "label", "result", "reason", "basis", "torso", "garment", "formal_p", "color", "white_p",
               "white_share", "tie_p", "jacket_p", "layering", "jacket", "ms"]


# --- şəkillər ---------------------------------------------------------------------------


@dataclass
class Item:
    path: Path
    rel: str  # qovluğa nisbətən, CSV-də yalnız bu yazılır
    label: str | None  # formal | not_formal | None (etiketsiz)
    group: str  # şəxs — çarpaz yoxlamada bölünməz vahid


def discover(root: Path) -> tuple[list[Item], bool]:
    """Şəkillər və qovluğun etiketli olub-olmadığı."""
    dirs = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
    labeled = [(d, LABEL_DIRS[name]) for name, d in sorted(dirs.items()) if name in LABEL_DIRS]
    items = []
    for folder, lab in labeled or [(root, None)]:
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXT:
                rel = p.relative_to(root).as_posix()
                items.append(Item(p, rel, lab, p.stem.split("_", 1)[0] if "_" in p.stem else rel))
    return items, bool(labeled)


def read_image(path: Path) -> np.ndarray | None:
    """Unicode yolları da oxuyur (cv2.imread Windows-da "ə" olan yolu aça bilmir)."""
    try:
        data = np.fromfile(path, np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


# --- analiz -----------------------------------------------------------------------------


def load_models(cfg: AppConfig) -> Models:
    from ..detector import FaceDetector
    from .encoder import SiglipEncoder, ZeroShot

    encoder = SiglipEncoder(cfg.attire.model_dir)
    return Models(FaceDetector(cfg.detector), encoder, ZeroShot(encoder, cfg.attire.prompts))


@dataclass
class Row:
    item: Item
    verdict: Verdict
    embedding: np.ndarray | None
    ms: float


def analyze(items: list[Item], models: Models, cfg: AttireConfig,
            progress: Callable[[int, int], None] | None = None) -> list[Row]:
    labels = {q: zq.labels for q, zq in models.zero_shot.questions.items()}
    rows = []
    for i, item in enumerate(items, 1):
        start = time.perf_counter()
        image = read_image(item.path)
        emb = None
        if image is None:
            verdict = Verdict(UNKNOWN, "şəkil oxunmadı", UNREADABLE)
        else:
            torso = find_torso(image, models.detector, cfg)
            scores = None
            if torso.ok:
                emb = models.encoder.encode_images([torso.crop])[0]
                scores = split_scores(models.zero_shot(emb[None]), labels)[0]
            verdict = decide(torso, scores, cfg)
        rows.append(Row(item, verdict, emb, (time.perf_counter() - start) * 1000.0))
        if progress:
            progress(i, len(items))
    return rows


# --- metrikalar -------------------------------------------------------------------------


def metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    n = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    recalls = []
    for c in CLASSES:
        idx = [i for i, t in enumerate(y_true) if t == c]
        if idx:
            recalls.append(sum(y_pred[i] == c for i in idx) / len(idx))
    return {
        "n": n,
        "acc": correct / n if n else None,
        "ci": wilson(correct, n),
        "balanced": float(np.mean(recalls)) if recalls else None,
        "cm": Counter(zip(y_true, y_pred)),
    }


def scored_rows(rows: list[Row]) -> list[Row]:
    """Etiketli və qiymətləndirilə bilən şəkillər."""
    return [r for r in rows if r.item.label and r.verdict.result != UNKNOWN]


def rule_predict(v: Verdict, cfg: AttireConfig) -> str:
    return FORMAL if is_formal(v, cfg) else NOT_FORMAL


def rule_params(cfg: AttireConfig) -> dict[str, Any]:
    """Qərara təsir edən ayarlar (rəng ayarları yalnız require_white-də)."""
    p: dict[str, Any] = {"formal_threshold": cfg.formal_threshold, "jacket_is_formal": cfg.jacket_is_formal}
    if cfg.jacket_is_formal:
        p["jacket_threshold"] = cfg.jacket_threshold
        p["jacket_min_layering"] = cfg.jacket_min_layering
    if cfg.require_white:
        p.update(white_threshold=cfg.white_threshold, white_min_share=cfg.white_min_share, color_rule=cfg.color_rule)
    return p


def format_params(p: dict[str, Any]) -> str:
    fmt = lambda v: str(v).lower() if isinstance(v, bool) else f"{v:g}" if isinstance(v, float) else str(v)  # noqa: E731
    return " · ".join(f"{k} {fmt(v)}" for k, v in p.items())


def _grid(cfg: AttireConfig):
    jackets = [{"jacket_is_formal": False}] + [{"jacket_is_formal": True, "jacket_threshold": t, "jacket_min_layering": m}
                                               for t, m in product(JACKET_GRID, LAYERING_GRID)]
    whites = [{}]
    if cfg.require_white:
        whites = [dict(white_threshold=w, white_min_share=s, color_rule=r)
                  for w, s, r in product(WHITE_GRID, SHARE_GRID, COLOR_RULES)]
    for f, j, w in product(FORMAL_GRID, jackets, whites):
        yield {"formal_threshold": f, **j, **w}


def calibrate(rows: list[Row], cfg: AttireConfig) -> dict[str, Any] | None:
    """Saxlanılmış ballardan hədləri yenidən sınayır. Eyni şəkillərdə seçildiyi üçün optimistdir."""
    scored = scored_rows(rows)
    y = [r.item.label for r in scored]
    if len(set(y)) < 2:
        return None
    best = None
    for params in _grid(cfg):
        variant = replace(cfg, **params)
        m = metrics(y, [rule_predict(r.verdict, variant) for r in scored])
        if best is None or (m["balanced"], m["acc"]) > (best["metrics"]["balanced"], best["metrics"]["acc"]):
            best = {"params": rule_params(variant), "metrics": m}
    return best


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Ridge təsnifatçı (hədəf ±1). n < d olduqda dual həll — 100 şəkil × 768 ölçü üçün sürətli."""
    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-6
    z = (x - mu) / sd
    yc = y - y.mean()
    n, d = z.shape
    if n < d:
        w = z.T @ np.linalg.solve(z @ z.T + lam * np.eye(n), yc)
    else:
        w = np.linalg.solve(z.T @ z + lam * np.eye(d), z.T @ yc)
    return mu, sd, w, float(y.mean())


def ridge_predict(model: tuple[np.ndarray, np.ndarray, np.ndarray, float], x: np.ndarray) -> np.ndarray:
    mu, sd, w, b = model
    return ((x - mu) / sd) @ w + b


def group_folds(groups: list[str], labels: list[str], k: int, seed: int = 0) -> list[np.ndarray]:
    """Şəxslər qatlara bölünür (eyni şəxs bir qatda), sinif nisbəti qatlarda təxminən eyni."""
    first_label: dict[str, str] = {}
    for g, lab in zip(groups, labels):
        first_label.setdefault(g, lab)
    uniq = sorted(first_label)
    np.random.default_rng(seed).shuffle(uniq)
    ordered = sorted(uniq, key=lambda g: first_label[g])  # sabit sıralama: sinif daxilində təsadüfi
    k = min(k, len(ordered))
    fold_of = {g: i % k for i, g in enumerate(ordered)}
    return [np.array([i for i, g in enumerate(groups) if fold_of[g] == f]) for f in range(k)]


def cross_validate(rows: list[Row], k: int) -> dict[str, Any] | None:
    """Öyrədilmiş təsnifatçı (B): SigLIP vektorları üzərində ridge, şəxslərə görə k-qatlı yoxlama."""
    scored = [r for r in scored_rows(rows) if r.embedding is not None]
    y = [r.item.label for r in scored]
    groups = [r.item.group for r in scored]
    if k < 2 or len(scored) < MIN_CV_IMAGES or len(set(y)) < 2 or len(set(groups)) < 2:
        return None
    x = np.stack([r.embedding for r in scored]).astype(np.float64)
    target = np.array([1.0 if lab == FORMAL else -1.0 for lab in y])
    folds = group_folds(groups, y, k)
    pred = np.empty(len(scored), dtype=object)
    for test in folds:
        train = np.setdiff1d(np.arange(len(scored)), test)
        if len(set(target[train])) < 2:  # öyrətmədə tək sinif — həmin sinfi proqnozlaşdır
            pred[test] = FORMAL if target[train][0] > 0 else NOT_FORMAL
            continue
        model = ridge_fit(x[train], target[train], RIDGE_LAMBDA_PER_DIM * x.shape[1])
        pred[test] = np.where(ridge_predict(model, x[test]) >= 0, FORMAL, NOT_FORMAL)
    zero_shot = metrics(y, [r.verdict.result for r in scored])
    return {"metrics": metrics(y, list(pred)), "zero_shot": zero_shot, "folds": len(folds),
            "groups": len(set(groups))}


# --- çıxış ------------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


def _f(x: float | None, digits: int = 4) -> str:
    return "" if x is None else f"{x:.{digits}f}"


def write_csv(rows: list[Row], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        for r in rows:
            v = r.verdict
            w.writerow([r.item.rel, r.item.label or "", v.result, v.reason, v.basis, v.torso, v.garment,
                        _f(v.formal_p), v.color, _f(v.white_p), _f(v.white_share), _f(v.tie_p), _f(v.jacket_p),
                        _f(v.layering), "" if v.jacket_seen is None else int(v.jacket_seen), _f(r.ms, 1)])
    return path


def _metrics_lines(m: dict[str, Any], indent: str = "  ") -> list[str]:
    ci = m["ci"]
    ci_text = f" (95% interval {ci[0]:.0%}–{ci[1]:.0%})" if ci else ""
    cm = m["cm"]
    return [
        f"{indent}Dəqiqlik {_pct(m['acc'])}{ci_text} · balanslı {_pct(m['balanced'])} · {m['n']} şəkil",
        f"{indent}{'':<14}proqnoz: {label(FORMAL):<8} {label(NOT_FORMAL)}",
        *[f"{indent}{label(t):<22} {cm[(t, FORMAL)]:>3}       {cm[(t, NOT_FORMAL)]:>3}" for t in CLASSES],
    ]


def format_console(rows: list[Row], labeled: bool, cfg: AttireConfig, calib: dict | None = None,
                   cv: dict | None = None) -> str:
    n = len(rows)
    results = Counter(r.verdict.result for r in rows)
    ms = [r.ms for r in rows if r.embedding is not None]
    lines = [f"Geyim yoxlaması — {n} şəkil" + (f" · {np.mean(ms) / 1000:.2f} s/şəkil" if ms else "")]
    if labeled:
        by_label = Counter(r.item.label for r in rows)
        lines[0] += f" (etiket: {label(FORMAL).lower()} {by_label[FORMAL]}, {label(NOT_FORMAL).lower()} " \
                    f"{by_label[NOT_FORMAL]})"
    assessable = n - results[UNKNOWN]
    lines.append(f"\nQiymətləndirilə bildi: {assessable} / {n} ({_pct(assessable / n if n else None)})")
    unknown = Counter(r.verdict.reason for r in rows if r.verdict.result == UNKNOWN)
    lines += [f"  {reason}: {count}" for reason, count in unknown.most_common()]
    if not labeled:
        lines.append(f"\n{label(FORMAL)}: {results[FORMAL]} · {label(NOT_FORMAL)}: {results[NOT_FORMAL]}")
    ok = [r for r in rows if r.verdict.result != UNKNOWN]
    if ok:
        tie = sum(r.verdict.tie_p >= 0.5 for r in ok)
        jacket = sum(bool(r.verdict.jacket_seen) for r in ok)
        note = "pencək rəsmi sayılır" if cfg.jacket_is_formal else "pencək nəticəyə təsir etmir"
        lines.append(f"Pencək: {jacket} · qalstuk: {tie} ({note}; qalstuk yalnız məlumatdır)")
    if not labeled:
        return "\n".join(lines)

    scored = scored_rows(rows)
    if not scored:
        lines.append("\nQiymətləndirilə bilən etiketli şəkil yoxdur.")
        return "\n".join(lines)
    lines.append(f"\nZero-shot (indiki ayarlar: {format_params(rule_params(cfg))}):")
    lines += _metrics_lines(metrics([r.item.label for r in scored], [r.verdict.result for r in scored]))

    formal = [r.verdict for r in scored if r.item.label == FORMAL]
    if formal:
        rate = lambda attr: _pct(np.mean([bool(getattr(v, attr)) for v in formal]))  # noqa: E731
        parts = [f"düyməli köynək {rate('garment_ok')}"]
        if cfg.jacket_is_formal:
            parts.append(f"pencək {rate('jacket_ok')}")
        if cfg.require_white:
            parts += [f"ağ (model) {rate('model_white')}", f"ağ (piksel) {rate('pixel_white')}"]
        lines.append(f"\n{label(FORMAL)} şəkillərdə siqnalların keçmə faizi (rəsmi səhvən rədd olunanda səbəbi budur):")
        lines.append("  " + " · ".join(parts))

    if calib:
        lines.append("\nƏn yaxşı ayarlar (eyni şəkillərdə seçilib — optimistdir, yeni şəkillərdə yoxlayın):")
        lines.append(f"  {format_params(calib['params'])}")
        lines += _metrics_lines(calib["metrics"])

    if cv:
        lines.append(f"\nÖyrədilmiş təsnifatçı (SigLIP vektorları + ridge, {cv['folds']}-qatlı çarpaz yoxlama, "
                     f"{cv['groups']} şəxs/qrup üzrə):")
        lines += _metrics_lines(cv["metrics"])
        delta = cv["metrics"]["balanced"] - cv["zero_shot"]["balanced"]
        lines.append(f"  Zero-shot ilə müqayisə (balanslı): {delta * 100:+.0f} bənd")
    elif len(scored) < MIN_CV_IMAGES:
        lines.append(f"\nÖyrədilmiş təsnifatçı: ən azı {MIN_CV_IMAGES} qiymətləndirilə bilən şəkil lazımdır.")

    errors = [r for r in scored if r.verdict.result != r.item.label]
    if errors:
        lines.append(f"\nSəhvlər ({len(errors)}):")
        for r in errors[:MAX_ERRORS_SHOWN]:
            lines.append(f"  {label(r.item.label)} → {label(r.verdict.result)}  {r.item.rel}  {r.verdict.reason}")
        if len(errors) > MAX_ERRORS_SHOWN:
            lines.append(f"  … daha {len(errors) - MAX_ERRORS_SHOWN} (hamısı CSV-də)")
    return "\n".join(lines)


# --- CLI --------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.attire", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="şəkil qovluğu (resmi/ və qeyri-resmi/ alt-qovluqları ilə — dəqiqlik üçün)")
    parser.add_argument("--config")
    parser.add_argument("--out", help="CSV yolu (default: reports/geyim_<tarix>.csv)")
    parser.add_argument("--folds", type=int, default=5, help="öyrədilmiş təsnifatçının çarpaz yoxlaması; 0 → söndür")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    from ..privacy import install_runtime_guards

    cfg = load_config(args.config)
    if not cfg.attire.prompts:
        print("config.yaml-da attire.prompts yoxdur — geyim yoxlaması üçün mətn təsvirləri lazımdır.")
        return 2
    root = Path(args.folder)
    if not root.is_dir():
        print(f"Qovluq tapılmadı: {root}")
        return 2
    items, labeled = discover(root)
    if not items:
        print(f"{root} — şəkil yoxdur ({', '.join(sorted(IMAGE_EXT))}).")
        return 1

    print(f"{len(items)} şəkil — modellər yüklənir…", file=sys.stderr)
    try:
        models = load_models(cfg)
    except FileNotFoundError as exc:
        print(exc)
        return 2
    guards = install_runtime_guards(cfg.privacy.block_network)

    def progress(done: int, total: int) -> None:
        print(f"\r  {done * 100 // total:>3}%  ({done}/{total})", end="" if done < total else "\n", file=sys.stderr,
              flush=True)

    try:
        rows = analyze(items, models, cfg.attire, progress)
    finally:
        guards.uninstall()

    calib = calibrate(rows, cfg.attire) if labeled else None
    cv = cross_validate(rows, args.folds) if labeled else None
    print(format_console(rows, labeled, cfg.attire, calib, cv))
    out = Path(args.out).resolve() if args.out else cfg.report.dir / f"geyim_{datetime.now():%Y-%m-%d_%H%M}.csv"
    print(f"\nCSV: {write_csv(rows, out)}")
    return 0
