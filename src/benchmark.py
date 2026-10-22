"""Modellərin sürət və yaddaş ölçüsü.

    python -m src.benchmark --model all                       # bütün modellər, test şəkli, CPU (+ GPU varsa)
    python -m src.benchmark --model all --camera              # webcam kadrları (razılıq ilə)
    python -m src.benchmark --model enet_b2_8 --video f.mp4 --frames 100 --device cpu
    python -m src.benchmark --model enet_b0_8_best_afew,deepface_emotion,vit_fer2013 --together

Ölçülür: detektor ms, model ms, ümumi emal FPS-i, RAM (modelin əlavə etdiyi və pik).
Hər konfiqurasiya ayrıca prosesdə işləyir ki, RAM ölçüsü bir-birinə qarışmasın.
--together seçilmiş modelləri müqayisə rejimindəki kimi EYNİ prosesdə, eyni kadrda
ardıcıl işlədir — kompüterin hamısını birlikdə çəkib-çəkmədiyini göstərir.
Kadrlar yalnız yaddaşda saxlanılır və proseslər arasında boru (pipe) ilə ötürülür.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    DEFAULT_BENCHMARK_MODELS,
    FER_KEYS,
    PROJECT_ROOT,
    SUPPORTED_MODELS,
    AppConfig,
    DetectorConfig,
    load_config,
)

ALL_MODELS = (*DEFAULT_BENCHMARK_MODELS, *FER_KEYS)
TOGETHER = "+"  # Run.model-də birlikdə ölçülən modelləri ayırır

DEFAULT_IMAGE = PROJECT_ROOT / "tests" / "fixtures" / "face.png"
MAX_DISTINCT_FRAMES = 30
WORKER_TIMEOUT_S = 600


@dataclass
class Run:
    model: str
    engine: str
    device: str
    skipped: str | None = None
    load_s: float | None = None
    detector_ms: dict[str, float] = field(default_factory=dict)
    model_ms: dict[str, float] = field(default_factory=dict)
    total_ms: dict[str, float] = field(default_factory=dict)
    fps: float | None = None
    ram_model_mb: float | None = None
    ram_peak_mb: float | None = None
    frames: int = 0
    faces: int = 0
    error: str | None = None


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    a = np.asarray(values)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95))}


def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 2**20


def _peak_mb() -> float:
    import psutil

    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows
    if peak is None:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB
    return peak / 2**20


def _worker(run: Run, det_cfg: DetectorConfig, frames: list[np.ndarray], warmup: int, n: int,
            block_network: bool, queue) -> None:
    """Ayrıca prosesdə: modeli yükləyir, ölçür, nəticəni növbəyə qoyur."""
    try:
        from .detector import FaceDetector
        from .model_zoo import make_recognizer
        from .privacy import install_runtime_guards

        base = _rss_mb()
        t0 = time.perf_counter()
        detector = FaceDetector(det_cfg)
        # "a+b+c" → bir prosesdə bir neçə model (müqayisə rejimi kimi)
        recognizers = [make_recognizer(key, run.engine, run.device) for key in run.model.split(TOGETHER)]
        run.load_s = time.perf_counter() - t0
        run.device = recognizers[0].device
        run.ram_model_mb = _rss_mb() - base
        install_runtime_guards(block_network)

        det_ms, mdl_ms, tot_ms = [], [], []
        for i in range(warmup + n):
            frame = frames[i % len(frames)]
            t_start = time.perf_counter()
            det = detector.detect(frame)
            t_det = time.perf_counter()
            if det.face is not None:
                crop = detector.crop(frame, det.face)
                for recognizer in recognizers:
                    recognizer.predict(crop)
            t_end = time.perf_counter()
            if i < warmup:
                continue
            det_ms.append((t_det - t_start) * 1000)
            tot_ms.append((t_end - t_start) * 1000)
            if det.face is not None:
                mdl_ms.append((t_end - t_det) * 1000)
                run.faces += 1
        run.frames = n
        run.detector_ms, run.model_ms, run.total_ms = _stats(det_ms), _stats(mdl_ms), _stats(tot_ms)
        run.fps = 1000.0 / run.total_ms["mean"] if run.total_ms else None
        run.ram_peak_mb = _peak_mb()
    except Exception as exc:  # noqa: BLE001 - nəticədə göstərilir
        run.error = f"{type(exc).__name__}: {exc}"
    queue.put(asdict(run))


def plan_runs(models: list[str], device: str, together: bool = False) -> list[Run]:
    """CPU → ONNX engine; GPU → torch + CUDA (yalnız əlçatandırsa, yalnız EmotiEffLib)."""
    from .model_zoo import is_installed

    cuda_reason = None
    if device in ("cuda", "all"):
        try:
            import torch

            if not torch.cuda.is_available():
                cuda_reason = "CUDA əlçatan deyil (torch.cuda.is_available() = False)"
        except ImportError:
            cuda_reason = "torch qurulmayıb (requirements-gpu.txt)"
    missing = {m: "quraşdırılmayıb — scripts/download_models.py --extra" for m in models if not is_installed(m)}
    runs = []
    for m in models:
        if device in ("cpu", "all"):
            runs.append(Run(m, "onnx", "cpu", skipped=missing.get(m)))
        if device in ("cuda", "all"):
            reason = "FER modeli yalnız ONNX/CPU ilə işləyir" if m in FER_KEYS else cuda_reason
            runs.append(Run(m, "torch", "cuda", skipped=missing.get(m) or reason))
    if together:
        ok = [m for m in models if m not in missing]
        if len(ok) > 1:
            runs.append(Run(TOGETHER.join(ok), "onnx", "cpu"))
    return runs


def run_isolated(run: Run, det_cfg: DetectorConfig, frames, warmup: int, n: int, block_network: bool) -> Run:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(run, det_cfg, frames, warmup, n, block_network, queue), daemon=True)
    proc.start()
    try:
        result = queue.get(timeout=WORKER_TIMEOUT_S)
    except Exception:  # noqa: BLE001
        proc.kill()
        return replace(run, error="vaxt bitdi")
    proc.join(timeout=10)
    return Run(**result)


# --- kadr mənbələri -------------------------------------------------------------


def frames_from_image(path: Path) -> list[np.ndarray]:
    from .capture import load_image

    return [load_image(path)]


def frames_from_video(path: Path, limit: int = MAX_DISTINCT_FRAMES) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılmadı: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or limit
    wanted = set(np.linspace(0, max(0, total - 1), num=min(limit, total), dtype=int).tolist())
    frames, i = [], 0
    while cap.grab():
        if i in wanted:
            ok, frame = cap.retrieve()
            if ok:
                frames.append(frame)
        i += 1
    cap.release()
    if not frames:
        raise RuntimeError("Videodan kadr oxunmadı")
    return frames


def frames_from_camera(cfg: AppConfig, count: int = MAX_DISTINCT_FRAMES) -> list[np.ndarray]:
    from .capture import CameraSource
    from .privacy import grant_consent

    print(
        f"Benchmark üçün kameradan {count} kadr götürüləcək (yalnız yaddaşda, diskə yazılmır).\n"
        "Kadrlar ölçü bitən kimi silinir."
    )
    if input("Razısınız? [b/X]: ").strip().lower() not in ("b", "bəli", "y", "yes"):
        raise SystemExit("Razılıq verilmədi — kamera açılmadı.")
    source = CameraSource(cfg.capture, grant_consent("cli"))
    frames: list[np.ndarray] = []
    try:
        deadline = time.monotonic() + 30
        while len(frames) < count and time.monotonic() < deadline:
            frame = source.latest()
            if frame is not None:
                frames.append(frame)
            time.sleep(0.1)
    finally:
        source.close()
    if not frames:
        raise RuntimeError("Kameradan kadr alınmadı")
    return frames


# --- çıxış ----------------------------------------------------------------------


def system_info() -> dict[str, Any]:
    import cv2
    import onnxruntime
    import psutil

    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_cores_logical": psutil.cpu_count(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "python": platform.python_version(),
        "onnxruntime": onnxruntime.__version__,
        "opencv": cv2.__version__,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["torch"] = None
    return info


def _f(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def format_table(runs: list[Run]) -> str:
    head = f"{'Model':<22} {'Mühərrik':<11} {'Yükl. s':>7} {'Detektor ms':>14} {'Model ms':>14} {'Cəmi ms':>8} {'FPS':>7} {'RAM+ MB':>8} {'Pik MB':>7}"
    sub = f"{'':<22} {'':<11} {'':>7} {'orta / p95':>14} {'orta / p95':>14} {'orta':>8} {'':>7} {'':>8} {'':>7}"
    lines = [head, sub, "─" * len(head)]
    for r in runs:
        eng = f"{r.engine}·{r.device}"
        name = r.model
        if TOGETHER in name:
            lines.append("─" * len(head))
            lines.append(f"Birlikdə ({len(name.split(TOGETHER))} model): {name.replace(TOGETHER, ' + ')}")
            name = "  hamısı bir kadrda"
        if r.skipped or r.error:
            lines.append(f"{name:<22} {eng:<11} keçildi: {r.skipped or r.error}")
            continue
        d, m, t = r.detector_ms, r.model_ms, r.total_ms
        det = f"{_f(d.get('mean'))} / {_f(d.get('p95'))}"
        mdl = f"{_f(m.get('mean'))} / {_f(m.get('p95'))}" if m else "üz yoxdur"
        lines.append(
            f"{name:<22} {eng:<11} {_f(r.load_s, 2):>7} {det:>14} {mdl:>14} {_f(t.get('mean')):>8} "
            f"{_f(r.fps):>7} {_f(r.ram_model_mb, 0):>8} {_f(r.ram_peak_mb, 0):>7}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.benchmark", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="config.yaml yolu")
    parser.add_argument("--model", default="all",
                        help=f"'all' ({', '.join(ALL_MODELS)}) və ya vergüllə siyahı")
    parser.add_argument("--together", action="store_true",
                        help="seçilmiş modelləri həm də birlikdə (müqayisə rejimi kimi) ölç")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--image", help=f"şəkil (default: {DEFAULT_IMAGE.relative_to(PROJECT_ROOT)})")
    src.add_argument("--video", help="video fayl")
    src.add_argument("--camera", action="store_true", help="webcam kadrları (razılıq soruşulur)")
    parser.add_argument("--device", choices=["cpu", "cuda", "all"], default="all",
                        help="cpu → onnx; cuda → torch+CUDA; all → hər ikisi (CUDA varsa)")
    parser.add_argument("--frames", type=int, help="ölçülən kadr sayı (default: config benchmark.frames)")
    parser.add_argument("--warmup", type=int, help="ölçüyə daxil olmayan isinmə kadrları")
    parser.add_argument("--out", help="JSON nəticə faylı (default: reports/benchmark_<vaxt>.json)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config(args.config)
    models = list(ALL_MODELS) if args.model == "all" else [m.strip() for m in args.model.split(",")]
    unknown = [m for m in models if m not in SUPPORTED_MODELS]
    if unknown:
        parser.error(f"naməlum model(lər): {unknown}. Seçimlər: {', '.join(SUPPORTED_MODELS)}")
    n = args.frames or cfg.benchmark.frames
    warmup = cfg.benchmark.warmup if args.warmup is None else args.warmup

    if args.camera:
        frames, source = frames_from_camera(cfg), "camera"
    elif args.video:
        frames, source = frames_from_video(Path(args.video)), f"video:{Path(args.video).name}"
    else:
        image = Path(args.image) if args.image else DEFAULT_IMAGE
        if not image.is_file():
            parser.error(f"şəkil tapılmadı: {image} (scripts/download_models.py --test-fixture)")
        frames, source = frames_from_image(image), f"image:{image.name}"
    h, w = frames[0].shape[:2]
    print(f"Mənbə: {source} · {len(frames)} fərqli kadr · {w}×{h} · {n} ölçü + {warmup} isinmə kadrı\n")

    runs = []
    for run in plan_runs(models, args.device, together=args.together):
        if run.skipped:
            runs.append(run)
            continue
        print(f"  … {run.model} ({run.engine}·{run.device})", flush=True)
        runs.append(run_isolated(run, cfg.detector, frames, warmup, n, cfg.privacy.block_network))
    frames.clear()  # kadrlar yaddaşdan silinir

    print()
    print(format_table(runs))
    out = Path(args.out) if args.out else cfg.report.dir / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "frame_size": [w, h],
        "frames": n,
        "warmup": warmup,
        "system": system_info(),
        "runs": [asdict(r) for r in runs],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nNəticələr: {out}")
    return 0 if all(r.error is None for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
