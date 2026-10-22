"""Giriş nöqtəsi.

    python -m src.app                          # UI + webcam (razılıq ekranı ilə)
    python -m src.app --video fayl.mp4         # UI, kamera əvəzinə video (real vaxt)
    python -m src.app --headless --video f.mp4 # UI-sız, video sürətli emal → CSV
    python -m src.app --headless               # UI-sız webcam (terminalda razılıq)
    python -m src.app --image a.jpg b.jpg      # tək şəkillər üzrə yoxlama (CSV yazılmır)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

from .config import SUPPORTED_MODELS, AppConfig, load_config
from .labels import EMOTION_LABELS, FACE_REASON_LABELS, STATUS_LABELS
from .smoothing import SmoothedResult

log = logging.getLogger("src")


def model_keys(cfg: AppConfig) -> list[str]:
    """Əsas model birinci, sonra (müqayisə aktivdirsə) compare.models."""
    if cfg.compare.mode == "off":
        return [cfg.model.name]
    return list(dict.fromkeys([cfg.model.name, *cfg.compare.models]))


def budget_limit_ms(cfg: AppConfig) -> float | None:
    """auto rejimində bir analizdə bütün modellər üçün ayrılan vaxt (ms)."""
    if cfg.compare.mode != "auto":
        return None
    return cfg.compare.budget_ratio * 1000.0 / cfg.capture.analysis_fps


def build_analyzer(cfg: AppConfig, keys: list[str] | None = None):
    """Detektor və modelləri yükləyir. Şəbəkə qoruyucusu BUNDAN SONRA qurulmalıdır.

    Əsas model yüklənməsə, xəta verilir; əlavə modellər yüklənməsə, səbəbi
    analyzer.missing-də saxlanılır və iş davam edir.
    """
    from .detector import FaceDetector
    from .model_zoo import is_installed, make_recognizer
    from .pipeline import Analyzer
    from .smoothing import EmotionSmoother

    keys = keys or model_keys(cfg)
    primary = make_recognizer(cfg.model.name, cfg.model.engine, cfg.model.device)
    analyzer = Analyzer(FaceDetector(cfg.detector), primary, EmotionSmoother(cfg.smoothing))
    analyzer.flip_tta = cfg.model.flip_tta
    for key in keys:
        if key == cfg.model.name:
            continue
        if not is_installed(key):
            analyzer.missing[key] = "quraşdırılmayıb — scripts/download_models.py --extra"
            continue
        try:
            analyzer.add_model(make_recognizer(key), EmotionSmoother(cfg.smoothing))
        except Exception as exc:  # noqa: BLE001 - əlavə model işi dayandırmamalıdır
            log.warning("%s yüklənmədi: %s", key, exc)
            analyzer.missing[key] = str(exc)
    return analyzer


def build_logger(cfg: AppConfig, analyzer):
    from .csv_logger import CsvLogger

    r = analyzer.recognizer
    return CsvLogger(cfg.logging.dir, cfg.operator_id, r.model_name, r.engine, r.device)


def describe(smoothed: SmoothedResult) -> str:
    if smoothed.has_estimate:
        name = EMOTION_LABELS.get(smoothed.top_emotion, smoothed.top_emotion)
        text = f"{name} {smoothed.top_prob:.0%}"
        if smoothed.status.value == "uncertain":
            text = f"Qeyri-müəyyən (ən yüksək: {text})"
        if smoothed.valence is not None:
            text += f"  V={smoothed.valence:+.2f} A={smoothed.arousal:+.2f}"
        return f"{text}  [{smoothed.n_frames} kadr]"
    return STATUS_LABELS[smoothed.status.value]


def run_images(cfg: AppConfig, paths: list[str]) -> int:
    from .capture import load_image
    from .config import EMOTION_CLASSES
    from .privacy import install_runtime_guards

    analyzer = build_analyzer(cfg)
    guards = install_runtime_guards(cfg.privacy.block_network)
    try:
        print("Şəkil rejimi — tək kadr, smoothing tətbiq olunmur, yalnız yoxlama üçündür.\n")
        for p in paths:
            img = load_image(p)
            det = analyzer.detector.detect(img)
            print(f"{p}")
            print(f"  üz: {FACE_REASON_LABELS[det.status.value]} (tapılan üz sayı: {det.num_faces}"
                  + (f", confidence {det.face.score:.2f})" if det.face else ")"))
            if not det.ok:
                print("  model işə salınmadı — üz tam görünmür\n")
                continue
            crop = analyzer.detector.crop(img, det.face)
            for runner in analyzer.active():
                pred = runner.recognizer.predict(crop)
                order = sorted(range(len(EMOTION_CLASSES)), key=lambda i: -pred.probs[i])
                top3 = "  ".join(f"{EMOTION_LABELS[EMOTION_CLASSES[i]]} {pred.probs[i]:.0%}" for i in order[:3])
                va = f"  V={pred.valence:+.2f} A={pred.arousal:+.2f}" if pred.valence is not None else ""
                print(f"  {runner.key:<22} {top3}{va}  ({pred.latency_ms:.1f} ms)")
            for key, reason in analyzer.missing.items():
                print(f"  {key:<22} — {reason}")
            print()
    finally:
        guards.uninstall()
    return 0


def run_headless_video(cfg: AppConfig, video: str) -> int:
    from .pipeline import run_video
    from .privacy import install_runtime_guards

    analyzer = build_analyzer(cfg)
    logger = build_logger(cfg, analyzer)
    guards = install_runtime_guards(cfg.privacy.block_network)
    try:
        def show(t, result):
            print(f"\r  {t:7.1f}s  {describe(result.smoothed):<60}", end="", flush=True)

        summary = run_video(video, analyzer, logger, cfg.capture.analysis_fps, on_result=show)
    finally:
        logger.close()
        guards.uninstall()
    print(f"\n\nEmal olunan kadr: {summary.frames}  (orta gecikmə {summary.mean_latency_ms:.1f} ms)")
    for status, n in sorted(summary.status_counts.items()):
        print(f"  {STATUS_LABELS.get(status, status):<16} {n}")
    if summary.csv_path:
        print(f"CSV: {summary.csv_path}")
    return 0


def run_headless_camera(cfg: AppConfig) -> int:
    from .capture import CameraSource
    from .pipeline import LivePipeline
    from .privacy import grant_consent, install_runtime_guards

    print(
        "Kamera açılacaq. Ölçülən: üz ifadəsinin ehtimalları (daxili hiss deyil).\n"
        "Saxlanılan: yalnız rəqəmlər (CSV). Video və şəkil yazılmır, şəbəkəyə heç nə getmir.\n"
        "Dayandırmaq üçün istənilən an Ctrl+C."
    )
    if input("Razısınız? [b/X] (bəli/xeyr): ").strip().lower() not in ("b", "bəli", "y", "yes"):
        print("Razılıq verilmədi — kamera açılmadı.")
        return 1
    consent = grant_consent("cli")

    analyzer = build_analyzer(cfg)
    logger = build_logger(cfg, analyzer)
    guards = install_runtime_guards(cfg.privacy.block_network)
    source = CameraSource(cfg.capture, consent)

    def show(result, cpu):
        reason = FACE_REASON_LABELS[result.detection.status.value]
        print(f"\r  ● {describe(result.smoothed):<52} | {reason:<26} | {result.latency_ms:5.1f} ms", end="", flush=True)

    pipeline = LivePipeline(source, analyzer, logger, cfg.capture.analysis_fps, on_result=show)
    pipeline.start()
    try:
        while pipeline.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        source.close()
        logger.close()
        guards.uninstall()
    print("\nDayandırıldı. Kamera bağlandı.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.app", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="config.yaml yolu")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, help="config-dəki əsas modeli əvəz edir")
    parser.add_argument("--compare", choices=["off", "auto", "on"], help="config-dəki compare.mode-u əvəz edir")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", help="kamera əvəzinə video fayl")
    source.add_argument("--image", nargs="+", help="bir və ya bir neçə şəkil")
    parser.add_argument("--headless", action="store_true", help="UI olmadan işlə")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    if args.model:
        from .model_zoo import spec

        engine = ("onnx", "cpu") if spec(args.model).family == "fer" else (cfg.model.engine, cfg.model.device)
        cfg.model = replace(cfg.model, name=args.model, engine=engine[0], device=engine[1])
    if args.compare:
        cfg.compare = replace(cfg.compare, mode=args.compare)

    if args.image:
        return run_images(cfg, args.image)
    if args.video and not Path(args.video).is_file():
        parser.error(f"video tapılmadı: {args.video}")
    if args.headless:
        return run_headless_video(cfg, args.video) if args.video else run_headless_camera(cfg)

    from .ui.window import run_ui

    return run_ui(cfg, video=args.video)


if __name__ == "__main__":
    raise SystemExit(main())
