"""Araşdırma dataseti: nəzarətli testin işarələnmiş üz şəkilləri.

Layihədə şəkli diskə yazan YEGANƏ moduldur. Qaydalar:
- yalnız nəzarətli testdə, testerin həmin test üçün ayrıca razılığı ilə;
- yalnız "saxlama" mərhələsinin kadrları (tapşırılan ifadə məlumdur);
- bütün kadr yox, üz ətrafında kəsik (research.crop_margin) — fon minimumdur;
- yalnız research.dir altına; şəbəkəyə heç nə getmir.
Adi monitorinqdə privacy.DiskImageGuard bütün şəkil yazma API-lərini bağlı saxlayır.

Struktur: dataset/<YYYYMMDD_HHMMSS>_<tester>/
    meta.json   — sessiya, tester (anonim ID), razılıq vaxtı, test planı
    frames.csv  — hər şəkil: ifadə, addım, üz qutusu və 5 nöqtə (şəkil koordinatlarında)
    00001_Happiness.jpg …
    .cache/     — python -m src.evaluate-in proqnoz keşi (yalnız rəqəmlər; sessiya ilə birlikdə silinir)

    python -m src.dataset                      # sessiyaların siyahısı
    python -m src.dataset --delete <sessiya>   # bir sessiyanı sil
    python -m src.dataset --delete-all --yes   # hamısını sil
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .config import CalibrationConfig, ResearchConfig, load_config
from .detector import Detection

FRAME_COLUMNS = [
    "file", "index", "timestamp", "tester_id", "step", "round", "expected", "hold_t",
    "face_status", "score", "box_x", "box_y", "box_w", "box_h",
    *[f"lm{i}_{a}" for i in range(5) for a in "xy"],
    "frame_w", "frame_h", "crop_x0", "crop_y0",
]  # fmt: skip


class DatasetWriter:
    def __init__(self, cfg: ResearchConfig, tester_id: str, operator_id: str, test_cfg: CalibrationConfig,
                 consent_at: datetime | None = None) -> None:
        self.cfg = cfg
        self.tester_id = tester_id
        now = datetime.now()
        self.session_id = f"{now:%Y%m%d_%H%M%S}_{tester_id}"
        self.dir = Path(cfg.dir) / self.session_id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.count = 0
        self._lock = threading.Lock()
        self._meta = {
            "session_id": self.session_id,
            "tester_id": tester_id,
            "workstation_operator_id": operator_id,
            "created": now.astimezone().isoformat(timespec="seconds"),
            "consent_at": (consent_at or now).astimezone().isoformat(timespec="seconds"),
            "consent": "tester bu test üçün şəkillərin araşdırma məqsədilə saxlanmasına razılıq verib",
            "crop_margin": cfg.crop_margin,
            "test_plan": {"expressions": test_cfg.expressions, "rounds": test_cfg.rounds,
                          "hold_seconds": test_cfg.hold_seconds},
            "frames": 0,
            "closed": False,
        }
        self._write_meta()
        self._fh = (self.dir / "frames.csv").open("w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._fh, fieldnames=FRAME_COLUMNS)
        self._csv.writeheader()

    def _write_meta(self) -> None:
        (self.dir / "meta.json").write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, frame_bgr: np.ndarray, detection: Detection, expected: str, step: int, round_: int,
            hold_t: float, ts: datetime) -> str | None:
        """Üz tapılıbsa (statusundan asılı olmayaraq) üz ətrafında kəsiyi yazır."""
        face = detection.face
        if face is None or self._fh is None:
            return None
        height, width = frame_bgr.shape[:2]
        m = self.cfg.crop_margin
        x0 = max(0, int(math.floor(face.x - face.w * m)))
        y0 = max(0, int(math.floor(face.y - face.h * m)))
        x1 = min(width, int(math.ceil(face.x + face.w * (1 + m))))
        y1 = min(height, int(math.ceil(face.y + face.h * (1 + m))))
        crop = frame_bgr[y0:y1, x0:x1]
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg.jpeg_quality)])
        if not ok:
            return None
        with self._lock:
            self.count += 1
            name = f"{self.count:05d}_{expected}.jpg"
            (self.dir / name).write_bytes(buf.tobytes())
            lm = face.landmarks - np.array([x0, y0], dtype=np.float32)
            row = {
                "file": name, "index": self.count, "timestamp": ts.astimezone().isoformat(timespec="milliseconds"),
                "tester_id": self.tester_id, "step": step, "round": round_, "expected": expected,
                "hold_t": f"{hold_t:.2f}", "face_status": detection.status.value, "score": f"{face.score:.4f}",
                "box_x": f"{face.x - x0:.1f}", "box_y": f"{face.y - y0:.1f}",
                "box_w": f"{face.w:.1f}", "box_h": f"{face.h:.1f}",
                "frame_w": width, "frame_h": height, "crop_x0": x0, "crop_y0": y0,
            }
            for i in range(5):
                row[f"lm{i}_x"], row[f"lm{i}_y"] = f"{lm[i, 0]:.1f}", f"{lm[i, 1]:.1f}"
            self._csv.writerow(row)
            self._fh.flush()
        return name

    def close(self) -> None:
        with self._lock:
            if self._fh is None:
                return
            self._fh.close()
            self._fh = None
            self._meta.update(frames=self.count, closed=True)
            self._write_meta()


# --- oxuma --------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    session_id: str
    tester_id: str
    path: Path
    expected: str
    step: int
    round: int
    face_status: str
    box: tuple[float, float, float, float]
    landmarks: np.ndarray  # (5, 2), şəkil koordinatlarında

    def image(self) -> np.ndarray:
        img = cv2.imread(str(self.path))
        if img is None:
            raise FileNotFoundError(self.path)
        return img


def sessions(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.glob("*") if (p / "frames.csv").is_file()) if root.is_dir() else []


def load_samples(root: Path, testers: set[str] | None = None, only_ok: bool = True) -> list[Sample]:
    out = []
    for sdir in sessions(root):
        meta = json.loads((sdir / "meta.json").read_text(encoding="utf-8"))
        tester = meta.get("tester_id", sdir.name)
        if testers and tester not in testers:
            continue
        with (sdir / "frames.csv").open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if only_ok and r["face_status"] != "ok":
                    continue
                lm = np.array([[float(r[f"lm{i}_x"]), float(r[f"lm{i}_y"])] for i in range(5)], np.float32)
                out.append(Sample(
                    session_id=sdir.name, tester_id=tester, path=sdir / r["file"], expected=r["expected"],
                    step=int(r["step"]), round=int(r["round"]), face_status=r["face_status"],
                    box=(float(r["box_x"]), float(r["box_y"]), float(r["box_w"]), float(r["box_h"])),
                    landmarks=lm,
                ))
    return out


def iter_sessions_info(root: Path) -> Iterator[dict]:
    for sdir in sessions(root):
        meta = json.loads((sdir / "meta.json").read_text(encoding="utf-8"))
        size = sum(f.stat().st_size for f in sdir.rglob("*") if f.is_file())
        yield {"session": sdir.name, "tester": meta.get("tester_id"), "frames": meta.get("frames"),
               "created": meta.get("created"), "closed": meta.get("closed"), "mb": size / 2**20}


# --- proqnoz keşi (python -m src.evaluate) ---------------------------------------

CACHE_DIR = ".cache"


def load_cache(session_dir: Path, name: str) -> dict[str, np.ndarray] | None:
    path = Path(session_dir) / CACHE_DIR / f"{name}.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (OSError, ValueError):
        return None


def save_cache(session_dir: Path, name: str, arrays: dict[str, np.ndarray]) -> None:
    """Yalnız rəqəmlər (ehtimallar, ms, fayl adları) — şəkil yox."""
    d = Path(session_dir) / CACHE_DIR
    d.mkdir(exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    (d / f"{name}.npz").write_bytes(buf.getvalue())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.dataset", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--delete", metavar="SESSIYA", help="bir sessiyanı sil")
    parser.add_argument("--delete-all", action="store_true", help="bütün sessiyaları sil")
    parser.add_argument("--yes", action="store_true", help="təsdiq soruşma")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    root = load_config(args.config).research.dir

    if args.delete or args.delete_all:
        targets = sessions(root) if args.delete_all else [root / args.delete]
        targets = [t for t in targets if (t / "frames.csv").is_file() and root in t.resolve().parents]
        if not targets:
            print("Silinəcək sessiya tapılmadı.")
            return 1
        print("Silinəcək:", ", ".join(t.name for t in targets))
        if not args.yes and input("Əminsiniz? [b/X]: ").strip().lower() not in ("b", "bəli", "y", "yes"):
            print("Ləğv edildi.")
            return 1
        for t in targets:
            shutil.rmtree(t)
        print(f"{len(targets)} sessiya silindi.")
        return 0

    infos = list(iter_sessions_info(root))
    if not infos:
        print(f"{root} — sessiya yoxdur.")
        return 0
    print(f"{'Sessiya':<34} {'Tester':<10} {'Şəkil':>6} {'MB':>6}")
    for i in infos:
        print(f"{i['session']:<34} {str(i['tester']):<10} {str(i['frames']):>6} {i['mb']:6.1f}")
    print(f"\nCəmi: {len(infos)} sessiya, {sum(i['frames'] or 0 for i in infos)} şəkil, "
          f"{sum(i['mb'] for i in infos):.1f} MB  ({root})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
