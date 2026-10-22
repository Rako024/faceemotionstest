"""Adi monitorinqdə video və şəkil diskə yazılmır — statik və iş vaxtı yoxlamaları.

Yeganə istisna: src/dataset.py — nəzarətli testdə, testerin ayrıca razılığı ilə, yalnız
research.dir altına (bax tests/test_dataset.py).
"""

import ast
import os
import tempfile
import time
from pathlib import Path

import cv2
import pytest

from src.app import build_analyzer, build_logger
from src.capture import StaticSource
from src.config import PROJECT_ROOT
from src.pipeline import LivePipeline, run_video
from src.privacy import install_runtime_guards

from .conftest import face_frame

SRC = PROJECT_ROOT / "src"
MEDIA_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif",
             ".mp4", ".avi", ".mkv", ".mov", ".webm", ".wmv", ".npy", ".npz", ".pkl"}  # fmt: skip
FORBIDDEN_ATTRS = {"imwrite", "imwritemulti", "imwriteanimation", "VideoWriter", "tofile", "save", "savez", "dump",
                   "write_bytes"}  # fmt: skip
# privacy.py — qoruyucunun özü (adları yalnız bağlamaq üçün çəkir);
# dataset.py — razılıqla nəzarətli test datasının yeganə yazıcısı.
ALLOWED_FILES = {"privacy.py", "dataset.py"}


def _src_files():
    return [p for p in SRC.rglob("*.py") if p.name not in ALLOWED_FILES]


@pytest.mark.parametrize("path", _src_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_source_has_no_media_write_calls(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            pytest.fail(f"{path.name}:{node.lineno} — qadağan çağırış .{node.attr}")
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "open":
            modes = [a.value for a in node.args[1:2] if isinstance(a, ast.Constant)]
            modes += [k.value.value for k in node.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)]
            assert not any("b" in m for m in modes), f"{path.name}:{node.lineno} — binar fayl yazılışı"


def _snapshot(*roots: Path) -> dict[Path, float]:
    files = {}
    skip = {".venv", ".git", "__pycache__", ".pytest_cache"}
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip]
            for name in filenames:
                p = Path(dirpath) / name
                try:
                    files[p] = p.stat().st_mtime
                except OSError:
                    pass
    return files


def _new_or_changed(before, after):
    return {p for p, m in after.items() if p not in before or before[p] != m}


@pytest.fixture
def spy_guards():
    """Real qoruyucular + hər hansı yazma cəhdinin sayılması."""
    guards = install_runtime_guards(block_network=True)
    yield guards
    guards.uninstall()


@pytest.mark.usefixtures("models_ready")
def test_video_pipeline_writes_only_csv(tmp_config, test_video, tmp_path, monkeypatch, spy_guards):
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    roots = [tmp_path, PROJECT_ROOT]
    before = _snapshot(*roots)
    start = time.time()

    analyzer = build_analyzer(tmp_config)
    logger = build_logger(tmp_config, analyzer)
    summary = run_video(test_video, analyzer, logger, analysis_fps=2.0)
    logger.close()

    changed = _new_or_changed(before, _snapshot(*roots))
    assert summary.frames > 0
    assert not spy_guards.disk.blocked_attempts, "kod kadrı diskə yazmağa cəhd etdi"
    assert not spy_guards.network.blocked_attempts, "kod şəbəkəyə çıxmağa cəhd etdi"
    media = {p for p in changed if p.suffix.lower() in MEDIA_EXT}
    # Sistem temp qovluğu böyük ola bilər — yalnız yuxarı səviyyədəki yeni media faylları
    with os.scandir(tempfile.gettempdir()) as it:
        media |= {
            Path(e.path) for e in it
            if e.is_file() and Path(e.name).suffix.lower() in MEDIA_EXT and e.stat().st_mtime >= start - 1
        }
    assert not media, f"diskə media yazıldı: {media}"
    in_tmp = {p for p in changed if tmp_path in p.parents}
    assert in_tmp and all(p.suffix == ".csv" for p in in_tmp), in_tmp
    assert list(work.iterdir()) == []


@pytest.mark.usefixtures("models_ready")
def test_live_pipeline_with_preview_writes_only_csv(tmp_config, face_bgr, tmp_path, spy_guards):
    roots = [tmp_path, PROJECT_ROOT]
    before = _snapshot(*roots)

    analyzer = build_analyzer(tmp_config)
    logger = build_logger(tmp_config, analyzer)
    pipe = LivePipeline(StaticSource([face_frame(face_bgr)]), analyzer, logger, 2.0)
    pipe.start()
    previews = []
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        jpg = pipe.preview_jpeg()
        if jpg:
            previews.append(len(jpg))
        time.sleep(0.1)
    pipe.stop()
    logger.close()

    assert previews, "önizləmə yaddaşda yaradılmalıdır"
    changed = _new_or_changed(before, _snapshot(*roots))
    assert changed and all(p.suffix == ".csv" for p in changed), changed
    assert not spy_guards.disk.blocked_attempts


def test_guard_catches_accidental_write(tmp_path, spy_guards):
    """Gələcəkdə kimsə imwrite əlavə etsə, iş vaxtı da tutulur."""
    import numpy as np

    with pytest.raises(PermissionError):
        cv2.imwrite(str(tmp_path / "leak.jpg"), np.zeros((4, 4, 3), np.uint8))
    assert spy_guards.disk.blocked_attempts == ["cv2.imwrite"]
    assert not (tmp_path / "leak.jpg").exists()
