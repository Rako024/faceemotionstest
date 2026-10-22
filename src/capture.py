"""Kadr mənbələri: webcam, video fayl, şəkil.

Canlı mənbələr arxa thread-də kadrları oxuyur və yalnız ƏN SON kadrı yaddaşda
saxlayır — bufer yığılmır, köhnə kadr emal olunmur, heç nə diskə yazılmır.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

from .config import CaptureConfig
from .privacy import ConsentToken, require_consent

log = logging.getLogger(__name__)


class FrameSource(Protocol):
    def latest(self) -> np.ndarray | None: ...

    @property
    def finished(self) -> bool: ...

    def close(self) -> None: ...


class _ThreadedSource:
    """Arxa thread-də oxuyub son kadrı saxlayan ümumi baza."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._thread = threading.Thread(target=self._run, name=type(self).__name__, daemon=True)

    def _start(self) -> None:
        self._thread.start()

    def _publish(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    def _run(self) -> None:  # pragma: no cover - alt siniflər təyin edir
        raise NotImplementedError

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)
        with self._lock:
            self._frame = None


class CameraSource(_ThreadedSource):
    """Webcam. Yalnız etibarlı ConsentToken ilə açılır.

    index — cameras.list_cameras() sırası (DirectShow); verilməyibsə capture.camera_index.
    """

    def __init__(self, cfg: CaptureConfig, consent: ConsentToken, index: int | None = None,
                 name: str | None = None) -> None:
        require_consent(consent)
        super().__init__()
        index = cfg.camera_index if index is None else index
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(index, backend)
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f"Kamera açılmadı: {name or 'index ' + str(index)} — başqa proqram istifadə edə bilər")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._start()

    def _run(self) -> None:
        failures = 0
        try:
            while not self._stop.is_set():
                ok, frame = self._cap.read()
                if not ok:
                    failures += 1
                    if failures > 50:
                        log.error("Kameradan kadr oxunmur — dayandırılır")
                        break
                    time.sleep(0.05)
                    continue
                failures = 0
                self._publish(frame)
        finally:
            self._cap.release()
            self._finished.set()


class SwitchableSource:
    """Canlı mənbəni pipeline-ı dayandırmadan dəyişməyə imkan verir (kamera seçimi).

    Yeni mənbə ÇAĞIRAN tərəfindən artıq açılmış verilir — açılmasa köhnə kamera işləməyə davam edir.
    """

    def __init__(self, inner: FrameSource) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self._closed = False

    def latest(self) -> np.ndarray | None:
        with self._lock:
            inner = self._inner
        return inner.latest()

    @property
    def finished(self) -> bool:
        return self._inner.finished

    def switch(self, new: FrameSource) -> bool:
        """False — mənbə artıq bağlanıb; yeni kamera da dərhal bağlanır (indikatorsuz açıq qalmasın)."""
        with self._lock:
            if self._closed:
                old = new
            else:
                old, self._inner = self._inner, new
        old.close()
        return old is not new

    def close(self) -> None:
        with self._lock:
            self._closed = True
            inner = self._inner
        inner.close()


class PacedVideoSource(_ThreadedSource):
    """Video faylını real vaxt sürətində oynadır — kamera əvəzinə (UI/demo rejimi)."""

    def __init__(self, path: str | Path, loop: bool = False) -> None:
        super().__init__()
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Video açılmadı: {self.path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_interval = 1.0 / fps if fps and fps > 0 else 1.0 / 25
        self._loop = loop
        self._start()

    def _run(self) -> None:
        try:
            next_t = time.monotonic()
            while not self._stop.is_set():
                ok, frame = self._cap.read()
                if not ok:
                    if self._loop:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                self._publish(frame)
                next_t += self._frame_interval
                delay = next_t - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
        finally:
            self._cap.release()
            self._finished.set()


class StaticSource:
    """Sabit kadr(lar) — testlər və şəkil rejimi üçün."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._i = 0

    def latest(self) -> np.ndarray | None:
        if not self._frames:
            return None
        frame = self._frames[self._i % len(self._frames)]
        self._i += 1
        return frame.copy()

    @property
    def finished(self) -> bool:
        return False

    def close(self) -> None:
        self._frames = []


def iter_video_samples(path: str | Path, analysis_fps: float) -> Iterator[tuple[float, np.ndarray]]:
    """Videonu sürətli emal üçün oxuyur: hər 1/analysis_fps saniyədən bir kadr (video vaxtı ilə).

    Ara kadrlar grab() ilə dekod edilmədən ötürülür.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılmadı: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = 1.0 / analysis_fps
        next_t = 0.0
        index = 0
        while cap.grab():
            t = index / fps
            index += 1
            if t + 1e-9 < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            yield t, frame
            next_t += step
    finally:
        cap.release()


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Şəkil oxunmadı: {path}")
    return img
