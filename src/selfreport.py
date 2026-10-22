"""Özünüqiymətləndirmə sorğusunun planlanması.

Hər `interval_minutes` dəqiqədən bir sorğu açılır, `timeout_seconds` ərzində
cavab verilməsə avtomatik bağlanır. Modelin nəticəsi sorğunun GÖSTƏRİLDİYİ anda
götürülür (pəncərənin özü ifadəni dəyişə bilər).
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from .config import DECLINED, TIMEOUT, SelfReportConfig
from .smoothing import SmoothedResult


@dataclass(frozen=True)
class Prompt:
    id: int
    shown_at: datetime  # divar saatı — CSV-yə yazılır
    shown_mono: float
    snapshot: SmoothedResult  # həmin an modelin smoothed nəticəsi


class SelfReportScheduler:
    def __init__(self, cfg: SelfReportConfig, clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self._clock = clock
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._active: Prompt | None = None
        self._last_opened = clock()  # ilk sorğu başlanğıcdan bir interval sonra

    @property
    def interval_s(self) -> float:
        return self.cfg.interval_minutes * 60.0

    @property
    def active(self) -> Prompt | None:
        return self._active

    def valid_answers(self) -> set[str]:
        return set(self.cfg.categories) | {DECLINED}

    def seconds_until_next(self) -> float:
        return max(0.0, self._last_opened + self.interval_s - self._clock())

    def due(self) -> bool:
        return self.cfg.enabled and self._active is None and self.seconds_until_next() <= 0

    def open(self, snapshot: SmoothedResult, now_wall: datetime | None = None) -> Prompt:
        with self._lock:
            now = self._clock()
            prompt = Prompt(next(self._ids), now_wall or datetime.now(), now, snapshot)
            self._active = prompt
            self._last_opened = now
            return prompt

    def answer(self, prompt_id: int, code: str) -> tuple[Prompt, str] | None:
        """Cavabı qəbul edir. Köhnə/naməlum sorğu və ya yanlış kod üçün None."""
        if code not in self.valid_answers():
            raise ValueError(f"Naməlum cavab kodu: {code!r}")
        with self._lock:
            if self._active is None or self._active.id != prompt_id:
                return None
            prompt, self._active = self._active, None
            return prompt, code

    def expire(self) -> tuple[Prompt, str] | None:
        """Vaxtı keçmiş sorğunu bağlayır və (prompt, "timeout") qaytarır."""
        with self._lock:
            p = self._active
            if p is None or self._clock() - p.shown_mono < self.cfg.timeout_seconds:
                return None
            self._active = None
            return p, TIMEOUT

    def cancel(self) -> Prompt | None:
        """Sessiya dayandırılanda açıq sorğunu yazmadan bağlayır."""
        with self._lock:
            p, self._active = self._active, None
            return p

    def postpone(self) -> None:
        """Növbəti sorğunu indidən bir interval sonraya keçirir (məs. nəzarətli testdən sonra)."""
        with self._lock:
            self._last_opened = self._clock()

    def seconds_left(self) -> float | None:
        p = self._active
        if p is None:
            return None
        return max(0.0, self.cfg.timeout_seconds - (self._clock() - p.shown_mono))
