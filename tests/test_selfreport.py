from datetime import datetime

import pytest

from src.config import SelfReportConfig
from src.smoothing import SmoothedResult, SmoothStatus
from src.selfreport import SelfReportScheduler

SNAP = SmoothedResult(SmoothStatus.NO_FACE, None, None, None, None, None, 0)


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def sched(clock):
    cfg = SelfReportConfig(
        interval_minutes=10, timeout_seconds=30, categories={"neutral": "Neytral", "tired": "Yorğun"}
    )
    return SelfReportScheduler(cfg, clock)


def test_not_due_at_start(sched):
    assert not sched.due()
    assert sched.seconds_until_next() == pytest.approx(600)


def test_due_after_interval(sched, clock):
    clock.t = 599
    assert not sched.due()
    clock.t = 600
    assert sched.due()


def test_postpone_restarts_interval(sched, clock):
    clock.t = 700
    assert sched.due()
    sched.postpone()  # nəzarətli testdən panelə qayıdış
    assert not sched.due() and sched.seconds_until_next() == 600
    clock.t = 1300
    assert sched.due()


def test_not_due_while_prompt_open(sched, clock):
    clock.t = 600
    sched.open(SNAP)
    clock.t = 1300
    assert not sched.due()


def test_answer_closes_prompt(sched, clock):
    clock.t = 600
    p = sched.open(SNAP, datetime(2026, 10, 22, 10, 0))
    got = sched.answer(p.id, "tired")
    assert got == (p, "tired")
    assert sched.active is None
    assert sched.answer(p.id, "tired") is None  # ikinci dəfə qəbul edilmir


def test_decline_is_valid_answer(sched, clock):
    clock.t = 600
    p = sched.open(SNAP)
    assert sched.answer(p.id, "declined")[1] == "declined"


def test_unknown_answer_rejected(sched, clock):
    clock.t = 600
    p = sched.open(SNAP)
    with pytest.raises(ValueError):
        sched.answer(p.id, "furious")
    with pytest.raises(ValueError):
        sched.answer(p.id, "timeout")  # timeout yalnız sistem tərəfindən


def test_stale_prompt_id_ignored(sched, clock):
    clock.t = 600
    p1 = sched.open(SNAP)
    clock.t = 640
    assert sched.expire() is not None
    clock.t = 1240
    p2 = sched.open(SNAP)
    assert sched.answer(p1.id, "neutral") is None
    assert sched.active is p2


def test_timeout_after_30_seconds(sched, clock):
    clock.t = 600
    p = sched.open(SNAP)
    clock.t = 629.9
    assert sched.expire() is None
    assert sched.seconds_left() == pytest.approx(0.1, abs=1e-6)
    clock.t = 630
    assert sched.expire() == (p, "timeout")
    assert sched.active is None


def test_next_interval_counts_from_last_prompt(sched, clock):
    clock.t = 600
    p = sched.open(SNAP)
    clock.t = 615
    sched.answer(p.id, "neutral")
    assert sched.seconds_until_next() == pytest.approx(585)


def test_snapshot_taken_when_shown(sched, clock):
    clock.t = 600
    snap = SmoothedResult(SmoothStatus.OK, None, "Neutral", 0.7, None, None, 10)
    p = sched.open(snap)
    assert p.snapshot is snap


def test_disabled(clock):
    s = SelfReportScheduler(SelfReportConfig(enabled=False, categories={"a": "A"}), clock)
    clock.t = 10_000
    assert not s.due()


def test_cancel(sched, clock):
    clock.t = 600
    p = sched.open(SNAP)
    assert sched.cancel() is p
    assert sched.active is None
