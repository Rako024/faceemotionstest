import numpy as np
import pytest

from src.config import EMOTION_CLASSES, SmoothingConfig
from src.recognizer import EmotionPrediction
from src.smoothing import EmotionSmoother, SmoothStatus

IDX = {name: i for i, name in enumerate(EMOTION_CLASSES)}


def pred(dominant="Neutral", p=0.8, valence=None, arousal=None):
    probs = np.full(len(EMOTION_CLASSES), (1.0 - p) / (len(EMOTION_CLASSES) - 1))
    probs[IDX[dominant]] = p
    return EmotionPrediction(probs, valence, arousal, latency_ms=10.0)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def smoother(clock):
    return EmotionSmoother(SmoothingConfig(window_seconds=10, min_frames=3, uncertain_threshold=0.5), clock)


def feed(smoother, clock, preds, step=0.5):
    result = None
    for p in preds:
        result = smoother.add(p)
        clock.t += step
    return result


def test_single_frame_is_never_shown(smoother):
    r = smoother.add(pred("Happiness", 0.99))
    assert r.status is SmoothStatus.WARMING_UP
    assert r.top_emotion is None and r.probs is None
    assert not r.has_estimate


def test_result_after_min_frames(smoother, clock):
    r = feed(smoother, clock, [pred("Neutral", 0.8)] * 3)
    assert r.status is SmoothStatus.OK
    assert r.top_emotion == "Neutral"
    assert r.top_prob == pytest.approx(0.8)
    assert r.n_frames == 3


def test_moving_average_is_mean_of_window(smoother, clock):
    preds = [pred("Neutral", 0.9), pred("Neutral", 0.6), pred("Happiness", 0.9)]
    r = feed(smoother, clock, preds)
    expected = np.mean([p.probs for p in preds], axis=0)
    np.testing.assert_allclose(r.probs, expected)
    assert r.probs.sum() == pytest.approx(1.0)


def test_single_outlier_does_not_flip_result(smoother, clock):
    feed(smoother, clock, [pred("Neutral", 0.85)] * 8)
    r = smoother.add(pred("Anger", 0.95))
    assert r.top_emotion == "Neutral"


def test_old_frames_leave_window(smoother, clock):
    feed(smoother, clock, [pred("Anger", 0.9)] * 5, step=1.0)
    clock.t += 6.0  # ilk kadrlar artıq 10 saniyədən köhnədir
    r = feed(smoother, clock, [pred("Happiness", 0.9)] * 3, step=1.0)
    assert r.top_emotion == "Happiness"
    assert r.n_frames == 3 + 1  # yalnız son 10 saniyə


def test_window_boundary_is_exclusive(smoother, clock):
    smoother.add(pred(), t=0.0)
    smoother.add(pred(), t=5.0)
    r = smoother.add(pred(), t=10.0)
    assert r.n_frames == 2  # t=0 kadrı tam 10 s köhnədir → çıxır


@pytest.mark.parametrize("p,status", [(0.49, SmoothStatus.UNCERTAIN), (0.5, SmoothStatus.OK), (0.51, SmoothStatus.OK)])
def test_uncertain_threshold(smoother, clock, p, status):
    r = feed(smoother, clock, [pred("Sadness", p)] * 3)
    assert r.status is status
    assert r.top_emotion == "Sadness"  # uncertain-da da ən yüksək sinif saxlanılır
    assert r.has_estimate


def test_no_face_hides_previous_estimate(smoother, clock):
    feed(smoother, clock, [pred("Neutral", 0.9)] * 5)
    r = smoother.add(None)
    assert r.status is SmoothStatus.NO_FACE
    assert r.probs is None and r.top_emotion is None
    assert r.n_frames == 5  # pəncərə silinmir


def test_face_returns_within_window_uses_recent_frames(smoother, clock):
    feed(smoother, clock, [pred("Neutral", 0.9)] * 4, step=1.0)
    feed(smoother, clock, [None] * 2, step=1.0)
    r = smoother.add(pred("Neutral", 0.7))
    assert r.status is SmoothStatus.OK
    assert r.n_frames == 5


def test_face_returns_after_long_absence_warms_up_again(smoother, clock):
    feed(smoother, clock, [pred("Neutral", 0.9)] * 4, step=1.0)
    feed(smoother, clock, [None] * 12, step=1.0)
    r = smoother.add(pred("Neutral", 0.9))
    assert r.status is SmoothStatus.WARMING_UP


def test_valence_arousal_averaged(smoother, clock):
    r = feed(smoother, clock, [pred(valence=0.2, arousal=-0.1), pred(valence=0.4, arousal=0.1), pred(valence=0.6, arousal=0.3)])
    assert r.valence == pytest.approx(0.4)
    assert r.arousal == pytest.approx(0.1)


def test_valence_none_for_non_mtl(smoother, clock):
    r = feed(smoother, clock, [pred()] * 3)
    assert r.valence is None and r.arousal is None


def test_current_does_not_mutate(smoother, clock):
    feed(smoother, clock, [pred("Neutral", 0.8)] * 3)
    a = smoother.current()
    b = smoother.current()
    assert a.n_frames == b.n_frames == 3
    assert a.top_emotion == b.top_emotion


def test_current_expires_with_time(smoother, clock):
    feed(smoother, clock, [pred()] * 3)
    clock.t += 30
    assert smoother.current().status is SmoothStatus.WARMING_UP


def test_initial_state_is_no_face(smoother):
    assert smoother.current().status is SmoothStatus.NO_FACE


def test_reset(smoother, clock):
    feed(smoother, clock, [pred()] * 3)
    smoother.reset()
    assert smoother.current().n_frames == 0
