"""Oflayn qiymətləndirici: metrikalar, ayar müqayisəsi, keş, CLI və hesabatın məxfiliyi."""

import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import EMOTION_CLASSES
from src.dataset import CACHE_DIR, DatasetWriter, load_samples
from src.evaluate import (
    Meta,
    build_variants,
    loto,
    main as evaluate_cli,
    render_html,
    run_evaluation,
    score,
    setting_effects,
    sign_test,
    write_csv,
)
from src.recognizer import EmotionPrediction
from src.variants import Variant

from .conftest import face_frame

IDX = {c: i for i, c in enumerate(EMOTION_CLASSES)}
EXPR = ["Happiness", "Neutral", "Sadness"]


def onehot(cls, p=0.9):
    v = np.full(len(EMOTION_CLASSES), (1 - p) / (len(EMOTION_CLASSES) - 1))
    v[IDX[cls]] = p
    return v


# --- saf funksiyalar ----------------------------------------------------------------


def test_sign_test():
    a = {i: 1.0 for i in range(10)}
    b = {i: 0.0 for i in range(10)}
    r = sign_test(a, b)
    assert (r["better"], r["worse"], r["ties"]) == (10, 0, 0)
    assert r["p"] == pytest.approx(2 / 2**10)
    assert sign_test(a, a)["p"] is None


def test_score_metrics(cfg):
    # 2 tester × 3 ifadə × 4 kadr; Sadness həmişə səhv (Neutral), qalanı düz
    exp, probs, steps, testers = [], [], [], []
    for t in range(2):
        for j, e in enumerate(EXPR):
            for _ in range(4):
                exp.append(IDX[e])
                probs.append(onehot("Neutral" if e == "Sadness" else e, 0.55 if e == "Neutral" else 0.9))
                steps.append(t * 10 + j)
                testers.append(t)
    meta = Meta(np.array(exp), np.array(steps), np.array(testers))
    r = score(np.array(probs), np.ones(len(exp)), meta, frozenset(EMOTION_CLASSES), cfg, (0.5, 0.6))
    assert r["acc"] == pytest.approx(2 / 3) and r["balanced"] == pytest.approx(2 / 3)
    assert r["step_acc"] == pytest.approx(2 / 3) and r["n_steps"] == 6
    assert r["per_expr"]["Sadness"] == (0.0, 8)
    assert r["conf"][0.5][0] == 1.0
    assert r["conf"][0.6] == (pytest.approx(2 / 3), 0.5)  # 0.9-luq kadrlar: Happiness düz, Sadness səhv
    assert r["spread"] == (pytest.approx(2 / 3), pytest.approx(2 / 3))
    # FER modelində dəstəklənməyən ifadə hesabdan çıxır
    fer = frozenset(EMOTION_CLASSES) - {"Sadness"}
    assert score(np.array(probs), np.ones(len(exp)), meta, fer, cfg, (0.5,))["acc"] == 1.0


def test_setting_effects_and_loto():
    rows = {
        Variant(0.1): {"balanced": 0.60}, Variant(0.1, True): {"balanced": 0.65},
        Variant(0.0): {"balanced": 0.70}, Variant(0.0, True): {"balanced": 0.71},
    }
    eff = setting_effects(rows, 0.1)
    assert eff["align"]["delta"] == pytest.approx(0.03) and eff["align"]["better"] == 2
    assert eff["m0"]["delta"] == pytest.approx(0.08) and eff["m0"]["n"] == 2
    assert "flip" not in eff
    # LOTO: "b" bütün testerlərdə yaxşıdır → hər dəfə seçilir
    cands = {"a": {0: 0.5, 1: 0.6, 2: 0.4}, "b": {0: 0.7, 1: 0.7, 2: 0.6}}
    lo = loto(cands, "a")
    assert lo["pick"] == "b" and lo["improved"] == 3 and lo["gain"] == pytest.approx(0.5 / 3)
    assert loto({"a": {0: 0.5, 1: 0.6}, "b": {0: 0.7, 1: 0.7}}, "a") is None  # < 3 tester


def test_build_variants_includes_live_setting():
    vs = build_variants([0.0, 0.2], (False,), (False, True), Variant(0.1, False, False))
    assert vs[0] == Variant(0.1) and len(vs) == 5


# --- süni dataset + saxta model ------------------------------------------------------


def make_dataset(root: Path, testers=("T-01", "T-02", "T-03"), frames_per_step=3):
    """Hər şəkil: boz fon, ortada "üz" kvadratı — rəngi gözlənilən sinfi kodlayır."""
    for t in testers:
        sdir = root / f"20261022_100000_{t}"
        sdir.mkdir(parents=True)
        (sdir / "meta.json").write_text(json.dumps({"tester_id": t, "frames": 0}), encoding="utf-8")
        with (sdir / "frames.csv").open("w", encoding="utf-8", newline="") as fh:
            cols = ["file", "step", "round", "expected", "face_status", "box_x", "box_y", "box_w", "box_h"]
            cols += [f"lm{i}_{a}" for i in range(5) for a in "xy"]
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            n = 0
            for step, e in enumerate(EXPR):
                for _ in range(frames_per_step):
                    n += 1
                    img = np.full((200, 200, 3), 128, np.uint8)
                    img[60:140, 60:140] = 20 * IDX[e] + 10
                    name = f"{n:05d}_{e}.jpg"
                    (sdir / name).write_bytes(cv2.imencode(".png", img)[1].tobytes())
                    row = {"file": name, "step": step, "round": 1, "expected": e, "face_status": "ok",
                           "box_x": 60, "box_y": 60, "box_w": 80, "box_h": 80}
                    for i, (x, y) in enumerate([(85, 90), (115, 90), (100, 105), (88, 120), (112, 120)]):
                        row[f"lm{i}_x"], row[f"lm{i}_y"] = x, y
                    w.writerow(row)


class FakeRecognizer:
    """Mərkəzdəki rəngdən sinfi "tanıyır", amma kəsik geniş olanda (boşluq > 10%) Neutral deyir."""

    calls = 0

    def predict(self, crop):
        FakeRecognizer.calls += 1
        h, w = crop.shape[:2]
        cls = EMOTION_CLASSES[int(round((int(crop[h // 2, w // 2, 0]) - 10) / 20))]
        if w > 98:  # 80 px üz + hər tərəfə 10%-dən çox
            cls = "Neutral"
        return EmotionPrediction(onehot(cls), None, None, 1.0)


def test_run_evaluation_finds_best_margin_and_caches(cfg, tmp_path):
    make_dataset(tmp_path)
    samples = load_samples(tmp_path)
    assert len(samples) == 27
    baseline = Variant(0.1, False, False)
    variants = build_variants([0.0, 0.1, 0.3], (False,), (False, True), baseline)
    models = ["enet_b0_8_best_afew", "deepface_emotion"]
    FakeRecognizer.calls = 0
    res = run_evaluation(cfg, samples, models, variants, baseline, recognizer_factory=lambda k: FakeRecognizer())
    first_calls = FakeRecognizer.calls
    assert first_calls == 27 * 3 * 2 * 2  # nümunə × kəsik × (adi + flip) × model

    bm = res["by_model"]["enet_b0_8_best_afew"]
    assert bm["baseline"]["balanced"] == 1.0
    assert bm["effects"]["m30"]["delta"] == pytest.approx(-2 / 3)  # geniş kəsik: yalnız Neutral düz
    assert bm["loto"]["gain"] == 0.0 and bm["loto"]["pick"].margin <= 0.1
    assert res["global"]["live"]["model"] == cfg.model.name
    assert res["n_testers"] == 3 and res["n_steps"] == 9
    assert all(r["balanced"] <= 1.0 for r in res["rows"])

    # Keş: ikinci qaçışda model çağırılmır
    assert any((p / CACHE_DIR).is_dir() for p in tmp_path.iterdir())
    run_evaluation(cfg, samples, models, variants, baseline, recognizer_factory=lambda k: FakeRecognizer())
    assert FakeRecognizer.calls == first_calls

    # Hesabat və CSV-də tester ID-ləri yoxdur
    html_text = render_html(res, cfg)
    out = write_csv(res, tmp_path / "r.csv")
    text = out.read_text(encoding="utf-8")
    for t in ("T-01", "T-02", "T-03"):
        assert t not in html_text and t not in text
    assert "Model × ayar" in html_text and "LOTO" in html_text
    rows = list(csv.DictReader(text.splitlines()))
    assert len(rows) == len(res["rows"]) and rows[0]["balanced"] == "1.0000"


# --- real model ilə CLI ----------------------------------------------------------------


@pytest.mark.usefixtures("models_ready")
def test_cli_end_to_end_real_model(cfg, face_bgr, tmp_path, monkeypatch, capsys):
    from src.detector import FaceDetector

    frame = face_frame(face_bgr)
    det = FaceDetector(cfg.detector).detect(frame)
    assert det.face is not None
    writer = DatasetWriter(cfg.research.__class__(save_test_frames=True, dir=tmp_path / "ds"), "T-21", "OP-1", cfg.test)
    for step, e in enumerate(["Happiness", "Neutral"]):
        for _ in range(2):
            writer.add(frame, det, e, step, 1, 1.0, datetime.now())
    writer.close()

    import src.evaluate as ev

    monkeypatch.setattr(ev, "load_config", lambda _p=None: cfg)
    out = tmp_path / "rep" / "e.html"
    code = evaluate_cli(["--dataset", str(tmp_path / "ds"), "--models", cfg.model.name, "--margins", "0,0.1",
                         "--align", "both", "--flip", "on", "--out", str(out)])
    assert code == 0
    assert out.is_file() and out.with_suffix(".csv").is_file()
    printed = capsys.readouterr().out
    assert "Hər model üçün ən yaxşı ayar" in printed
    assert "T-21" not in out.read_text(encoding="utf-8")
