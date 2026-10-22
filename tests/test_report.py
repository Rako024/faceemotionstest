import csv
import re
from datetime import datetime

import pandas as pd
import pytest

from src import report as rep
from src.csv_logger import COLUMNS
from src.report import (
    NO_PREDICTION,
    UNCERTAIN,
    ReportError,
    accuracy,
    cohen_kappa,
    compute,
    find_logs,
    load_logs,
    main,
    render_html,
    wilson,
)

from .synthetic import write_synthetic_logs


def make_df(rows):
    """Minimal CSV sətirlərindən DataFrame (load_logs ilə eyni tiplər)."""
    full = []
    for r in rows:
        base = {c: "" for c in COLUMNS}
        base.update(timestamp="2026-10-22T10:00:00.000+04:00", operator_id="OP-1A", model_name="m",
                    engine="onnx", device="cpu", record_type="self_report", face_detected="1")
        base.update(r)
        full.append(base)
    df = pd.DataFrame(full)
    for col in rep.NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = df["timestamp"].str.slice(0, 10)
    df["hour"] = pd.to_numeric(df["timestamp"].str.slice(11, 13))
    return df


def report_row(answer, status="ok", top="Neutral", arousal=""):
    return {"self_report": answer, "status": status, "top_emotion": top if status == "ok" else ("uncertain" if status == "uncertain" else ""), "arousal": arousal}


@pytest.fixture
def small_df():
    return make_df([
        report_row("neutral", top="Neutral"),        # uyğun
        report_row("neutral", top="Neutral"),        # uyğun
        report_row("happy", top="Happiness"),        # uyğun
        report_row("happy", top="Neutral"),          # uyğunsuz
        report_row("angry", top="Disgust"),          # uyğun (Disgust → angry)
        report_row("stressed", top="Surprise"),      # uyğun (Surprise → stressed)
        report_row("sad", status="uncertain"),       # qeyri-müəyyən
        report_row("tired", top="Neutral"),          # modeldə qarşılığı yoxdur
        report_row("neutral", status="no_face"),     # proqnoz yoxdur
        report_row("declined", top="Neutral"),       # sayılmır
        report_row("timeout", top="Neutral"),        # sayılmır
    ])


def test_accuracy_counts(small_df, cfg):
    acc = accuracy(small_df, cfg)
    assert acc["n_prompts"] == 11
    assert acc["n_answered"] == 9
    assert acc["n_declined"] == 1 and acc["n_timeout"] == 1
    assert acc["overall"] == (5, 9)
    assert acc["confident"] == (5, 7)       # uncertain və no_face çıxır
    assert acc["comparable"] == (5, 6)      # "tired" çıxır
    assert acc["unmappable"] == ["tired"]
    assert acc["coverage"] == pytest.approx(7 / 9)


def test_confusion_matrix(small_df, cfg):
    acc = accuracy(small_df, cfg)
    cats, cols, m = acc["categories"], acc["columns"], acc["matrix"]
    assert cols[-2:] == [UNCERTAIN, NO_PREDICTION]
    cell = lambda r, c: m[cats.index(r)][cols.index(c)]  # noqa: E731
    assert cell("neutral", "neutral") == 2
    assert cell("neutral", NO_PREDICTION) == 1
    assert cell("happy", "neutral") == 1
    assert cell("sad", UNCERTAIN) == 1
    assert cell("tired", "neutral") == 1
    assert sum(map(sum, m)) == acc["n_answered"]


def test_per_category_recall_precision(small_df, cfg):
    per = {c["code"]: c for c in accuracy(small_df, cfg)["per_category"]}
    assert per["happy"]["recall"] == pytest.approx(0.5)
    assert per["neutral"]["precision"] == pytest.approx(2 / 4)  # 2 neutral + happy + tired → neutral
    assert per["sad"]["recall"] is None
    assert per["tired"]["mappable"] is False


def test_va_tired_rule(small_df, make_config):
    cfg = make_config(report__va_tired_rule={"enabled": True, "arousal_max": -0.3})
    df = small_df.copy()
    df.loc[df["self_report"] == "tired", "arousal"] = -0.6
    acc = accuracy(df, cfg)
    assert acc["unmappable"] == []
    per = {c["code"]: c for c in acc["per_category"]}
    assert per["tired"]["recall"] == 1.0


def test_wilson_interval():
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.404, abs=1e-3) and hi == pytest.approx(0.596, abs=1e-3)
    assert wilson(0, 0) is None
    assert wilson(10, 10)[1] == 1.0


def test_cohen_kappa():
    labels = ["a", "b"]
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"], labels) == pytest.approx(1.0)
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"], labels) == pytest.approx(0.0)
    assert cohen_kappa([], [], labels) is None


@pytest.fixture
def synth_logs(tmp_path):
    d = tmp_path / "logs"
    write_synthetic_logs(d, days=1, hours=range(9, 13), samples_per_hour=30)
    return d


def test_load_logs_from_real_logger_output(synth_logs):
    df = load_logs(find_logs(synth_logs))
    assert set(df["operator_id"]) == {"OP-1A", "OP-2B", "OP-3C"}
    assert df["latency_ms"].dtype.kind == "f"
    assert set(df["hour"].dropna().astype(int)) == {9, 10, 11, 12}


def test_load_logs_rejects_foreign_csv(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ReportError, match="çatışmayan"):
        load_logs([p])


def test_quality_shares_sum_to_one(synth_logs, cfg):
    data = compute(load_logs(find_logs(synth_logs)), cfg)
    q = data["quality"]
    assert sum(q["share"].values()) == pytest.approx(1.0)
    assert q["n_no_face"] == q["counts"]["no_face"]
    assert sum(k for _, k in q["reasons"]) == q["n_no_face"]


def test_hourly_flags_small_hours(synth_logs, cfg):
    data = compute(load_logs(find_logs(synth_logs)), cfg)
    rows = data["hourly"]["rows"]
    assert [r["hour"] for r in rows] == [9, 10, 11, 12]
    assert all(r["enough"] for r in rows)  # 3 operator × ~24 qiymətli kadr
    for r in rows:
        assert sum(share for _, share in r["shares"].values()) == pytest.approx(1.0)


def test_html_has_no_individual_data(synth_logs, cfg):
    data = compute(load_logs(find_logs(synth_logs)), cfg)
    html = render_html(data, cfg, generated=datetime(2026, 10, 22, 12, 0))
    for op in ("OP-1A", "OP-2B", "OP-3C"):
        assert op not in html, "hesabatda operator ID olmamalıdır"
    assert "reytinq" in html  # "Fərdi reytinq yoxdur" qeydi
    assert not re.search(r"https?://", html)
    assert "default-src 'none'" in html
    for section in ("Model nəticəsi və operatorun öz cavabı", "Məlumat keyfiyyəti",
                    "Gün ərzində emosiya paylanması", "Performans"):
        assert section in html
    assert html.count("Cədvəl görünüşü") == 3


def test_small_group_banner(tmp_path, cfg):
    d = tmp_path / "one"
    write_synthetic_logs(d, operators=("OP-9Z",), days=1, hours=range(10, 11))
    html = render_html(compute(load_logs(find_logs(d)), cfg), cfg)
    assert "Kiçik qrup" in html


def test_no_small_group_banner_with_enough_operators(synth_logs, cfg):
    html = render_html(compute(load_logs(find_logs(synth_logs)), cfg), cfg)
    assert "Kiçik qrup" not in html


def test_va_section_only_for_mtl(tmp_path, cfg):
    d = tmp_path / "mtl"
    write_synthetic_logs(d, days=1, hours=range(9, 12), model="enet_b0_8_va_mtl")
    data = compute(load_logs(find_logs(d)), cfg)
    assert data["va"] is not None
    assert "Valentlik və oyanma" in render_html(data, cfg)


def test_model_comparison_when_several_models(tmp_path, cfg):
    d = tmp_path / "multi"
    write_synthetic_logs(d, operators=("OP-1A",), days=1, model="enet_b0_8_best_afew")
    write_synthetic_logs(d, operators=("OP-2B",), days=1, model="mbf_va_mtl", seed=3)
    data = compute(load_logs(find_logs(d)), cfg)
    assert {m["model"] for m in data["by_model"]} == {"enet_b0_8_best_afew", "mbf_va_mtl"}
    assert "Modellərin müqayisəsi" in render_html(data, cfg)


@pytest.fixture
def multi_logs(tmp_path):
    d = tmp_path / "parallel"
    write_synthetic_logs(d, days=1, hours=range(9, 13), samples_per_hour=30, reports_per_hour=3,
                         extra_models={"deepface_emotion": 0.8, "vit_fer2013": 0.5})
    return d


def test_primary_model_drives_headline(multi_logs, cfg):
    from dataclasses import replace

    cfg = replace(cfg, model=replace(cfg.model, name="enet_b0_8_best_afew"))  # sintetik logların əsas modeli
    df = load_logs(find_logs(multi_logs))
    data = compute(df, cfg)
    assert data["primary"] == "enet_b0_8_best_afew"
    primary_rows = df[(df["model_name"] == "enet_b0_8_best_afew") & (df["record_type"] == "sample")]
    assert data["n_samples"] == len(primary_rows), "keyfiyyət/saat göstəriciləri yalnız əsas model üzrə"
    assert data["accuracy"]["n_prompts"] == int(((df["model_name"] == "enet_b0_8_best_afew") & (df["record_type"] == "self_report")).sum())


def test_models_compared_on_same_prompts(multi_logs, cfg):
    data = compute(load_logs(find_logs(multi_logs)), cfg)
    by = {m["model"]: m for m in data["by_model"]}
    assert set(by) == {"enet_b0_8_best_afew", "deepface_emotion", "vit_fer2013"}
    answered = {m["answered"] for m in by.values()}
    assert len(answered) == 1, "hər model eyni sorğu cavabları üzrə qiymətləndirilməlidir"
    assert by["deepface_emotion"]["dataset"] == "FER2013"
    assert by["vit_fer2013"]["model_ms"] > 0


def test_pairwise_agreement_reflects_similarity(multi_logs, cfg):
    data = compute(load_logs(find_logs(multi_logs)), cfg)
    pw = data["pairwise"]
    cells = pw["cells"]
    assert cells[("enet_b0_8_best_afew", "enet_b0_8_best_afew")]["agree"] == 1.0
    close = cells[("enet_b0_8_best_afew", "deepface_emotion")]["agree"]
    far = cells[("enet_b0_8_best_afew", "vit_fer2013")]["agree"]
    assert close > far, "80% eyni nəticə verən model 50%-lik modeldən daha razı olmalıdır"
    assert cells[("deepface_emotion", "vit_fer2013")] == cells[("vit_fer2013", "deepface_emotion")] or \
        cells[("deepface_emotion", "vit_fer2013")]["agree"] == cells[("vit_fer2013", "deepface_emotion")]["agree"]


def test_comparison_section_rendered(multi_logs, cfg):
    html = render_html(compute(load_logs(find_logs(multi_logs)), cfg), cfg)
    assert "Modellərin müqayisəsi" in html and "Modellər arası razılaşma" in html
    assert "ViT (HF)" in html and "DeepFace" in html
    assert html.index("Modellərin müqayisəsi") < html.index("Model nəticəsi və operatorun öz cavabı")
    for op in ("OP-1A", "OP-2B", "OP-3C"):
        assert op not in html


@pytest.fixture
def test_logs(tmp_path):
    from .synthetic import write_synthetic_test

    d = tmp_path / "calib"
    write_synthetic_logs(d, days=1, hours=range(9, 11), samples_per_hour=20)
    write_synthetic_test(d, {
        "enet_b0_8_best_afew": {"Neutral": 0.6, "Happiness": 0.9, "Anger": 0.5},
        "deepface_emotion": {"Neutral": 0.3, "Happiness": 1.0, "Anger": 0.2},
    })
    return d


def test_calibration_accuracy_from_test_rows(test_logs, cfg):
    data = compute(load_logs(find_logs(test_logs)), cfg)
    cal = data["calibration"]
    assert cal["expressions"] == ["Anger", "Happiness", "Neutral"]
    by = {m["model"]: m for m in cal["models"]}
    enet = by["enet_b0_8_best_afew"]
    assert enet["face"] == 60 and enet["frames"] == 61  # üzsüz kadr məxrəcdə deyil
    assert enet["accuracy"] == pytest.approx((12 + 18 + 10) / 60)
    assert enet["per_expression"]["Happiness"]["acc"] == pytest.approx(0.9)
    assert enet["balanced"] == pytest.approx((0.6 + 0.9 + 0.5) / 3)
    assert enet["uncertain"] > 0  # qeyri-müəyyən kadrlar argmax ilə yenə düzgün sayılıb
    assert cal["models"][0]["model"] == "enet_b0_8_best_afew"  # balanslı düzgünlüyə görə sıralanır
    assert by["deepface_emotion"]["confusion"]["Anger"]["Neutral"] == 16


def test_test_rows_do_not_touch_field_stats(test_logs, cfg):
    df = load_logs(find_logs(test_logs))
    data = compute(df, cfg)
    field = df[(df["record_type"] == "sample") & (df["model_name"] == data["primary"])]
    assert data["n_samples"] == len(field)
    assert data["accuracy"]["n_prompts"] == int(
        ((df["record_type"] == "self_report") & (df["model_name"] == data["primary"])).sum()
    )


def test_calibration_section_rendered(test_logs, cfg):
    html = render_html(compute(load_logs(find_logs(test_logs)), cfg), cfg)
    assert "Nəzarətli test — göstərilən ifadələr" in html
    assert "Səhvlərin quruluşu" in html
    assert "OP-5T" not in html


def test_report_without_test_rows_has_no_section(synth_logs, cfg):
    data = compute(load_logs(find_logs(synth_logs)), cfg)
    assert data["calibration"] is None
    assert "Nəzarətli test" not in render_html(data, cfg)


def test_old_schema_csv_without_expected_loads(tmp_path, cfg):
    from src.csv_logger import COLUMNS

    old_cols = [c for c in COLUMNS if c != "expected"]
    p = tmp_path / "2026-10-01_OP-1A.csv"
    row = {c: "" for c in old_cols}
    row.update(timestamp="2026-10-01T10:00:00.000+04:00", operator_id="OP-1A", model_name="enet_b0_8_best_afew",
               record_type="sample", status="no_face", face_detected="0", n_frames="0")
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=old_cols)
        w.writeheader()
        w.writerow(row)
    df = load_logs([p])
    assert df["expected"].tolist() == [""]
    assert compute(df, cfg)["calibration"] is None


def test_fer_rows_have_blank_contempt(multi_logs):
    df = load_logs(find_logs(multi_logs))
    fer = df[(df["model_name"] == "deepface_emotion") & (df["status"] == "ok")]
    assert fer["p_contempt"].isna().all()


def test_cli_writes_report(synth_logs, tmp_path, capsys):
    out = tmp_path / "r.html"
    assert main(["--logs", str(synth_logs), "--out", str(out)]) == 0
    assert out.is_file() and out.stat().st_size > 5000
    assert "uyğunluq" in capsys.readouterr().out


def test_cli_date_filter_empty(synth_logs, tmp_path, capsys):
    assert main(["--logs", str(synth_logs), "--out", str(tmp_path / "x.html"), "--from", "2030-01-01"]) == 1
    assert "Xəta" in capsys.readouterr().err


def test_cli_no_logs(tmp_path, capsys):
    assert main(["--logs", str(tmp_path), "--out", str(tmp_path / "x.html")]) == 1
