import csv
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.config import EMOTION_CLASSES
from src.csv_logger import COLUMNS, PROB_COLUMNS, CsvLogger
from src.detector import FaceStatus
from src.smoothing import SmoothedResult, SmoothStatus

TS = datetime(2026, 10, 22, 14, 30, 5, 123000).astimezone()

EXPECTED_HEADER = [
    "timestamp", "operator_id", "model_name", "engine", "device", "record_type", "status",
    "face_detected", "face_reason", "top_emotion", "top_prob",
    "p_anger", "p_contempt", "p_disgust", "p_fear", "p_happiness", "p_neutral", "p_sadness", "p_surprise",
    "valence", "arousal", "n_frames", "self_report", "latency_ms", "detector_ms", "model_ms", "cpu_percent",
    "expected",
]  # fmt: skip


def smoothed(status=SmoothStatus.OK, top="Neutral", p=0.7, valence=None, arousal=None, n=5):
    if status in (SmoothStatus.NO_FACE, SmoothStatus.WARMING_UP):
        return SmoothedResult(status, None, None, None, None, None, n)
    probs = np.full(8, (1 - p) / 7)
    probs[EMOTION_CLASSES.index(top)] = p
    return SmoothedResult(status, probs, top, p, valence, arousal, n)


@pytest.fixture
def logger(tmp_path):
    lg = CsvLogger(tmp_path, "OP-7F3A", "enet_b0_8_best_afew", "onnx", "cpu")
    yield lg
    lg.close()


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_header_exact(logger, tmp_path):
    logger.log_sample(TS, smoothed(), FaceStatus.OK, 5.0, 12.0, 20.0)
    path = tmp_path / "2026-10-22_OP-7F3A.csv"
    assert path.is_file()
    with path.open(encoding="utf-8") as fh:
        assert fh.readline().strip().split(",") == EXPECTED_HEADER
    assert COLUMNS == EXPECTED_HEADER
    assert PROB_COLUMNS == [f"p_{c.lower()}" for c in EMOTION_CLASSES]


def test_ok_sample_row(logger, tmp_path):
    logger.log_sample(TS, smoothed(top="Happiness", p=0.82), FaceStatus.OK, 4.25, 13.5, 17.3)
    (row,) = read_rows(logger.path_for(TS))
    assert row["timestamp"].startswith("2026-10-22T14:30:05.123")
    assert row["operator_id"] == "OP-7F3A"
    assert row["model_name"] == "enet_b0_8_best_afew"
    assert row["record_type"] == "sample"
    assert row["status"] == "ok"
    assert row["face_detected"] == "1"
    assert row["face_reason"] == "ok"
    assert row["top_emotion"] == "Happiness"
    assert row["top_prob"] == "0.8200"
    assert sum(float(row[c]) for c in PROB_COLUMNS) == pytest.approx(1.0, abs=1e-3)
    assert row["latency_ms"] == "17.8"  # detektor + model
    assert row["detector_ms"] == "4.2" or row["detector_ms"] == "4.3"
    assert row["model_ms"] == "13.5"
    assert row["cpu_percent"] == "17.3"
    assert row["valence"] == "" and row["arousal"] == ""
    assert row["self_report"] == ""


def test_uncertain_row_marked_but_keeps_probabilities(logger):
    logger.log_sample(TS, smoothed(status=SmoothStatus.UNCERTAIN, top="Sadness", p=0.41), FaceStatus.OK, 5, 12, 10)
    (row,) = read_rows(logger.path_for(TS))
    assert row["status"] == "uncertain"
    assert row["top_emotion"] == "uncertain"
    assert row["top_prob"] == "0.4100"
    assert row["p_sadness"] == "0.4100"


def test_no_face_row_has_no_prediction(logger):
    logger.log_sample(TS, smoothed(status=SmoothStatus.NO_FACE, n=4), FaceStatus.OCCLUDED, 6.0, None, 9.0)
    (row,) = read_rows(logger.path_for(TS))
    assert row["status"] == "no_face"
    assert row["face_detected"] == "0"
    assert row["face_reason"] == "occluded"
    assert row["top_emotion"] == "" and row["top_prob"] == ""
    assert all(row[c] == "" for c in PROB_COLUMNS)
    assert row["model_ms"] == "" and row["latency_ms"] == "6.0"


def test_warming_up_row(logger):
    logger.log_sample(TS, smoothed(status=SmoothStatus.WARMING_UP, n=1), FaceStatus.OK, 5, 12, 10)
    (row,) = read_rows(logger.path_for(TS))
    assert row["status"] == "warming_up"
    assert row["face_detected"] == "1"
    assert row["top_emotion"] == ""


def test_mtl_valence_arousal(logger):
    logger.log_sample(TS, smoothed(valence=0.3456, arousal=-0.12), FaceStatus.OK, 5, 12, 10)
    (row,) = read_rows(logger.path_for(TS))
    assert row["valence"] == "0.3456"
    assert row["arousal"] == "-0.1200"


def test_self_report_row(logger):
    logger.log_self_report(TS, smoothed(top="Neutral", p=0.66), "tired")
    (row,) = read_rows(logger.path_for(TS))
    assert row["record_type"] == "self_report"
    assert row["self_report"] == "tired"
    assert row["top_emotion"] == "Neutral"
    assert row["latency_ms"] == ""


def test_append_does_not_duplicate_header(tmp_path):
    for _ in range(2):
        lg = CsvLogger(tmp_path, "OP-7F3A", "m", "onnx", "cpu")
        lg.log_sample(TS, smoothed(), FaceStatus.OK, 5, 12, 10)
        lg.close()
    rows = read_rows(tmp_path / "2026-10-22_OP-7F3A.csv")
    assert len(rows) == 2


def test_daily_rotation(logger, tmp_path):
    logger.log_sample(TS, smoothed(), FaceStatus.OK, 5, 12, 10)
    logger.log_sample(TS + timedelta(days=1), smoothed(), FaceStatus.OK, 5, 12, 10)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["2026-10-22_OP-7F3A.csv", "2026-10-23_OP-7F3A.csv"]


def test_old_schema_file_is_not_corrupted(tmp_path):
    old = tmp_path / "2026-10-22_OP-7F3A.csv"
    old.write_text("timestamp,emotion\n2026-10-22,Neutral\n", encoding="utf-8")
    lg = CsvLogger(tmp_path, "OP-7F3A", "m", "onnx", "cpu")
    lg.log_sample(TS, smoothed(), FaceStatus.OK, 5, 12, 10)
    lg.close()
    assert old.read_text(encoding="utf-8") == "timestamp,emotion\n2026-10-22,Neutral\n"
    assert len(read_rows(tmp_path / "2026-10-22_OP-7F3A_2.csv")) == 1


def test_only_csv_files_created(logger, tmp_path):
    for i in range(20):
        logger.log_sample(TS + timedelta(seconds=i), smoothed(), FaceStatus.OK, 5, 12, 10)
    assert [p.suffix for p in tmp_path.iterdir()] == [".csv"]


def test_rows_parse_as_numbers(logger):
    logger.log_sample(TS, smoothed(valence=0.1, arousal=0.2), FaceStatus.OK, 5, 12, 10)
    (row,) = read_rows(logger.path_for(TS))
    for col in [*PROB_COLUMNS, "top_prob", "valence", "arousal", "latency_ms", "cpu_percent", "n_frames"]:
        float(row[col])
