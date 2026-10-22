import json
import sys
import types

import pytest

from src.benchmark import Run, format_table, frames_from_video, main, plan_runs


def _fake_torch(cuda: bool):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    return torch


def test_plan_cpu_only():
    runs = plan_runs(["enet_b2_8"], "cpu")
    assert [(r.engine, r.device, r.skipped) for r in runs] == [("onnx", "cpu", None)]


def test_plan_gpu_skipped_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    runs = plan_runs(["mbf_va_mtl"], "all")
    assert len(runs) == 2
    gpu = runs[1]
    assert (gpu.engine, gpu.device) == ("torch", "cuda")
    assert "torch" in gpu.skipped


def test_plan_gpu_skipped_without_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    assert "CUDA" in plan_runs(["mbf_va_mtl"], "cuda")[0].skipped


def test_plan_gpu_runs_with_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True))
    assert plan_runs(["mbf_va_mtl"], "cuda")[0].skipped is None


def test_plan_fer_models_cpu_only(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True))
    runs = plan_runs(["deepface_emotion"], "all")
    assert runs[0].skipped is None or "quraşdırılmayıb" in runs[0].skipped
    assert "ONNX" in runs[1].skipped or "quraşdırılmayıb" in runs[1].skipped


def test_plan_together_adds_combined_run():
    runs = plan_runs(["enet_b0_8_best_afew", "mbf_va_mtl"], "cpu", together=True)
    assert runs[-1].model == "enet_b0_8_best_afew+mbf_va_mtl"
    assert runs[-1].skipped is None


@pytest.mark.usefixtures("models_ready")
def test_together_end_to_end(tmp_path):
    out = tmp_path / "t.json"
    assert main(["--model", "enet_b0_8_best_afew,mbf_va_mtl", "--device", "cpu", "--together",
                 "--frames", "3", "--warmup", "1", "--out", str(out)]) == 0
    runs = json.loads(out.read_text(encoding="utf-8"))["runs"]
    single = {r["model"]: r for r in runs}
    combo = single["enet_b0_8_best_afew+mbf_va_mtl"]
    assert combo["model_ms"]["mean"] > single["mbf_va_mtl"]["model_ms"]["mean"]


def test_format_table_shows_skips_and_numbers():
    runs = [
        Run("enet_b0_8_best_afew", "onnx", "cpu", load_s=0.4, detector_ms={"mean": 5, "p95": 6},
            model_ms={"mean": 12, "p95": 14}, total_ms={"mean": 17}, fps=58.8, ram_model_mb=70, ram_peak_mb=160),
        Run("enet_b0_8_best_afew", "torch", "cuda", skipped="CUDA əlçatan deyil"),
    ]
    text = format_table(runs)
    assert "58.8" in text and "12.0 / 14.0" in text
    assert "keçildi: CUDA əlçatan deyil" in text


def test_frames_from_video(test_video):
    frames = frames_from_video(test_video, limit=5)
    assert len(frames) == 5
    assert frames[0].shape == (720, 1280, 3)


@pytest.mark.usefixtures("models_ready")
def test_benchmark_end_to_end(tmp_path, capsys):
    out = tmp_path / "b.json"
    code = main(["--model", "mbf_va_mtl", "--device", "cpu", "--frames", "4", "--warmup", "1", "--out", str(out)])
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    (run,) = data["runs"]
    assert run["error"] is None
    assert run["frames"] == 4 and run["faces"] == 4
    assert run["detector_ms"]["mean"] > 0 and run["model_ms"]["mean"] > 0
    assert run["fps"] > 0 and run["ram_peak_mb"] > 0
    assert data["system"]["onnxruntime"]
    assert "mbf_va_mtl" in capsys.readouterr().out
    assert [p.suffix for p in tmp_path.iterdir()] == [".json"], "benchmark yalnız JSON yazmalıdır"


def test_unknown_model_rejected():
    with pytest.raises(SystemExit):
        main(["--model", "resnet50"])
