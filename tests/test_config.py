import pytest

from src.config import EMOTION_CLASSES, PROJECT_ROOT, ConfigError, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.model.name == "enet_b0_8_best_afew"
    assert cfg.model.engine == "onnx"
    assert cfg.detector.score_threshold == 0.8
    assert cfg.smoothing.window_seconds == 10
    assert cfg.smoothing.uncertain_threshold == 0.3
    assert cfg.self_report.interval_minutes == 10
    assert cfg.self_report.timeout_seconds == 30
    assert set(cfg.mapping) == set(EMOTION_CLASSES)
    assert 1 <= cfg.capture.analysis_fps <= 2


def test_relative_paths_resolved_against_project_root():
    cfg = load_config()
    assert cfg.detector.model_path.is_absolute()
    assert cfg.detector.model_path.parent == PROJECT_ROOT / "models"
    assert cfg.logging.dir == PROJECT_ROOT / "logs"


@pytest.mark.parametrize("model", ["enet_b0_8_va_mtl", "enet_b2_8", "mbf_va_mtl"])
def test_alternative_models_accepted(make_config, model):
    cfg = make_config(model__name=model)
    assert cfg.model.name == model
    assert cfg.model.is_mtl == model.endswith("_mtl")


def test_seven_class_model_rejected(make_config):
    with pytest.raises(ConfigError, match="Dəstəklənməyən"):
        make_config(model__name="enet_b2_7")


def test_onnx_with_cuda_rejected(make_config):
    with pytest.raises(ConfigError, match="CPU"):
        make_config(model__engine="onnx", model__device="cuda")


def test_torch_cuda_combination_parses(make_config):
    cfg = make_config(model__engine="torch", model__device="cuda")
    assert (cfg.model.engine, cfg.model.device) == ("torch", "cuda")


@pytest.mark.parametrize("bad_id", ["Operator", "Ayşən Məmmədova", "a", "", None, "x" * 40])
def test_operator_id_must_be_anonymous(make_config, bad_id):
    with pytest.raises(ConfigError):
        make_config(operator_id=bad_id)


@pytest.mark.parametrize("fps", [0.1, 5, 30])
def test_analysis_fps_bounded(make_config, fps):
    with pytest.raises(ConfigError, match="analysis_fps"):
        make_config(capture__analysis_fps=fps)


def test_mapping_must_cover_all_classes(make_config, raw_config):
    mapping = dict(raw_config["mapping"])
    mapping.pop("Fear")
    with pytest.raises(ConfigError, match="Fear"):
        make_config(mapping=mapping)


def test_mapping_targets_must_be_known_categories(make_config, raw_config):
    mapping = dict(raw_config["mapping"])
    mapping["Fear"] = "panic"
    with pytest.raises(ConfigError, match="panic"):
        make_config(mapping=mapping)


def test_unknown_key_rejected(make_config):
    with pytest.raises(ConfigError, match="naməlum"):
        make_config(model__nmae="typo")


def test_reserved_category_codes_rejected(make_config, raw_config):
    cats = dict(raw_config["self_report"]["categories"], timeout="Vaxt bitdi")
    with pytest.raises(ConfigError, match="timeout"):
        make_config(self_report__categories=cats)
