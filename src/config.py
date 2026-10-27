"""config.yaml oxunması və yoxlanılması."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# EmotiEffLib-in 8 sinifli modellərindəki sıra (idx_to_emotion_class).
EMOTION_CLASSES: tuple[str, ...] = (
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
)

# EmotiEffLib: yalnız 8 sinifli modellər (enet_b2_7 daxil deyil).
EMOTIEFFLIB_KEYS: tuple[str, ...] = (
    "enet_b0_8_best_afew",
    "enet_b0_8_best_vgaf",
    "enet_b0_8_va_mtl",
    "enet_b2_8",
    "mbf_va_mtl",
    "mobilevit_va_mtl",
)
# FER2013 modelləri (7 sinif, ONNX, yalnız CPU) — bax src/model_zoo.py
FER_KEYS: tuple[str, ...] = ("deepface_emotion", "vit_fer2013", "kaggle_fer_cnn")
SUPPORTED_MODELS: tuple[str, ...] = EMOTIEFFLIB_KEYS + FER_KEYS
DEFAULT_BENCHMARK_MODELS: tuple[str, ...] = (
    "enet_b0_8_best_afew",
    "enet_b0_8_va_mtl",
    "enet_b2_8",
    "mbf_va_mtl",
)

DECLINED = "declined"
TIMEOUT = "timeout"

# Geyim yoxlamasının sualları və hər birinin mütləq cavabları (attire.prompts)
ATTIRE_QUESTIONS: dict[str, tuple[str, ...]] = {
    "garment": (),
    "color": ("white",),
    "tie": ("present", "absent"),
    "jacket": ("present", "absent"),
}

_OPERATOR_ID_RE = re.compile(r"^[A-Za-z0-9_-]{2,32}$")


class ConfigError(ValueError):
    """config.yaml-da yanlış dəyər."""


@dataclass
class ModelConfig:
    name: str = "enet_b0_8_best_afew"
    engine: str = "onnx"
    device: str = "cpu"
    flip_tta: bool = False

    @property
    def is_mtl(self) -> bool:
        return "_mtl" in self.name


@dataclass
class CompareConfig:
    mode: str = "off"  # off | auto | on
    models: list[str] = field(default_factory=list)
    budget_ratio: float = 0.5


@dataclass
class DetectorConfig:
    model_path: Path = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
    score_threshold: float = 0.8
    nms_threshold: float = 0.3
    max_input_side: int = 640
    min_face_px: int = 80
    edge_margin_px: int = 4
    max_yaw_ratio: float = 0.35
    max_roll_deg: float = 25.0
    min_region_detail: float = 0.45
    crop_margin: float = 0.1
    align: bool = False


@dataclass
class CaptureConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    analysis_fps: float = 1.5


@dataclass
class SmoothingConfig:
    window_seconds: float = 10.0
    min_frames: int = 3
    uncertain_threshold: float = 0.5


@dataclass
class SelfReportConfig:
    enabled: bool = True
    interval_minutes: float = 10.0
    timeout_seconds: float = 30.0
    categories: dict[str, str] = field(default_factory=dict)
    decline_label: str = "Cavab vermək istəmirəm"


@dataclass
class CalibrationConfig:
    expressions: list[str] = field(
        default_factory=lambda: ["Neutral", "Happiness", "Sadness", "Surprise", "Anger", "Fear", "Disgust"]
    )
    rounds: int = 2
    intro_seconds: float = 4.0
    prepare_seconds: float = 3.0
    hold_seconds: float = 8.0
    rest_seconds: float = 2.0
    shuffle: bool = True


@dataclass
class ResearchConfig:
    save_test_frames: bool = False
    dir: Path = PROJECT_ROOT / "dataset"
    crop_margin: float = 0.6
    jpeg_quality: int = 95


@dataclass
class LoggingConfig:
    dir: Path = PROJECT_ROOT / "logs"


@dataclass
class UIConfig:
    show_prediction: bool = True
    show_preview: bool = False
    preview_fps: float = 8.0
    material: str = "auto"


@dataclass
class PrivacyConfig:
    block_network: bool = True


@dataclass
class VATiredRule:
    enabled: bool = False
    arousal_max: float = -0.3


@dataclass
class ReportConfig:
    dir: Path = PROJECT_ROOT / "reports"
    min_group_size: int = 3
    va_tired_rule: VATiredRule = field(default_factory=VATiredRule)


@dataclass
class BenchmarkConfig:
    frames: int = 60
    warmup: int = 5


@dataclass
class AttireLiveConfig:
    """Paneldə geyim kartı: canlı kameradan bir neçə kadr → səs çoxluğu. Nəticə yalnız ekranda."""

    enabled: bool = False
    on_start: bool = True  # kamera açılandan sonra bir dəfə avtomatik
    frames: int = 5
    interval_s: float = 0.7
    threads: int = 2  # torch nüvələri — emosiya pipeline-ı CPU-suz qalmasın


@dataclass
class AttireConfig:
    """Geyim yoxlaması: oflayn (python -m src.attire) və istəyə görə paneldə (live)."""

    model_dir: Path = PROJECT_ROOT / "models" / "attire" / "siglip2-base-patch16-224"
    # Ölçülər üz qutusuna nisbətəndir (en → üzün eni, hündürlük → üzün hündürlüyü), çənədən aşağı
    torso_width: float = 3.4
    torso_height: float = 2.0
    collar_width: float = 1.6  # rəng ölçülən yaxa-döş zonası
    collar_top: float = 0.15
    collar_height: float = 1.0
    min_visible: float = 0.8  # çənədən aşağı kadra düşən minimum hündürlük
    min_torso_px: int = 112
    min_brightness: float = 40.0  # ÜZÜN orta parlaqlığı, 0–255 (qara paltar qaranlıq sayılmasın)
    min_sharpness: float = 8.0  # üzün Laplas dispersiyası (128 px enə gətirilmiş); 0 → söndür
    # Zero-shot: sual → cavab → mətn təsvirləri (bax ATTIRE_QUESTIONS). Boşdursa, geyim yoxlaması işləmir.
    prompts: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    formal_garments: list[str] = field(default_factory=lambda: ["shirt"])
    # Qərar: rəsmi = (formal_garments cəmi >= formal_threshold) VƏ YA (pencək >= jacket_threshold);
    # require_white → üstəlik ağ olmalıdır
    formal_threshold: float = 0.5
    jacket_is_formal: bool = True
    jacket_threshold: float = 0.5
    jacket_min_layering: float = 0.25  # pencək üçün "iki qat" əlaməti (orta zolaq yanlardan fərqli); 0 → söndür
    require_white: bool = False
    white_threshold: float = 0.4  # SigLIP-in "white" cavabının ehtimalı
    white_min_lightness: float = 60.0  # ağ piksel: L* (0–100) bundan yuxarı…
    white_max_chroma: float = 20.0  # …və rəng doyğunluğu sqrt(a*² + b*²) bundan aşağı
    white_min_share: float = 0.15  # yaxa zonasında ağ piksellərin minimum payı
    color_rule: str = "both"  # both | model | pixels
    live: AttireLiveConfig = field(default_factory=AttireLiveConfig)


@dataclass
class AppConfig:
    operator_id: str
    model: ModelConfig
    compare: CompareConfig
    detector: DetectorConfig
    capture: CaptureConfig
    smoothing: SmoothingConfig
    self_report: SelfReportConfig
    mapping: dict[str, str]
    test: CalibrationConfig
    research: ResearchConfig
    logging: LoggingConfig
    ui: UIConfig
    privacy: PrivacyConfig
    report: ReportConfig
    benchmark: BenchmarkConfig
    attire: AttireConfig = field(default_factory=AttireConfig)


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' bölməsi lüğət (mapping) olmalıdır")
    return value


def _build(cls, data: dict[str, Any], section: str):
    known = set(cls.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"'{section}' bölməsində naməlum açar(lar): {sorted(unknown)}")
    return cls(**data)


def _resolve(path: str | Path, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


def validate_operator_id(value: Any) -> str:
    if not isinstance(value, str) or not _OPERATOR_ID_RE.match(value):
        raise ConfigError(
            "operator_id 2–32 simvol olmalıdır (hərf, rəqəm, '-' və '_'), məsələn 'OP-7F3A'"
        )
    if not any(ch.isdigit() for ch in value):
        raise ConfigError(
            "operator_id ən azı bir rəqəm saxlamalıdır — ad yox, anonim ID istifadə edin"
        )
    return value


def load_config(path: str | Path | None = None) -> AppConfig:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise ConfigError(f"Konfiqurasiya faylı tapılmadı: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError("config.yaml lüğət formatında olmalıdır")
    return parse_config(raw, base_dir=cfg_path.resolve().parent)


def parse_config(raw: dict[str, Any], base_dir: Path = PROJECT_ROOT) -> AppConfig:
    operator_id = validate_operator_id(raw.get("operator_id"))

    model = _build(ModelConfig, _section(raw, "model"), "model")
    compare = _build(CompareConfig, _section(raw, "compare"), "compare")
    compare.models = list(compare.models or [])
    detector = _build(DetectorConfig, _section(raw, "detector"), "detector")
    detector.model_path = _resolve(detector.model_path, base_dir)
    capture = _build(CaptureConfig, _section(raw, "capture"), "capture")
    smoothing = _build(SmoothingConfig, _section(raw, "smoothing"), "smoothing")
    self_report = _build(SelfReportConfig, _section(raw, "self_report"), "self_report")
    test = _build(CalibrationConfig, _section(raw, "test"), "test")
    test.expressions = list(test.expressions)
    research = _build(ResearchConfig, _section(raw, "research"), "research")
    research.dir = _resolve(research.dir, base_dir)
    logging_cfg = _build(LoggingConfig, _section(raw, "logging"), "logging")
    logging_cfg.dir = _resolve(logging_cfg.dir, base_dir)
    ui = _build(UIConfig, _section(raw, "ui"), "ui")
    privacy = _build(PrivacyConfig, _section(raw, "privacy"), "privacy")
    report_raw = dict(_section(raw, "report"))
    va_rule = _build(VATiredRule, report_raw.pop("va_tired_rule", None) or {}, "va_tired_rule")
    report = _build(ReportConfig, report_raw, "report")
    report.dir = _resolve(report.dir, base_dir)
    report.va_tired_rule = va_rule
    benchmark = _build(BenchmarkConfig, _section(raw, "benchmark"), "benchmark")
    attire_raw = dict(_section(raw, "attire"))
    attire_live = _build(AttireLiveConfig, attire_raw.pop("live", None) or {}, "attire.live")
    attire = _build(AttireConfig, attire_raw, "attire")
    attire.model_dir = _resolve(attire.model_dir, base_dir)
    attire.live = attire_live
    mapping = _section(raw, "mapping")

    cfg = AppConfig(
        operator_id=operator_id,
        model=model,
        compare=compare,
        detector=detector,
        capture=capture,
        smoothing=smoothing,
        self_report=self_report,
        mapping=dict(mapping),
        test=test,
        research=research,
        logging=logging_cfg,
        ui=ui,
        privacy=privacy,
        report=report,
        benchmark=benchmark,
        attire=attire,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: AppConfig) -> None:
    m = cfg.model
    if m.name not in SUPPORTED_MODELS:
        raise ConfigError(f"Dəstəklənməyən model '{m.name}'. Seçimlər: {', '.join(SUPPORTED_MODELS)}")
    if m.engine not in ("onnx", "torch"):
        raise ConfigError("model.engine 'onnx' və ya 'torch' olmalıdır")
    if m.device not in ("cpu", "cuda"):
        raise ConfigError("model.device 'cpu' və ya 'cuda' olmalıdır")
    if m.engine == "onnx" and m.device == "cuda":
        raise ConfigError(
            "ONNX engine yalnız CPU-da işləyir (kitabxana CPUExecutionProvider-ə bağlıdır). "
            "GPU üçün engine: torch, device: cuda seçin"
        )
    if m.name in FER_KEYS and (m.engine, m.device) != ("onnx", "cpu"):
        raise ConfigError(f"{m.name} yalnız engine: onnx, device: cpu ilə işləyir")

    c = cfg.compare
    if c.mode not in ("off", "auto", "on"):
        raise ConfigError("compare.mode: off | auto | on")
    unknown = [k for k in c.models if k not in SUPPORTED_MODELS]
    if unknown:
        raise ConfigError(f"compare.models-də naməlum model(lər): {unknown}")
    if len(set(c.models)) != len(c.models):
        raise ConfigError("compare.models-də təkrar model var")
    if not 0.05 <= c.budget_ratio <= 1.0:
        raise ConfigError("compare.budget_ratio 0.05–1 aralığında olmalıdır")

    d = cfg.detector
    if not 0.0 < d.score_threshold <= 1.0:
        raise ConfigError("detector.score_threshold (0, 1] aralığında olmalıdır")
    if d.max_input_side < 160 or d.min_face_px < 16:
        raise ConfigError("detector.max_input_side >= 160 və min_face_px >= 16 olmalıdır")
    if not 0.0 <= d.crop_margin <= 0.5:
        raise ConfigError("detector.crop_margin [0, 0.5] aralığında olmalıdır")
    if not 0.0 <= d.min_region_detail <= 1.5:
        raise ConfigError("detector.min_region_detail [0, 1.5] aralığında olmalıdır (0 = söndür)")

    if not 0.2 <= cfg.capture.analysis_fps <= 2.0:
        raise ConfigError("capture.analysis_fps 0.2–2 aralığında olmalıdır (30 FPS emal etmirik)")

    s = cfg.smoothing
    if s.window_seconds <= 0 or s.min_frames < 1:
        raise ConfigError("smoothing.window_seconds > 0 və min_frames >= 1 olmalıdır")
    if not 0.0 < s.uncertain_threshold < 1.0:
        raise ConfigError("smoothing.uncertain_threshold (0, 1) aralığında olmalıdır")

    sr = cfg.self_report
    if not sr.categories:
        raise ConfigError("self_report.categories boş ola bilməz")
    reserved = {DECLINED, TIMEOUT}
    if reserved & set(sr.categories):
        raise ConfigError(f"self_report.categories kodları {sorted(reserved)} ola bilməz")
    if sr.interval_minutes <= 0 or sr.timeout_seconds <= 0:
        raise ConfigError("self_report.interval_minutes və timeout_seconds müsbət olmalıdır")

    missing = set(EMOTION_CLASSES) - set(cfg.mapping)
    extra = set(cfg.mapping) - set(EMOTION_CLASSES)
    if missing or extra:
        raise ConfigError(
            f"mapping 8 sinfin hamısını əhatə etməlidir. Çatışmır: {sorted(missing)}, "
            f"artıq: {sorted(extra)}"
        )
    bad_targets = {v for v in cfg.mapping.values() if v not in sr.categories}
    if bad_targets:
        raise ConfigError(f"mapping-də naməlum kateqoriya kodları: {sorted(bad_targets)}")

    t = cfg.test
    bad = [e for e in t.expressions if e not in EMOTION_CLASSES]
    if bad or not t.expressions:
        raise ConfigError(f"test.expressions yalnız {list(EMOTION_CLASSES)} ola bilər, yanlış: {bad}")
    if len(set(t.expressions)) != len(t.expressions):
        raise ConfigError("test.expressions-da təkrar ifadə var")
    if t.rounds < 1 or t.hold_seconds < 2 or min(t.intro_seconds, t.prepare_seconds, t.rest_seconds) < 0:
        raise ConfigError("test: rounds >= 1, hold_seconds >= 2, digər müddətlər >= 0 olmalıdır")

    r = cfg.research
    if not 0.2 <= r.crop_margin <= 1.5:
        raise ConfigError("research.crop_margin 0.2–1.5 aralığında olmalıdır (sonradan yenidən kəsmək üçün)")
    if not 50 <= r.jpeg_quality <= 100:
        raise ConfigError("research.jpeg_quality 50–100 aralığında olmalıdır")

    if not 0 < cfg.ui.preview_fps <= 15:
        raise ConfigError("ui.preview_fps (0, 15] aralığında olmalıdır")
    if cfg.ui.material not in ("auto", "mica", "acrylic", "none"):
        raise ConfigError("ui.material: auto | mica | acrylic | none")
    if cfg.report.min_group_size < 1:
        raise ConfigError("report.min_group_size >= 1 olmalıdır")
    if cfg.benchmark.frames < 1 or cfg.benchmark.warmup < 0:
        raise ConfigError("benchmark.frames >= 1 və warmup >= 0 olmalıdır")

    a = cfg.attire
    if min(a.torso_width, a.torso_height, a.collar_width, a.collar_height) <= 0 or a.collar_top < 0:
        raise ConfigError("attire: kəsik ölçüləri müsbət olmalıdır (collar_top >= 0)")
    if a.collar_width > a.torso_width or a.collar_top + a.collar_height > a.torso_height:
        raise ConfigError("attire: yaxa zonası gövdə kəsiyinin içində olmalıdır")
    if not 0 < a.min_visible <= a.torso_height:
        raise ConfigError("attire.min_visible (0, torso_height] aralığında olmalıdır")
    if a.min_torso_px < 32 or not 0 <= a.min_brightness < 255 or a.min_sharpness < 0:
        raise ConfigError("attire: min_torso_px >= 32, min_brightness [0, 255), min_sharpness >= 0 olmalıdır")
    if not all(0 < t < 1 for t in (a.formal_threshold, a.jacket_threshold, a.white_threshold)) \
            or not 0 <= a.white_min_share <= 1:
        raise ConfigError("attire: formal_/jacket_/white_threshold (0, 1), white_min_share [0, 1] olmalıdır")
    if not 0 <= a.white_min_lightness <= 100 or a.white_max_chroma <= 0:
        raise ConfigError("attire: white_min_lightness [0, 100], white_max_chroma > 0 olmalıdır")
    if not 0 <= a.jacket_min_layering < 1:
        raise ConfigError("attire.jacket_min_layering [0, 1) aralığında olmalıdır (0 = söndür)")
    if a.color_rule not in ("both", "model", "pixels"):
        raise ConfigError("attire.color_rule: both | model | pixels")
    if a.prompts:
        _validate_attire_prompts(a)
    lv = a.live
    if not (1 <= lv.frames <= 15 and 0.1 <= lv.interval_s <= 5 and 1 <= lv.threads <= 16):
        raise ConfigError("attire.live: frames 1–15, interval_s 0.1–5, threads 1–16 olmalıdır")
    if lv.enabled and not a.prompts:
        raise ConfigError("attire.live.enabled üçün attire.prompts lazımdır")


def _validate_attire_prompts(a: AttireConfig) -> None:
    if set(a.prompts) != set(ATTIRE_QUESTIONS):
        raise ConfigError(f"attire.prompts yalnız bu sualları saxlamalıdır: {list(ATTIRE_QUESTIONS)}")
    for question, required in ATTIRE_QUESTIONS.items():
        answers = a.prompts[question]
        if not isinstance(answers, dict) or len(answers) < 2:
            raise ConfigError(f"attire.prompts.{question}: ən azı iki cavab olmalıdır")
        missing = [r for r in required if r not in answers]
        if missing:
            raise ConfigError(f"attire.prompts.{question}: {missing} cavabı mütləqdir")
        for label, texts in answers.items():
            if not isinstance(label, str):
                raise ConfigError(f"attire.prompts.{question}: cavab adı mətn olmalıdır ({label!r}) — "
                                  "YAML-da yes/no/on/off dırnaqsız yazılsa, true/false olur")
            if not isinstance(texts, list) or not texts or not all(isinstance(t, str) and t.strip() for t in texts):
                raise ConfigError(f"attire.prompts.{question}.{label}: boş olmayan mətnlər siyahısı olmalıdır")
    unknown = [g for g in a.formal_garments if g not in a.prompts["garment"]]
    if not a.formal_garments or unknown:
        raise ConfigError(f"attire.formal_garments attire.prompts.garment cavablarından olmalıdır, yanlış: {unknown}")
