"""Geyim yoxlaması: gövdə kəsiyi, keyfiyyət filtri, config və əsas tətbiqdən ayrılıq."""

import subprocess
import sys

import numpy as np
import pytest

from src.attire.crop import (Torso, TorsoStatus, collar_box, extract_torso, find_torso, layering, letterbox,
                             torso_box)
from src.attire.encoder import Question, ZeroShot, build_question
from src.attire.verdict import FORMAL, NOT_FORMAL, UNKNOWN, decide, split_scores, white_share
from src.config import PROJECT_ROOT, AttireConfig, ConfigError

FACE = (600.0, 150.0, 80.0, 100.0)  # x, y, w, h — çənə y=250-də
SHAPE = (720, 1280, 3)


def textured(shape, base=200, seed=0, sd=25):
    rng = np.random.default_rng(seed)
    return np.clip(np.asarray(base) + rng.normal(0, sd, shape), 0, 255).astype(np.uint8)


def paint_face(frame, face=FACE, lit=True):
    """Üz: B=0, R=255 (və ya qaranlıqda 20) — kəsikdə tanınan imza; G təsadüfi — kəskinlik üçün tekstura."""
    x, y, w, h = (int(v) for v in face)
    region = frame[y : y + h, x : x + w]
    region[..., 0] = 0
    region[..., 1] = np.random.default_rng(1).integers(0, 255 if lit else 40, region.shape[:2])
    region[..., 2] = 255 if lit else 20
    return frame


def frame_with_face(face=FACE, shape=SHAPE, base=200, lit=True):
    return paint_face(textured(shape, base), face, lit)


def is_face_pixel(img):
    return (img[..., 0] == 0) & (img[..., 2] == 255)


# --- həndəsə -------------------------------------------------------------------------


def test_torso_box_starts_at_chin_and_scales_with_face():
    cfg = AttireConfig()
    (x0, y0, x1, y1), visible = torso_box(FACE, SHAPE, cfg)
    assert y0 == 250  # çənə
    assert (x0 + x1) / 2 == pytest.approx(640, abs=1)  # üzün mərkəzi
    assert x1 - x0 == pytest.approx(cfg.torso_width * 80, abs=2)
    assert y1 - y0 == pytest.approx(cfg.torso_height * 100, abs=2)
    assert visible == pytest.approx(cfg.torso_height)


def test_torso_box_is_clipped_to_frame():
    (x0, y0, x1, y1), visible = torso_box((10.0, 600.0, 80.0, 100.0), SHAPE, AttireConfig())
    assert x0 == 0 and y1 == 720
    assert visible == pytest.approx(0.2)


def test_collar_box_is_inside_torso_box():
    cfg = AttireConfig()
    tx0, ty0, tx1, ty1 = torso_box(FACE, SHAPE, cfg)[0]
    cx0, cy0, cx1, cy1 = collar_box(FACE, SHAPE, cfg)
    assert tx0 <= cx0 < cx1 <= tx1 and ty0 < cy0 < cy1 <= ty1


def test_letterbox_makes_square_without_resizing():
    img = np.zeros((50, 120, 3), np.uint8)
    out = letterbox(img)
    assert out.shape == (120, 120, 3)
    assert (out[:35] == 128).all() and (out[35:85] == 0).all()


# --- keyfiyyət filtri ------------------------------------------------------------------


def test_ok_crop_is_square_and_excludes_face():
    t = extract_torso(frame_with_face(), FACE, AttireConfig())
    assert t.ok, t.reason
    assert t.crop.shape[0] == t.crop.shape[1]
    assert not is_face_pixel(t.crop).any()  # üz pikselləri kəsiyə düşmür
    assert t.collar is not None and t.collar.size > 0


def test_torso_hidden_when_camera_shows_only_head():
    face = (600.0, 560.0, 80.0, 100.0)  # çənə y=660, aşağıda 60 px (0.6 üz)
    t = extract_torso(frame_with_face(face), face, AttireConfig())
    assert t.status is TorsoStatus.TORSO_HIDDEN
    assert t.crop is None and "gövdə" in t.reason


def test_too_small_when_face_is_far():
    face = (600.0, 150.0, 20.0, 25.0)  # gövdə eni 68 px
    t = extract_torso(frame_with_face(face), face, AttireConfig())
    assert t.status is TorsoStatus.TOO_SMALL


def test_dark_room_is_judged_by_the_face():
    t = extract_torso(frame_with_face(base=15, lit=False), FACE, AttireConfig())
    assert t.status is TorsoStatus.DARK and "işıq" in t.reason


def test_black_shirt_is_not_too_dark_or_blurry():
    """Qara, düz parça gövdəni qaranlıq/teksturasız edir — keyfiyyət üzdə ölçüldüyü üçün keçməlidir."""
    frame = np.full(SHAPE, (24, 24, 26), np.uint8)
    t = extract_torso(paint_face(frame), FACE, AttireConfig())
    assert t.ok, t.reason


def test_blurry_face_and_switch_off():
    frame = np.full(SHAPE, 180, np.uint8)  # teksturasız üz → kəskinlik 0
    assert extract_torso(frame, FACE, AttireConfig()).status is TorsoStatus.BLURRY
    assert extract_torso(frame, FACE, AttireConfig(min_sharpness=0)).ok


# --- pencək: "iki qat" əlaməti -------------------------------------------------------------

LFACE = (590.0, 150.0, 100.0, 120.0)  # çənə y=270, mərkəz x=640


def side_lit(frame):
    """Yan işıq: sol tərəf açıq, sağ tünd — eyni parça olsa da yanlar fərqli görünür."""
    return np.clip(frame * np.linspace(1.25, 0.8, frame.shape[1])[None, :, None], 0, 255).astype(np.uint8)


def jacket_frame(jacket_bgr=(60, 40, 30), shirt_bgr=(235, 235, 235), tie=True):
    frame = textured(SHAPE, jacket_bgr, sd=6)
    for yy in range(270, SHAPE[0]):  # V-şəkilli açıqlıq: köynək, ortada qalstuk
        half = max(0, int(45 - (yy - 270) * 0.25))
        frame[yy, 640 - half : 640 + half] = shirt_bgr
        if tie:
            frame[yy, 630:650] = (40, 40, 160)
    return frame


@pytest.mark.parametrize("make, low, high", [
    (lambda: side_lit(textured(SHAPE, (25, 25, 28), sd=6)), 0.0, 0.1),  # qara köynək
    (lambda: side_lit(textured(SHAPE, (235, 235, 235), sd=6)), 0.0, 0.1),  # ağ köynək
    (lambda: jacket_frame(), 0.5, 1.0),  # tünd pencək + ağ köynək + qalstuk
    (lambda: jacket_frame((120, 120, 120), (20, 20, 20), tie=False), 0.5, 1.0),  # boz blazer + qara köynək
])
def test_layering_separates_jacket_from_plain_shirt(make, low, high):
    assert low <= layering(make(), LFACE) <= high


def test_layering_averages_out_small_checks():
    frame = textured(SHAPE, (200, 200, 200), sd=6)
    for i in range(0, SHAPE[1], 24):
        frame[:, i : i + 12] = (60, 60, 170)
    for j in range(0, SHAPE[0], 24):
        frame[j : j + 12, :] = (60, 60, 170)
    assert layering(frame, LFACE) < AttireConfig().jacket_min_layering


def test_layering_unknown_when_sides_are_out_of_frame():
    assert layering(jacket_frame(), (5.0, 150.0, 100.0, 120.0)) is None
    assert layering(jacket_frame(), (590.0, 560.0, 100.0, 120.0)) is None  # çənədən aşağı az yer


def test_find_torso_without_face(cfg, yunet_path):
    from src.detector import FaceDetector

    det = FaceDetector(cfg.detector)
    assert find_torso(textured(SHAPE, 120), det, cfg.attire).status is TorsoStatus.NO_FACE
    assert find_torso(np.zeros(SHAPE, np.uint8), det, cfg.attire).status is TorsoStatus.DARK


def test_find_torso_on_fixture(cfg, yunet_path, face_bgr):
    from src.detector import FaceDetector

    det = FaceDetector(cfg.detector)
    face = det.detect(face_bgr).face
    t = find_torso(face_bgr, det, cfg.attire)
    assert t.ok, t.reason
    assert t.box[1] >= face.y + face.h - 1  # kəsik çənədən aşağıdadır


# --- zero-shot -------------------------------------------------------------------------


class FakeEncoder:
    """Hər mətn öz açar sözünə uyğun ox üzrə vektor alır: 'white' → e0, 'dark' → e1, qalanı → e2."""

    logit_scale = 10.0
    AXES = ("white", "dark")

    def encode_texts(self, texts):
        out = np.zeros((len(texts), 3), np.float32)
        for i, t in enumerate(texts):
            axis = next((k for k, word in enumerate(self.AXES) if word in t), 2)
            out[i, axis] = 1.0
        return out


def test_question_probs_softmax_over_answers():
    q = Question("color", ["white", "dark"], np.eye(2, dtype=np.float32))
    p = q.probs(np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], np.float32), logit_scale=10.0)
    assert p.shape == (3, 2)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p[0, 0] > 0.99 and p[1, 1] > 0.99 and p[2, 0] == pytest.approx(0.5)


def test_build_question_averages_prompts_per_answer():
    q = build_question("color", {"white": ["white shirt", "other thing"], "dark": ["dark shirt"]},
                       FakeEncoder().encode_texts)
    assert q.labels == ["white", "dark"]
    assert np.allclose(np.linalg.norm(q.embeddings, axis=1), 1.0)
    assert np.allclose(q.embeddings[0], np.array([1, 0, 1]) / np.sqrt(2))  # iki təsvirin ortası
    assert np.allclose(q.embeddings[1], [0, 1, 0])


def test_zero_shot_scores_every_question():
    zs = ZeroShot(FakeEncoder(), {"color": {"white": ["white"], "dark": ["dark"]}, "tie": {"present": ["tie"],
                                                                                           "absent": ["dark"]}})
    res = zs(np.array([[1.0, 0.0, 0.0]], np.float32))
    assert set(res) == {"color", "tie"}
    assert res["color"][0, 0] > 0.99


@pytest.fixture(scope="session")
def siglip(cfg):
    if not (cfg.attire.model_dir / "model.safetensors").is_file():
        pytest.skip("SigLIP 2 yoxdur — scripts/download_models.py --attire")
    from src.attire.encoder import SiglipEncoder

    enc = SiglipEncoder(cfg.attire.model_dir)
    return enc, ZeroShot(enc, cfg.attire.prompts)


@pytest.mark.models
def test_siglip_white_vs_dark(siglip):
    enc, zs = siglip
    emb = enc.encode_images([np.full((224, 224, 3), 240, np.uint8), np.full((224, 224, 3), 30, np.uint8)])
    assert emb.shape == (2, 768) and np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)
    white = zs(emb)["color"][:, zs.questions["color"].labels.index("white")]
    assert white[0] > 0.4 > 0.1 > white[1]


@pytest.mark.models
def test_siglip_spacesuit_is_not_white_shirt(siglip, cfg, yunet_path, face_bgr):
    from src.detector import FaceDetector

    enc, zs = siglip
    torso = find_torso(face_bgr, FaceDetector(cfg.detector), cfg.attire)
    res = zs(enc.encode_images([torso.crop]))
    garment = zs.questions["garment"].labels[int(res["garment"][0].argmax())]
    assert garment not in cfg.attire.formal_garments
    assert res["color"][0, zs.questions["color"].labels.index("white")] < 0.2


# --- ağlıq və qərar ------------------------------------------------------------------------


def patch(bgr, shape=(60, 80)):
    return np.full((*shape, 3), bgr, np.uint8)


@pytest.mark.parametrize("bgr, expected", [
    ((255, 255, 255), 1.0),
    ((215, 240, 255), 1.0),  # isti lampa altında ağ — sarımtıl, amma az doyğun
    ((128, 128, 128), 0.0),  # boz — kifayət qədər açıq deyil
    ((230, 190, 150), 0.0),  # açıq mavi — doyğun
    ((30, 30, 30), 0.0),
])
def test_white_share_by_color(bgr, expected):
    assert white_share(patch(bgr), 60, 20) == pytest.approx(expected)


def test_white_share_with_dark_tie_in_the_middle():
    region = patch((250, 250, 250))
    region[:, 28:52] = (40, 20, 20)  # qalstuk zonanın 30%-i
    assert white_share(region, 60, 20) == pytest.approx(0.7)
    assert white_share(None, 60, 20) == 0.0


def scores(garment="shirt", p_garment=0.9, color="white", p_color=0.8, tie=0.7, jacket=0.2):
    garments = ["shirt", "tshirt", "polo", "knitwear", "other"]
    colors = ["white", "light_blue", "beige", "dark"]
    g = {k: (p_garment if k == garment else (1 - p_garment) / (len(garments) - 1)) for k in garments}
    c = {k: (p_color if k == color else (1 - p_color) / (len(colors) - 1)) for k in colors}
    return {"garment": g, "color": c, "tie": {"present": tie, "absent": 1 - tie},
            "jacket": {"present": jacket, "absent": 1 - jacket}}


def ok_torso(collar_bgr=(250, 250, 250), layering=None):
    return Torso(TorsoStatus.OK, crop=patch(collar_bgr, (224, 224)), collar=patch(collar_bgr), layering=layering)


def test_shirt_is_formal_and_all_scores_are_reported():
    v = decide(ok_torso(), scores(), AttireConfig())
    assert v.result == FORMAL and v.reason == "" and v.text == "Rəsmi" and v.basis == "düyməli köynək"
    assert (v.garment, v.color, v.tie_p, v.jacket_p) == ("shirt", "white", 0.7, 0.2)
    assert v.white_share == pytest.approx(1.0)
    assert set(v.garment_probs) == {"shirt", "tshirt", "polo", "knitwear", "other"} and v.color_probs["white"] == 0.8


@pytest.mark.parametrize("bgr, color", [((128, 128, 128), "grey"), ((230, 190, 150), "light_blue"),
                                        ((200, 200, 200), "patterned")])
def test_shirt_of_any_color_is_formal(bgr, color):
    v = decide(ok_torso(bgr), scores(color=color), AttireConfig())
    assert v.result == FORMAL and v.reason == ""


def test_only_white_when_required():
    v = decide(ok_torso((230, 190, 150)), scores(color="light_blue"), AttireConfig(require_white=True))
    assert v.result == NOT_FORMAL
    assert "açıq mavi" in v.reason and "ağ az" in v.reason
    assert v.model_white is False and v.pixel_white is False


def test_jacket_counts_as_formal_unless_disabled():
    s = scores(garment="tshirt", jacket=0.8)
    v = decide(ok_torso(), s, AttireConfig())
    assert v.result == FORMAL and v.basis == "pencək" and v.jacket_ok and not v.garment_ok
    assert decide(ok_torso(), s, AttireConfig(jacket_is_formal=False)).result == NOT_FORMAL
    both = decide(ok_torso(), scores(jacket=0.9), AttireConfig())
    assert both.basis == "düyməli köynək + pencək"


def test_black_shirt_is_not_a_jacket_without_two_layers():
    """Model qara köynəyə "pencək" deyir (0.9), amma ortası və yanları eyni parçadır."""
    v = decide(ok_torso((25, 25, 28), layering=0.02), scores(color="dark", jacket=0.9), AttireConfig())
    assert v.result == FORMAL and v.basis == "düyməli köynək"  # köynək özü rəsmidir
    assert not v.jacket_seen and v.layering == 0.02
    # Qara futbolka + model "pencək" deyir → artıq rəsmi sayılmır
    t = decide(ok_torso((25, 25, 28), layering=0.02), scores(garment="tshirt", color="dark", jacket=0.9),
               AttireConfig())
    assert t.result == NOT_FORMAL and not t.jacket_ok
    # İki qat görünür → pencək
    assert decide(ok_torso(layering=0.7), scores(garment="tshirt", jacket=0.9), AttireConfig()).basis == "pencək"
    # Əlamət söndürülüb → yalnız modelə etibar
    off = AttireConfig(jacket_min_layering=0)
    assert decide(ok_torso(layering=0.02), scores(garment="tshirt", jacket=0.9), off).result == FORMAL


def test_tie_does_not_change_the_result():
    assert decide(ok_torso(), scores(tie=0.0), AttireConfig()).result == FORMAL
    assert decide(ok_torso(), scores(garment="tshirt", tie=1.0), AttireConfig()).result == NOT_FORMAL


def test_tshirt_is_not_formal():
    v = decide(ok_torso(), scores(garment="tshirt"), AttireConfig())
    assert v.result == NOT_FORMAL and not v.garment_ok and not v.jacket_ok
    assert v.reason == "düyməli köynək və ya pencək görünmür — futbolka" and v.basis == ""


def test_uncertain_shirt_reason():
    v = decide(ok_torso(), scores(p_garment=0.45), AttireConfig())
    assert v.result == NOT_FORMAL and "dəqiq deyil (45%)" in v.reason


@pytest.mark.parametrize("rule, expected", [("both", NOT_FORMAL), ("model", FORMAL), ("pixels", NOT_FORMAL)])
def test_color_rule_when_signals_disagree(rule, expected):
    # Model "ağ" deyir, yaxa zonası isə boz görünür (məs. qaranlıq otaq)
    v = decide(ok_torso((128, 128, 128)), scores(), AttireConfig(require_white=True, color_rule=rule))
    assert v.result == expected and v.model_white and not v.pixel_white


def test_white_polo_counts_when_configured():
    s = scores(garment="polo")
    assert decide(ok_torso(), s, AttireConfig()).result == NOT_FORMAL
    assert decide(ok_torso(), s, AttireConfig(formal_garments=["shirt", "polo"])).result == FORMAL


def test_unknown_when_torso_is_not_usable():
    v = decide(Torso(TorsoStatus.TORSO_HIDDEN), None, AttireConfig())
    assert v.result == UNKNOWN and v.torso == "torso_hidden" and "gövdə" in v.reason
    assert v.formal_p is None


def test_split_scores_per_image():
    probs = {"color": np.array([[0.8, 0.2], [0.1, 0.9]]), "tie": np.array([[0.3, 0.7], [0.6, 0.4]])}
    rows = split_scores(probs, {"color": ["white", "dark"], "tie": ["present", "absent"]})
    assert rows[1] == {"color": {"white": 0.1, "dark": 0.9}, "tie": {"present": 0.6, "absent": 0.4}}


# --- qiymətləndirmə və CLI ----------------------------------------------------------------

from src.attire import evaluate as ev  # noqa: E402
from src.detector import Detection, Face, FaceStatus  # noqa: E402


def save_png(path, image):
    """Testin GİRİŞ şəkli (unicode yolla da)."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(buf.tobytes())


def test_discover_labeled_folder_with_unicode_names(tmp_path):
    img = textured((40, 40, 3))
    save_png(tmp_path / "rəsmi" / "P01_1.png", img)
    save_png(tmp_path / "rəsmi" / "alt" / "P02_1.jpg", img)
    save_png(tmp_path / "qeyri-resmi" / "P01_2.png", img)
    (tmp_path / "qeyri-resmi" / "qeyd.txt").write_text("x", encoding="utf-8")
    save_png(tmp_path / "digər" / "x.png", img)  # etiket qovluğu deyil — nəzərə alınmır
    items, labeled = ev.discover(tmp_path)
    assert labeled
    assert sorted((i.rel, i.label, i.group) for i in items) == [
        ("qeyri-resmi/P01_2.png", NOT_FORMAL, "P01"),
        ("rəsmi/P01_1.png", FORMAL, "P01"),
        ("rəsmi/alt/P02_1.jpg", FORMAL, "P02"),
    ]
    assert ev.read_image(items[0].path).shape == (40, 40, 3)


def test_discover_unlabeled_folder(tmp_path):
    save_png(tmp_path / "a.png", textured((10, 10, 3)))
    save_png(tmp_path / "gün" / "b.png", textured((10, 10, 3)))
    items, labeled = ev.discover(tmp_path)
    assert not labeled and [i.label for i in items] == [None, None]
    assert {i.group for i in items} == {"a.png", "gün/b.png"}  # "_" yoxdur — hər şəkil öz qrupu


def test_read_image_handles_broken_files(tmp_path):
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    (tmp_path / "empty.jpg").write_bytes(b"")
    assert ev.read_image(tmp_path / "broken.jpg") is None
    assert ev.read_image(tmp_path / "empty.jpg") is None
    assert ev.read_image(tmp_path / "missing.jpg") is None


def test_metrics_balanced_accuracy():
    m = ev.metrics([FORMAL, FORMAL, FORMAL, NOT_FORMAL], [FORMAL, FORMAL, NOT_FORMAL, NOT_FORMAL])
    assert m["acc"] == 0.75 and m["balanced"] == pytest.approx((2 / 3 + 1) / 2)
    assert m["cm"][(FORMAL, NOT_FORMAL)] == 1 and m["ci"][0] < 0.75 < m["ci"][1]


def make_row(lab, formal_p, white_p, share, emb=None, group="g", jacket_p=0.0, cfg=None):
    from src.attire.verdict import Verdict

    v = Verdict(NOT_FORMAL, formal_p=formal_p, white_p=white_p, white_share=share, jacket_p=jacket_p)
    v.result = ev.rule_predict(v, cfg or AttireConfig())
    return ev.Row(ev.Item(None, f"{group}.png", lab, group), v, emb, 1.0)


def test_calibrate_finds_better_thresholds():
    # Rəsmi köynəklərdə ehtimal 0.42 civarındadır — indiki 0.5 həddi onları rədd edir
    rows = [make_row(FORMAL, 0.42, 0.3, 0.6) for _ in range(5)] + [make_row(NOT_FORMAL, 0.1, 0.1, 0.6) for _ in range(5)]
    assert ev.metrics([r.item.label for r in rows], [r.verdict.result for r in rows])["balanced"] == 0.5
    best = ev.calibrate(rows, AttireConfig())
    assert best["metrics"]["balanced"] == 1.0
    assert best["params"]["formal_threshold"] <= 0.4
    assert "white_threshold" not in best["params"]  # ağ tələb olunmur — rəng hədləri sınanmır


def test_calibrate_white_thresholds_when_white_required():
    cfg = AttireConfig(require_white=True)
    # Rəsmi şəkillərdə "white" ehtimalı 0.3 civarındadır (isti işıq) — indiki 0.4 həddi onları rədd edir
    rows = [make_row(FORMAL, 0.9, 0.3, 0.6, cfg=cfg) for _ in range(5)] + \
        [make_row(NOT_FORMAL, 0.9, 0.1, 0.6, cfg=cfg) for _ in range(5)]
    best = ev.calibrate(rows, cfg)
    assert best["metrics"]["balanced"] == 1.0
    assert 0.1 < best["params"]["white_threshold"] <= 0.3


def test_calibrate_can_switch_jacket_off():
    # Qeyri-rəsmi şəkillərdə model səhvən "pencək" görür — ən yaxşı ayar pencəyi söndürmək və ya həddi qaldırmaqdır
    rows = [make_row(FORMAL, 0.9, 0.5, 0.5, jacket_p=0.2) for _ in range(5)] + \
        [make_row(NOT_FORMAL, 0.1, 0.5, 0.5, jacket_p=0.75) for _ in range(5)]
    best = ev.calibrate(rows, AttireConfig())
    assert best["metrics"]["balanced"] == 1.0
    p = best["params"]
    assert p["jacket_is_formal"] is False or p["jacket_threshold"] > 0.75
    assert "jacket_is_formal false" in ev.format_params({"jacket_is_formal": False})


def test_calibrate_needs_both_classes():
    assert ev.calibrate([make_row(FORMAL, 0.9, 0.9, 0.9)], AttireConfig()) is None


def test_group_folds_keep_a_person_in_one_fold():
    groups = [f"P{i // 3}" for i in range(30)]  # 10 şəxs × 3 şəkil
    labels = [FORMAL if i % 2 else NOT_FORMAL for i in range(30)]
    folds = ev.group_folds(groups, labels, 5)
    assert len(folds) == 5 and sorted(np.concatenate(folds).tolist()) == list(range(30))
    for f in folds:
        others = set().union(*(set(groups[i] for i in g) for g in folds if g is not f))
        assert not {groups[i] for i in f} & others


def test_cross_validate_learns_separable_embeddings():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(80):
        lab = FORMAL if i % 2 else NOT_FORMAL
        emb = rng.normal(0, 1, 32) + (2.0 if lab == FORMAL else -2.0) * np.eye(32)[0]
        # zero-shot hamısını "qeyri-rəsmi" deyir — öyrədilmiş təsnifatçı fərqi görməlidir
        rows.append(make_row(lab, 0.1, 0.1, 0.0, emb, group=f"P{i // 2}"))
    cv = ev.cross_validate(rows, 5)
    assert cv["folds"] == 5 and cv["groups"] == 40
    assert cv["metrics"]["balanced"] >= 0.85 and cv["zero_shot"]["balanced"] == 0.5


def test_cross_validate_skips_small_data():
    rows = [make_row(FORMAL, 0.9, 0.9, 0.9, np.ones(4)), make_row(NOT_FORMAL, 0.1, 0.1, 0.0, np.zeros(4))]
    assert ev.cross_validate(rows, 5) is None
    assert ev.cross_validate(rows * 10, 0) is None


class FakeDetector:
    def detect(self, frame):
        return Detection(FaceStatus.OK, Face(270.0, 60.0, 100.0, 120.0, 0.9, np.zeros((5, 2), np.float32)), 1)


class FakeImageEncoder(FakeEncoder):
    def encode_images(self, images):
        m = np.array([img.mean() / 255.0 for img in images])
        v = np.stack([m, 1.0 - m, np.full_like(m, 0.1)], axis=1)
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


FAKE_PROMPTS = {
    "garment": {"shirt": ["white shirt"], "tshirt": ["dark tshirt"]},
    "color": {"white": ["white"], "dark": ["dark"]},
    "tie": {"present": ["white tie"], "absent": ["dark"]},
    "jacket": {"present": ["jacket"], "absent": ["dark"]},
}


def shirt_frame(bgr, seed=0):
    frame = textured((480, 640, 3), 128, seed)
    rng = np.random.default_rng(seed)
    frame[180:] = np.clip(np.array(bgr) + rng.normal(0, 8, (300, 640, 3)), 0, 255).astype(np.uint8)
    return frame


@pytest.fixture
def fake_cli(monkeypatch, make_config):
    cfg = make_config(attire__prompts=FAKE_PROMPTS)
    monkeypatch.setattr(ev, "load_config", lambda path=None: cfg)
    encoder = FakeImageEncoder()
    monkeypatch.setattr(ev, "load_models", lambda c: ev.Models(FakeDetector(), encoder, ZeroShot(encoder, c.attire.prompts)))
    return cfg


def test_cli_end_to_end_with_fake_models(tmp_path, fake_cli, capsys):
    root = tmp_path / "geyim_test"
    for i in range(6):
        save_png(root / "resmi" / f"P{i:02d}_ag.png", shirt_frame((245, 245, 245), i))
        save_png(root / "qeyri-resmi" / f"P{i:02d}_tund.png", shirt_frame((120, 70, 40), i))
    (root / "resmi" / "P99_broken.jpg").write_bytes(b"xx")
    out = tmp_path / "out" / "geyim.csv"

    assert ev.main([str(root), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "Qiymətləndirilə bildi: 12 / 13" in text and "şəkil oxunmadı: 1" in text
    assert "Dəqiqlik 100%" in text and "Ən yaxşı ayarlar" in text and "Öyrədilmiş təsnifatçı" in text
    assert "düyməli köynək 100% · pencək" in text  # rəsmi şəkillərdə siqnalların keçmə faizi

    import csv

    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == ev.CSV_COLUMNS and len(rows) == 13
    by_file = {r["file"]: r for r in rows}
    assert by_file["resmi/P00_ag.png"]["result"] == FORMAL
    assert by_file["qeyri-resmi/P00_tund.png"]["result"] == NOT_FORMAL
    assert by_file["resmi/P99_broken.jpg"]["torso"] == "unreadable"
    assert not any(str(tmp_path) in r["file"] for r in rows)  # yalnız nisbi yol


def test_cli_unlabeled_and_errors(tmp_path, fake_cli, capsys):
    save_png(tmp_path / "kadrlar" / "a.png", shirt_frame((245, 245, 245)))
    assert ev.main([str(tmp_path / "kadrlar"), "--out", str(tmp_path / "x.csv")]) == 0
    text = capsys.readouterr().out
    assert "Rəsmi: 1 · Qeyri-rəsmi: 0" in text and "Dəqiqlik" not in text
    assert ev.main([str(tmp_path / "yoxdur")]) == 2
    (tmp_path / "bos").mkdir()
    assert ev.main([str(tmp_path / "bos")]) == 1


# --- canlı yoxlama (panel kartı) ------------------------------------------------------------

from src.attire import live  # noqa: E402
from src.attire.verdict import Verdict  # noqa: E402


def V(result, reason="", torso="ok", **kw):
    base = dict(garment="shirt", formal_p=0.9, color="white", white_p=0.8, white_share=0.6, tie_p=0.7, jacket_p=0.1)
    if result == UNKNOWN:
        return Verdict(UNKNOWN, reason, torso)
    return Verdict(result, reason, torso, **{**base, **kw})


def test_aggregate_majority_and_means():
    v, votes = live.aggregate([V(FORMAL), V(FORMAL), V(NOT_FORMAL, "ağ deyil", formal_p=0.6), V(FORMAL),
                               V(UNKNOWN, "x", "dark")], 5)
    assert (v.result, votes) == (FORMAL, 3)
    assert v.formal_p == pytest.approx((0.9 * 3 + 0.6) / 4) and v.garment == "shirt"


def test_aggregate_tie_is_not_formal():
    v, votes = live.aggregate([V(FORMAL), V(FORMAL), V(NOT_FORMAL, "futbolka"), V(NOT_FORMAL, "futbolka")], 4)
    assert (v.result, v.reason, votes) == (NOT_FORMAL, "futbolka", 2)


def test_aggregate_unknown_when_most_frames_unusable():
    frames = [V(FORMAL), V(UNKNOWN, "gövdə kadra kifayət qədər düşmür", "torso_hidden"),
              V(UNKNOWN, "gövdə kadra kifayət qədər düşmür", "torso_hidden"), V(UNKNOWN, "qaranlıq", "dark"), V(FORMAL)]
    v, votes = live.aggregate(frames, 5)
    assert (v.result, v.torso, votes) == (UNKNOWN, "torso_hidden", 2)
    assert live.aggregate([], 3)[0].torso == "no_frame"


@pytest.fixture
def live_cfg(make_config):
    return make_config(attire__prompts=FAKE_PROMPTS, attire__live__frames=3, attire__live__interval_s=0.1)


def fake_loader(calls):
    def load(cfg):
        calls.append(cfg)
        enc = FakeImageEncoder()
        return ev.Models(FakeDetector(), enc, ZeroShot(enc, cfg.attire.prompts))

    return load


def run_check(checker):
    assert checker.start()
    checker.join(10)
    assert not checker.busy
    return checker.state()


def test_live_check_white_shirt_and_model_loaded_once(live_cfg):
    calls, changes = [], []
    frame = shirt_frame((245, 245, 245))
    checker = live.AttireChecker(live_cfg, lambda: frame, on_change=lambda: changes.append(checker.status),
                                 loader=fake_loader(calls))
    assert checker.state()["status"] == "idle" and calls == []  # model ilk yoxlamaya qədər yüklənmir
    st = run_check(checker)
    assert st["status"] == "done" and st["loaded"]
    r = st["result"]
    assert (r["result"], r["label"], r["votes"], r["hint"]) == (FORMAL, "Rəsmi", 3, None)
    assert r["garment"] == "düyməli köynək" and r["color"] == "ağ" and r["age_s"] == 0
    assert r["jacket"] is False and r["jacket_rejected"] is False and "layering" in r
    assert changes[:2] == ["loading", "checking"] and changes[-1] == "done"
    run_check(checker)
    assert len(calls) == 1


def test_live_check_dark_top_is_not_formal(live_cfg):
    frame = shirt_frame((120, 70, 40))
    st = run_check(live.AttireChecker(live_cfg, lambda: frame, loader=fake_loader([])))
    assert st["result"]["result"] == NOT_FORMAL and st["result"]["reason"]


def test_live_check_gives_hint_when_torso_hidden(live_cfg):
    class LowFace(FakeDetector):
        def detect(self, frame):
            return Detection(FaceStatus.OK, Face(270.0, 340.0, 100.0, 120.0, 0.9, np.zeros((5, 2))), 1)

    def load(cfg):
        enc = FakeImageEncoder()
        return ev.Models(LowFace(), enc, ZeroShot(enc, cfg.attire.prompts))

    frame = shirt_frame((245, 245, 245))
    st = run_check(live.AttireChecker(live_cfg, lambda: frame, loader=load))
    r = st["result"]
    assert r["result"] == UNKNOWN and "gövdə" in r["reason"] and "aşağı əyin" in r["hint"]


def test_live_check_without_frames(live_cfg, monkeypatch):
    monkeypatch.setattr(live, "FIRST_FRAME_WAIT_S", 0.2)
    st = run_check(live.AttireChecker(live_cfg, lambda: None, loader=fake_loader([])))
    assert st["result"]["result"] == UNKNOWN and st["result"]["hint"] == live.HINTS["no_frame"]


def test_live_check_reports_load_error(live_cfg):
    def broken(cfg):
        raise FileNotFoundError("SigLIP 2 modeli tapılmadı")

    st = run_check(live.AttireChecker(live_cfg, lambda: None, loader=broken))
    assert st["status"] == "error" and "tapılmadı" in st["error"] and st["result"] is None


def test_live_check_cancel_and_single_run(live_cfg):
    frame = shirt_frame((245, 245, 245))
    checker = live.AttireChecker(live_cfg, lambda: frame, loader=fake_loader([]))
    assert checker.start()
    assert not checker.start()  # eyni anda ikinci yoxlama yoxdur
    checker.cancel()
    checker.join(10)
    assert checker.status in ("idle", "done") and not checker.busy


@pytest.mark.models
def test_live_loader_drops_text_model(siglip, cfg, yunet_path):
    models = live.load_live_models(cfg)  # ayrıca nüsxə — paylaşılan fixture dəyişmir
    assert models.encoder.model.text_model is None and models.encoder.tokenizer is None
    assert set(models.zero_shot.questions) == {"garment", "color", "tie", "jacket"}
    with pytest.raises(RuntimeError):
        models.encoder.encode_texts(["x"])


# --- config və ayrılıq -------------------------------------------------------------------


def test_attire_config_defaults(cfg):
    assert cfg.attire.model_dir.is_absolute()
    assert cfg.attire.model_dir.is_relative_to(PROJECT_ROOT / "models")


@pytest.mark.parametrize("override", [
    {"attire__collar_width": 5.0},
    {"attire__collar_height": 3.0},
    {"attire__min_visible": 0},
    {"attire__min_torso_px": 8},
    {"attire__min_sharpness": -1},
    {"attire__unknown_key": 1},
    {"attire__formal_garments": ["suit"]},
    {"attire__formal_garments": []},
    {"attire__prompts__extra": {"a": ["x"], "b": ["y"]}},
    {"attire__prompts__color": {"red": ["red"], "blue": ["blue"]}},
    {"attire__prompts__tie": {True: ["tie"], False: ["no tie"]}},
    {"attire__prompts__jacket": {"present": [], "absent": ["x"]}},
    {"attire__prompts__garment": {"shirt": ["shirt"]}},
    {"attire__formal_threshold": 1.0},
    {"attire__white_threshold": 0},
    {"attire__white_min_share": 1.5},
    {"attire__white_min_lightness": 120},
    {"attire__white_max_chroma": 0},
    {"attire__color_rule": "any"},
    {"attire__jacket_threshold": 0},
    {"attire__jacket_min_layering": 1.0},
    {"attire__live__frames": 0},
    {"attire__live__interval_s": 10},
    {"attire__live__threads": 0},
    {"attire__live__unknown": 1},
    {"attire__prompts": {}, "attire__live__enabled": True},  # kartda mətn təsvirləri mütləqdir
])
def test_attire_config_validation(make_config, override):
    with pytest.raises(ConfigError):
        make_config(**override)


def test_main_app_loads_torch_only_on_first_attire_check():
    """Əsas tətbiq və geyim kartı torch/transformers-i ilk yoxlamaya qədər yükləmir."""
    code = ("import sys, src.app, src.attire.live; "
            "print('transformers' in sys.modules, 'torch' in sys.modules, 'pandas' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.split()[:2] == ["False", "False"]
