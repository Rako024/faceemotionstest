"""Bir dəfəlik quraşdırma: modelləri endirir.

Bu, layihədə şəbəkədən istifadə edən YEGANƏ addımdır. Tətbiq özü işləyərkən
şəbəkəyə heç nə göndərmir (privacy.block_network).

İstifadə:
    python scripts/download_models.py               # YuNet + 4 ONNX emosiya modeli
    python scripts/download_models.py --torch       # + torch çəkiləri (GPU üçün)
    python scripts/download_models.py --test-fixture  # + pytest üçün test şəkli
    python scripts/download_models.py --extra       # + FER2013 modelləri (DeepFace, ViT, Kaggle CNN)

--extra üçün bir dəfəlik: pip install -r requirements-convert.txt (torch CPU + transformers + h5py).
Kaggle modeli giriş tələb edir — face_model.h5 faylını özünüz endirib models/ qovluğuna qoyun:
    https://www.kaggle.com/datasets/abhisheksingh016/machine-model-for-emotion-detection
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.config import EMOTIEFFLIB_KEYS, load_config  # noqa: E402

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
# NASA rəsmi portreti (ictimai mülkiyyət), scikit-image test məlumatlarından.
FIXTURE_URL = (
    "https://raw.githubusercontent.com/scikit-image/scikit-image/"
    "v0.24.0/skimage/data/astronaut.png"
)
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "face.png"

DEEPFACE_URL = "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.onnx"
VIT_REPO = "abhilash88/face-emotion-detection"
VIT_REVISION = "b947d8e7fd5b787f3d58ce9328ed473bda3a484c"  # yoxlanılmış commit
KAGGLE_H5 = ROOT / "models" / "face_model.h5"
KAGGLE_URL = "https://www.kaggle.com/datasets/abhisheksingh016/machine-model-for-emotion-detection"


def _parity_crop():
    """Paritet yoxlaması üçün test şəklindən üz kəsiyi (yoxdursa təsadüfi şəkil)."""
    import cv2
    import numpy as np

    from src.config import load_config
    from src.detector import FaceDetector

    if FIXTURE_PATH.is_file():
        img = cv2.imread(str(FIXTURE_PATH))
        det = FaceDetector(load_config().detector)
        d = det.detect(img)
        if d.face is not None:
            return det.crop(img, d.face)
    return np.random.default_rng(0).integers(0, 255, (160, 140, 3), dtype=np.uint8)


def export_vit(dest: Path) -> None:
    """Hugging Face ViT modelini ONNX-ə çevirir və torch ilə nəticələri müqayisə edir."""
    if dest.is_file():
        print(f"  ✓ artıq var: {dest.name}  sha256={sha256(dest)[:16]}…")
        return
    try:
        import torch
        from transformers import ViTForImageClassification
    except ImportError:
        print("  ✗ ViT: torch/transformers yoxdur — pip install -r requirements-convert.txt")
        return
    import numpy as np
    import onnxruntime as ort

    from src.fer_models import preprocess_face
    from src.model_zoo import spec

    print(f"  ↓ {VIT_REPO}@{VIT_REVISION[:8]} (~343 MB, bir dəfə)")
    model = ViTForImageClassification.from_pretrained(VIT_REPO, revision=VIT_REVISION, attn_implementation="eager")
    model.eval()

    class Logits(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m(pixel_values=pixel_values).logits

    tmp = dest.with_suffix(".onnx.part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = dict(input_names=["pixel_values"], output_names=["logits"],
                         dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(Logits(model), (torch.zeros(1, 3, 224, 224),), str(tmp), dynamo=False, **export_kwargs)
    except TypeError:  # köhnə torch-da dynamo arqumenti yoxdur
        torch.onnx.export(Logits(model), (torch.zeros(1, 3, 224, 224),), str(tmp), **export_kwargs)

    # Paritet: eyni ön emal olunmuş giriş üçün torch və onnxruntime nəticələri
    batch = preprocess_face(_parity_crop(), spec("vit_fer2013"), channels=3, nchw=True)
    with torch.no_grad():
        ref = model(pixel_values=torch.from_numpy(batch)).logits.numpy()
    got = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"]).run(None, {"pixel_values": batch})[0]
    diff = float(np.abs(ref - got).max())
    if diff > 1e-3:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ViT ONNX paritet xətası: max |Δ| = {diff:.2e}")
    tmp.replace(dest)
    print(f"  ✓ {dest.name}  paritet max |Δ| = {diff:.1e}  sha256={sha256(dest)[:16]}…")


def convert_kaggle(src_h5: Path, dest: Path) -> None:
    if not src_h5.is_file():
        print(f"  – Kaggle CNN: {src_h5.relative_to(ROOT)} yoxdur. Faylı buradan endirib bu yola qoyun:\n    {KAGGLE_URL}")
        return
    import numpy as np
    import onnxruntime as ort

    from keras_h5_to_onnx import convert, keras_reference

    info = convert(src_h5, dest)
    h, w, c = info["input_shape"]
    print(f"  ✓ {dest.name}  giriş {h}×{w}×{c}, {info['classes']} sinif, qatlar: {', '.join(info['layers'])}")
    rng = np.random.default_rng(0)
    batch = rng.random((4, h, w, c), dtype=np.float32)
    got = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"]).run(None, {"input": batch})[0]
    ref = keras_reference(src_h5, batch)
    if ref is None:
        print("    (Keras qurulmayıb — paritet yoxlaması ötürüldü: pip install keras)")
    else:
        diff = float(np.abs(ref - got).max())
        if diff > 1e-4:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"Kaggle ONNX paritet xətası: max |Δ| = {diff:.2e}")
        print(f"    paritet (Keras ilə) max |Δ| = {diff:.1e}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, min_bytes: int) -> None:
    if dest.is_file() and dest.stat().st_size >= min_bytes:
        print(f"  ✓ artıq var: {dest.name}  sha256={sha256(dest)[:16]}…")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  ↓ {dest.name}")
    urllib.request.urlretrieve(url, tmp)
    head = tmp.read_bytes()[:64]
    if tmp.stat().st_size < min_bytes or head.startswith(b"version https://git-lfs"):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{url} gözlənilməz məzmun qaytardı")
    tmp.replace(dest)
    print(f"  ✓ {dest.name}  sha256={sha256(dest)[:16]}…")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default=None, help="config.yaml yolu")
    parser.add_argument("--torch", action="store_true", help="torch (.pt) çəkilərini də endir")
    parser.add_argument("--test-fixture", action="store_true", help="pytest üçün üz şəklini endir")
    parser.add_argument("--extra", action="store_true", help="FER2013 modelləri: DeepFace, ViT, Kaggle CNN")
    parser.add_argument("--kaggle-h5", default=str(KAGGLE_H5), help="Kaggle face_model.h5 yolu")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config(args.config)

    print("YuNet üz detektoru:")
    fetch(YUNET_URL, cfg.detector.model_path, min_bytes=100_000)

    from emotiefflib.utils import get_model_path_onnx, get_model_path_torch

    # Hamısı: oflayn qiymətləndirici (python -m src.evaluate) bütün quraşdırılmışları müqayisə edir
    models = list(dict.fromkeys([cfg.model.name, *EMOTIEFFLIB_KEYS]))
    print("EmotiEffLib ONNX modelləri (~/.emotiefflib):")
    for name in models:
        path = Path(get_model_path_onnx(name))
        print(f"  ✓ {name}  sha256={sha256(path)[:16]}…")

    if args.torch:
        print("EmotiEffLib torch modelləri:")
        for name in models:
            path = Path(get_model_path_torch(name))
            print(f"  ✓ {name}  sha256={sha256(path)[:16]}…")

    if args.test_fixture:
        print("Test şəkli:")
        fetch(FIXTURE_URL, FIXTURE_PATH, min_bytes=100_000)

    if args.extra:
        from src.model_zoo import spec

        print("FER2013 modelləri:")
        fetch(DEEPFACE_URL, spec("deepface_emotion").onnx_path, min_bytes=1_000_000)
        export_vit(spec("vit_fer2013").onnx_path)
        convert_kaggle(Path(args.kaggle_h5), spec("kaggle_fer_cnn").onnx_path)

    print("Hazırdır. Tətbiq artıq oflayn işləyə bilər.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
