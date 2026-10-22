"""Keras .h5 (Sequential/xətti) modelini TensorFlow olmadan ONNX-ə çevirir.

Yalnız h5py + onnx istifadə edir. Dəstəklənən qatlar: InputLayer, Conv2D,
MaxPooling2D, AveragePooling2D, BatchNormalization, Dropout, SpatialDropout2D,
Flatten, Dense, Activation, ReLU, LeakyReLU, Softmax, GlobalAveragePooling2D,
GlobalMaxPooling2D, ZeroPadding2D. Tanınmayan qat aydın xəta verir.

Giriş NHWC saxlanılır (Keras ilə eyni), daxildə NCHW-yə çevrilir; Flatten-dən
əvvəl yenidən NHWC sırasına qaytarılır ki, Dense çəkiləri düz düşsün.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ACTIVATIONS = {"relu": "Relu", "softmax": "Softmax", "sigmoid": "Sigmoid", "tanh": "Tanh", "elu": "Elu"}


class UnsupportedLayer(ValueError):
    pass


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _layer_weights(h5, layer_name: str) -> list[np.ndarray]:
    group = h5["model_weights"] if "model_weights" in h5 else h5
    if layer_name not in group:
        return []
    g = group[layer_name]
    names = [_decode(n) for n in g.attrs.get("weight_names", [])]
    return [np.asarray(g[n]) for n in names]


def _activation_name(cfg) -> str:
    act = cfg.get("activation", "linear")
    if isinstance(act, dict):  # Keras 3 serialization
        act = act.get("config", {}).get("name") or act.get("class_name", "linear")
    return str(act).lower()


def convert(h5_path: str | Path, onnx_path: str | Path, opset: int = 13) -> dict:
    import h5py
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    with h5py.File(h5_path, "r") as h5:
        raw = h5.attrs.get("model_config")
        if raw is None:
            raise UnsupportedLayer("h5 faylında model_config yoxdur (yalnız çəkilər saxlanılıb) — arxitektura bilinmir")
        model_cfg = json.loads(_decode(raw))
        layers = model_cfg["config"]["layers"] if isinstance(model_cfg["config"], dict) else model_cfg["config"]
        if model_cfg["class_name"] not in ("Sequential", "Functional", "Model"):
            raise UnsupportedLayer(f"Model tipi dəstəklənmir: {model_cfg['class_name']}")

        nodes, inits = [], []
        counter = {"n": 0}

        def name(prefix: str) -> str:
            counter["n"] += 1
            return f"{prefix}_{counter['n']}"

        def const(arr: np.ndarray, prefix: str) -> str:
            n = name(prefix)
            inits.append(numpy_helper.from_array(np.ascontiguousarray(arr.astype(np.float32)), n))
            return n

        def node(op: str, inputs: list[str], **attrs) -> str:
            out = name(op.lower())
            nodes.append(helper.make_node(op, inputs, [out], **attrs))
            return out

        def activate(x: str, act: str, cfg=None) -> str:
            if act in ("linear", None, ""):
                return x
            if act == "softmax":
                return node("Softmax", [x], axis=-1 if layout == "flat" else 1)
            if act == "leaky_relu":
                return node("LeakyRelu", [x], alpha=float((cfg or {}).get("alpha", (cfg or {}).get("negative_slope", 0.3))))
            if act not in ACTIVATIONS:
                raise UnsupportedLayer(f"Aktivasiya dəstəklənmir: {act}")
            return node(ACTIVATIONS[act], [x])

        # Giriş forması
        first = layers[0]["config"]
        shape = first.get("batch_input_shape") or first.get("batch_shape") or first.get("input_shape")
        if shape is None:
            # Sequential-da InputLayer olmaya bilər — ilk qatın batch_input_shape-i
            shape = next((l["config"].get("batch_input_shape") for l in layers if l["config"].get("batch_input_shape")), None)
        if shape is None or len(shape) != 4:
            raise UnsupportedLayer(f"Giriş forması 4 ölçülü olmalıdır (N,H,W,C), alındı: {shape}")
        _, H, W, C = shape
        graph_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", H, W, C])
        x = node("Transpose", ["input"], perm=[0, 3, 1, 2])
        layout = "nchw"
        used = []

        for layer in layers:
            cls, cfg = layer["class_name"], layer["config"]
            lname = cfg.get("name", cls)
            if cfg.get("data_format", "channels_last") != "channels_last":
                raise UnsupportedLayer(f"{lname}: yalnız channels_last dəstəklənir")
            w = _layer_weights(h5, lname)
            used.append(cls)

            if cls == "InputLayer" or cls in ("Dropout", "SpatialDropout2D", "GaussianNoise", "GaussianDropout"):
                continue
            if cls == "Conv2D":
                kernel = w[0]  # (kh, kw, in, out)
                ins = [x, const(kernel.transpose(3, 2, 0, 1), "W")]
                if cfg.get("use_bias", True):
                    ins.append(const(w[1], "B"))
                attrs = dict(kernel_shape=list(kernel.shape[:2]), strides=list(cfg.get("strides", [1, 1])),
                             dilations=list(cfg.get("dilation_rate", [1, 1])))
                attrs["auto_pad"] = "SAME_UPPER" if cfg.get("padding", "valid") == "same" else "VALID"
                x = activate(node("Conv", ins, **attrs), _activation_name(cfg))
            elif cls in ("MaxPooling2D", "AveragePooling2D"):
                pool = list(cfg.get("pool_size", [2, 2]))
                strides = list(cfg.get("strides") or pool)
                pad = "SAME_UPPER" if cfg.get("padding", "valid") == "same" else "VALID"
                op = "MaxPool" if cls == "MaxPooling2D" else "AveragePool"
                extra = {} if op == "MaxPool" else {"count_include_pad": 0}
                x = node(op, [x], kernel_shape=pool, strides=strides, auto_pad=pad, **extra)
            elif cls == "ZeroPadding2D":
                p = cfg.get("padding", [[1, 1], [1, 1]])
                if isinstance(p, int):
                    p = [[p, p], [p, p]]
                (t, b), (l, r) = p
                pads = name("pads")
                inits.append(numpy_helper.from_array(np.array([0, 0, t, l, 0, 0, b, r], dtype=np.int64), pads))
                x = node("Pad", [x, pads])
            elif cls == "BatchNormalization":
                axis = cfg.get("axis", -1)
                if isinstance(axis, list):
                    axis = axis[0]
                if axis not in (-1, 3, 1):
                    raise UnsupportedLayer(f"{lname}: BatchNormalization axis={axis}")
                i = 0
                gamma = w[i] if cfg.get("scale", True) else None
                i += 1 if cfg.get("scale", True) else 0
                beta = w[i] if cfg.get("center", True) else None
                i += 1 if cfg.get("center", True) else 0
                mean, var = w[i], w[i + 1]
                n_ch = mean.shape[0]
                gamma = np.ones(n_ch) if gamma is None else gamma
                beta = np.zeros(n_ch) if beta is None else beta
                x = node("BatchNormalization", [x, const(gamma, "g"), const(beta, "b"), const(mean, "m"), const(var, "v")],
                         epsilon=float(cfg.get("epsilon", 1e-3)))
            elif cls in ("Flatten", "GlobalAveragePooling2D", "GlobalMaxPooling2D"):
                if cls == "Flatten":
                    if layout == "nchw":
                        x = node("Transpose", [x], perm=[0, 2, 3, 1])  # Keras NHWC sırası ilə düzləndir
                    x = node("Flatten", [x], axis=1)
                else:
                    op = "GlobalAveragePool" if cls == "GlobalAveragePooling2D" else "GlobalMaxPool"
                    x = node("Flatten", [node(op, [x])], axis=1)
                layout = "flat"
            elif cls == "Dense":
                if layout != "flat":
                    raise UnsupportedLayer(f"{lname}: Dense-dən əvvəl Flatten gözlənilir")
                kernel = w[0]
                bias = w[1] if cfg.get("use_bias", True) else np.zeros(kernel.shape[1])
                x = activate(node("Gemm", [x, const(kernel, "W"), const(bias, "B")]), _activation_name(cfg))
            elif cls == "Activation":
                x = activate(x, _activation_name(cfg), cfg)
            elif cls == "ReLU":
                x = node("Relu", [x])
            elif cls == "LeakyReLU":
                x = activate(x, "leaky_relu", cfg)
            elif cls == "Softmax":
                x = activate(x, "softmax")
            else:
                raise UnsupportedLayer(f"Qat dəstəklənmir: {cls} ({lname})")

        if layout != "flat":
            raise UnsupportedLayer("Model 2 ölçülü çıxışla bitməlidir")
        out_dim = int(np.asarray(_layer_weights(h5, [l for l in layers if l["class_name"] == "Dense"][-1]["config"]["name"])[0]).shape[1])
        nodes.append(helper.make_node("Identity", [x], ["output"]))
        graph = helper.make_graph(nodes, "keras_h5", [graph_input],
                                  [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", out_dim])], inits)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], producer_name="keras_h5_to_onnx")
        model.ir_version = 9  # onnxruntime-ın dəstəklədiyi IR (yeni onnx paketləri default olaraq daha yenisini yazır)
        onnx.checker.check_model(model)
        Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(onnx_path))
        return {"input_shape": [H, W, C], "classes": out_dim, "layers": used}


def keras_reference(h5_path: str | Path, batch: np.ndarray) -> np.ndarray | None:
    """Keras (torch backend) quraşdırılıbsa, orijinal modelin çıxışını qaytarır — paritet yoxlaması üçün."""
    import os

    os.environ.setdefault("KERAS_BACKEND", "torch")
    try:
        import keras
    except ImportError:
        return None
    model = keras.models.load_model(str(h5_path), compile=False)
    return np.asarray(model.predict(batch, verbose=0))
