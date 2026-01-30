import logging
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = _REPO_ROOT / "model"
MODEL_PATH = MODEL_DIR / "model.keras"

_model_instance: tf.keras.Model | None = None
_input_shape: tuple[int, int, int] | None = None
_class_names: list[str] = ["adenocarcinoma", "squamous_cell_carcinoma", "normal"]


def ensure_model_file() -> Path:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1024 * 1024:
        return MODEL_PATH
    from huggingface_hub import hf_hub_download
    space_id = os.environ.get("SPACE_ID", "liviuorehovschi/histomancer-api")
    token = os.environ.get("HF_TOKEN")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    downloaded_path = hf_hub_download(
        repo_id=space_id,
        filename="model/model.keras",
        local_dir=str(_REPO_ROOT),
        local_dir_use_symlinks=False,
        token=token,
        force_download=True,
    )
    downloaded = Path(downloaded_path)
    if downloaded.stat().st_size < 1024 * 1024:
        raise RuntimeError(f"Downloaded file is only {downloaded.stat().st_size} bytes (expected >1MB). Xet pointer not resolved.")
    if downloaded.resolve() != MODEL_PATH.resolve():
        import shutil
        shutil.copy2(downloaded, MODEL_PATH)
    return MODEL_PATH


def load_model() -> tf.keras.Model:
    global _model_instance, _input_shape
    if _model_instance is not None:
        return _model_instance
    path = ensure_model_file()
    _model_instance = tf.keras.models.load_model(str(path), compile=False)
    try:
        layer = _model_instance.input
        if hasattr(layer, "shape") and layer.shape is not None:
            s = layer.shape
            if len(s) == 4:
                _input_shape = (int(s[1]), int(s[2]), int(s[3]))
            else:
                _input_shape = (224, 224, 3)
        else:
            _input_shape = (224, 224, 3)
    except AttributeError:
        _input_shape = (224, 224, 3)
        dummy = tf.zeros((1, 224, 224, 3))
        _model_instance(dummy)
        try:
            s = _model_instance.input_shape
            if s and len(s) == 4:
                _input_shape = (int(s[1]), int(s[2]), int(s[3]))
        except Exception:
            pass
    return _model_instance


def get_input_shape() -> tuple[int, int, int]:
    if _input_shape is None:
        load_model()
    return _input_shape or (224, 224, 3)


def get_class_names() -> list[str]:
    return _class_names


def get_model_diagnostics() -> dict:
    p = MODEL_PATH
    out = {
        "model_path": str(p),
        "model_path_exists": p.exists(),
        "model_dir": str(MODEL_DIR),
        "model_dir_exists": MODEL_DIR.exists(),
        "repo_root": str(_REPO_ROOT),
    }
    if p.exists():
        out["model_path_size"] = p.stat().st_size
    if MODEL_DIR.exists():
        out["model_dir_listing"] = [x.name for x in MODEL_DIR.iterdir()]
    else:
        out["model_dir_listing"] = []
    return out


def preprocess_image(image: Image.Image) -> np.ndarray:
    shape = get_input_shape()
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = np.array(image)
    img = tf.image.resize(img, (shape[0], shape[1]))
    img = tf.cast(img, tf.float32) / 255.0
    return np.expand_dims(img.numpy(), axis=0)


def predict(image_batch: np.ndarray) -> tuple[int, np.ndarray]:
    model = load_model()
    logits = model(image_batch, training=False)
    if hasattr(logits, "numpy"):
        probs = tf.nn.softmax(logits).numpy()
    else:
        probs = np.asarray(tf.nn.softmax(logits))
    probs = np.squeeze(probs)
    if probs.ndim == 0:
        probs = np.expand_dims(probs, 0)
    idx = int(np.argmax(probs))
    return idx, probs
