from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
_model_instance: tf.keras.Model | None = None
_input_shape: tuple[int, int, int] | None = None
_class_names: list[str] = ["adenocarcinoma", "squamous_cell_carcinoma", "normal"]


def _find_model_path() -> str:
    if MODEL_DIR.is_dir():
        keras_file = next(MODEL_DIR.glob("*.keras"), None) or next(MODEL_DIR.glob("**/*.keras"), None)
        if keras_file:
            return str(keras_file)
        saved_model = MODEL_DIR / "saved_model.pb"
        if saved_model.exists():
            return str(MODEL_DIR)
    return str(MODEL_DIR)


def load_model() -> tf.keras.Model:
    global _model_instance, _input_shape
    if _model_instance is not None:
        return _model_instance
    path = _find_model_path()
    _model_instance = tf.keras.models.load_model(path)
    layer = _model_instance.input
    if hasattr(layer, "shape") and layer.shape is not None:
        s = layer.shape
        if len(s) == 4:
            _input_shape = (int(s[1]), int(s[2]), int(s[3]))
        else:
            _input_shape = (224, 224, 3)
    else:
        _input_shape = (224, 224, 3)
    return _model_instance


def get_input_shape() -> tuple[int, int, int]:
    if _input_shape is None:
        load_model()
    return _input_shape or (224, 224, 3)


def get_class_names() -> list[str]:
    return _class_names


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
