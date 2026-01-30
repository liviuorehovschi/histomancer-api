import json
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)

# Repo root: same locally and on HF Space. Model lives at model/model.keras.
_REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = _REPO_ROOT / "model"
MODEL_PATH = MODEL_DIR / "model.keras"

_model_instance: tf.keras.Model | None = None
_input_shape: tuple[int, int, int] | None = None
_class_names: list[str] = ["adenocarcinoma", "squamous_cell_carcinoma", "normal"]


def _is_lfs_pointer(path: Path) -> bool:
    """True if the file is a Git LFS pointer (not the actual model)."""
    if not path.exists() or path.stat().st_size < 200:
        return True
    try:
        with open(path, "rb") as f:
            first = f.read(100).decode("utf-8", errors="ignore")
        return first.strip().startswith("version https://git-lfs.github.com")
    except Exception:
        return False


def _ensure_model_file() -> None:
    """If model is missing or an LFS pointer, download it from this Space repo (HF serves the real file)."""
    if MODEL_PATH.exists() and not _is_lfs_pointer(MODEL_PATH):
        return
    space_id = os.environ.get("SPACE_ID", "liviuorehovschi/histomancer-api")
    token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import hf_hub_download

        logger.info("Model missing or LFS pointer; downloading from Space repo %s", space_id)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for filename in ("model/model.keras", "model.keras"):
            try:
                path = hf_hub_download(
                    repo_id=space_id,
                    filename=filename,
                    local_dir=str(_REPO_ROOT),
                    local_dir_use_symlinks=False,
                    token=token,
                )
                if path and Path(path).exists() and not _is_lfs_pointer(Path(path)):
                    if Path(path).resolve() != MODEL_PATH.resolve():
                        MODEL_DIR.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, MODEL_PATH)
                    return
            except Exception as e:
                logger.debug("Download %s failed: %s", filename, e)
                continue
    except Exception as e:
        logger.warning("Could not download model from Hub: %s", e)


def _find_model_path() -> str:
    _ensure_model_file()
    if MODEL_PATH.exists() and not _is_lfs_pointer(MODEL_PATH):
        return str(MODEL_PATH)
    if MODEL_DIR.is_dir():
        for p in list(MODEL_DIR.glob("*.keras")) or list(MODEL_DIR.glob("**/*.keras")):
            if p.exists() and not _is_lfs_pointer(p):
                return str(p)
    return str(MODEL_PATH)


def _rewrite_keras_config_for_old_tf(path: str) -> str:
    """Rewrite .keras zip: batch_shape -> batch_input_shape so TF <2.15 can load it."""
    path = Path(path)
    if path.suffix != ".keras" or not path.exists():
        return str(path)
    tmp = Path(tempfile.mktemp(suffix=".keras"))
    try:
        with zipfile.ZipFile(path, "r") as zin:
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
                for name in zin.namelist():
                    data = zin.read(name)
                    if name.endswith(".json") and b"batch_shape" in data:
                        config = json.loads(data.decode("utf-8"))
                        def fix(d):
                            if isinstance(d, dict):
                                if "batch_shape" in d:
                                    d["batch_input_shape"] = d.pop("batch_shape")
                                for v in d.values():
                                    fix(v)
                            elif isinstance(d, list):
                                for v in d:
                                    fix(v)
                        fix(config)
                        data = json.dumps(config, indent=2).encode("utf-8")
                    zout.writestr(name, data)
        return str(tmp)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return str(path)


def load_model() -> tf.keras.Model:
    global _model_instance, _input_shape
    if _model_instance is not None:
        return _model_instance
    path = _find_model_path()
    logger.info("Loading model from %s (exists=%s)", path, Path(path).exists())
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found at {path}")
    if _is_lfs_pointer(Path(path)):
        raise RuntimeError(
            f"Model at {path} is a Git LFS pointer, not the actual file. "
            "Upload the real model.keras to the Space repo (Files tab) or ensure LFS is resolved."
        )
    load_path = _rewrite_keras_config_for_old_tf(path)
    try:
        try:
            _model_instance = tf.keras.models.load_model(
                load_path, compile=False, safe_mode=False
            )
        except TypeError:
            _model_instance = tf.keras.models.load_model(load_path, compile=False)
    finally:
        if load_path != path and Path(load_path).exists():
            Path(load_path).unlink(missing_ok=True)
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
    """Return path, exists, size, is_lfs, and model dir listing for debugging."""
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
        out["model_path_is_lfs_pointer"] = _is_lfs_pointer(p)
    if MODEL_DIR.exists():
        out["model_dir_listing"] = [str(x.name) for x in MODEL_DIR.iterdir()]
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
