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

_REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = _REPO_ROOT / "model"
MODEL_PATH = MODEL_DIR / "model.keras"

_model_instance: tf.keras.Model | None = None
_input_shape: tuple[int, int, int] | None = None
_class_names: list[str] = ["adenocarcinoma", "squamous_cell_carcinoma", "normal"]


def _fix_dense_18_inbound(config: dict) -> None:
    """Force dense_18 to have exactly one inbound connection (fix TF load bug)."""
    if isinstance(config, dict):
        # Keras 3: graph might be in config.config.nodes OR in layers[i].inbound_nodes OR in build_config
        if "config" in config and isinstance(config["config"], dict):
            cfg = config["config"]
            if "layers" in cfg and isinstance(cfg["layers"], list):
                layers = cfg["layers"]
                # Check layer-level inbound_nodes (Keras 3 style)
                for layer_idx, layer in enumerate(layers):
                    if isinstance(layer, dict) and layer.get("name") == "dense_18":
                        if "inbound_nodes" in layer:
                            nodes = layer["inbound_nodes"]
                            if isinstance(nodes, list) and len(nodes) > 0:
                                # nodes is [[[layer_name, node_idx, tensor_idx, kwargs], ...], ...]
                                if isinstance(nodes[0], list) and len(nodes[0]) > 1:
                                    layer["inbound_nodes"] = [nodes[0][:1]]
                                    logger.info("Patched dense_18 layers[%d].inbound_nodes to single input", layer_idx)
                        break
                # Also check config.config.nodes (Keras 2 style)
                if "nodes" in cfg and isinstance(cfg["nodes"], list):
                    nodes = cfg["nodes"]
                    for layer_idx, layer in enumerate(layers):
                        if isinstance(layer, dict) and layer.get("name") == "dense_18":
                            if layer_idx < len(nodes):
                                node = nodes[layer_idx]
                                if isinstance(node, list) and len(node) > 1:
                                    nodes[layer_idx] = [node[0]]
                                    logger.info("Patched dense_18 nodes[%d] to single inbound", layer_idx)
                            break
        # Also check build_config for input/output layers
        if "build_config" in config and isinstance(config["build_config"], dict):
            bc = config["build_config"]
            # If build_config has input_layers/output_layers, we might need to fix those
            # But typically the issue is in inbound_nodes, not build_config
        # Recursive check for nested structures
        if config.get("name") == "dense_18" and "inbound_nodes" in config:
            nodes = config["inbound_nodes"]
            if isinstance(nodes, list) and nodes:
                if isinstance(nodes[0], list) and len(nodes[0]) > 1:
                    config["inbound_nodes"] = [nodes[0][:1]] if nodes[0] else []
                    logger.info("Patched dense_18 inbound_nodes (direct) to single input")
        for v in config.values():
            _fix_dense_18_inbound(v)
    elif isinstance(config, list):
        for v in config:
            _fix_dense_18_inbound(v)


class DenseAcceptTwoInputs(tf.keras.layers.Dense):
    """Dense layer that accepts 2 inputs but uses only the first (fixes TF load bug)."""
    
    def call(self, inputs, **kwargs):
        if isinstance(inputs, (list, tuple)) and len(inputs) > 1:
            inputs = inputs[0]
        return super().call(inputs, **kwargs)
    
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 1:
            input_shape = input_shape[0]
        return super().compute_output_shape(input_shape)
    
    @classmethod
    def from_config(cls, config):
        # Ensure we can deserialize from saved config
        return cls(**config)


# Register globally so Keras finds it when deserializing
tf.keras.utils.get_custom_objects()["Dense"] = DenseAcceptTwoInputs


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 200:
        return True
    try:
        with open(path, "rb") as f:
            first = f.read(100).decode("utf-8", errors="ignore")
        return first.strip().startswith("version https://git-lfs.github.com")
    except Exception:
        return False


def _ensure_model_file() -> None:
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
            f"Model at {path} is a Git LFS pointer. Upload the real model.keras to the Space repo."
        )
    # Use custom Dense layer that accepts 2 inputs but uses first (fixes TF load bug).
    # Use custom Dense layer registered globally + custom_objects (double registration for Keras 3).
    custom_objects = {"Dense": DenseAcceptTwoInputs}
    try:
        _model_instance = tf.keras.models.load_model(path, compile=False, safe_mode=False, custom_objects=custom_objects)
    except (TypeError, ValueError) as e:
        # If safe_mode fails, try without it
        try:
            _model_instance = tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects)
        except ValueError as e2:
            # If custom_objects doesn't work, the model file itself is corrupted
            raise RuntimeError(
                f"Model load failed: {e2}. The model file has dense_18 wired with 2 inputs. "
                "You need to re-save the model with only 1 input to dense_18."
            ) from e2
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


def _dump_keras_config_for_dense_18(path: Path) -> dict | None:
    """Read .keras zip and return config structure so we can see how dense_18 is wired."""
    if not path.exists() or path.suffix != ".keras" or _is_lfs_pointer(path):
        return None
    try:
        with zipfile.ZipFile(path, "r") as z:
            for name in z.namelist():
                if not name.endswith(".json") or "dense_18" not in z.read(name).decode("utf-8", errors="ignore"):
                    continue
                config = json.loads(z.read(name).decode("utf-8"))
                out = {"config_file": name, "top_level_keys": list(config.keys()) if isinstance(config, dict) else []}
                found = []

                def collect(d, path_prefix=""):
                    if isinstance(d, dict):
                        if d.get("name") == "dense_18":
                            found.append({"path": path_prefix, "keys": list(d.keys()), "inbound_nodes": d.get("inbound_nodes"), "full_layer": d})
                        for k, v in d.items():
                            collect(v, f"{path_prefix}.{k}")
                    elif isinstance(d, list):
                        for i, v in enumerate(d):
                            collect(v, f"{path_prefix}[{i}]")
                collect(config)
                out["dense_18_occurrences"] = found[:5]
                if isinstance(config, dict) and "config" in config and isinstance(config["config"], dict):
                    cfg = config["config"]
                    out["config_config_keys"] = list(cfg.keys())
                    if "layers" in cfg and isinstance(cfg["layers"], list):
                        out["layers_count"] = len(cfg["layers"])
                        for i, layer in enumerate(cfg["layers"]):
                            if isinstance(layer, dict) and layer.get("name") == "dense_18":
                                out["dense_18_in_layers_index"] = i
                                out["dense_18_layer_keys"] = list(layer.keys())
                                out["dense_18_inbound_nodes_raw"] = layer.get("inbound_nodes")
                                break
                    # Check build_config for graph topology (Keras 3 might store it there)
                    if "build_config" in config and isinstance(config["build_config"], dict):
                        bc = config["build_config"]
                        out["build_config_keys"] = list(bc.keys())
                        if "input_layers" in bc:
                            out["build_config_input_layers"] = bc["input_layers"]
                        if "output_layers" in bc:
                            out["build_config_output_layers"] = bc["output_layers"]
                    # Graph topology: nodes[i] is the node for layers[i] (Keras 2 style)
                    if "nodes" in cfg:
                        nodes = cfg["nodes"]
                        out["nodes_exists"] = True
                        out["nodes_type"] = type(nodes).__name__
                        if isinstance(nodes, list):
                            out["nodes_count"] = len(nodes)
                            for layer_idx, layer in enumerate(cfg.get("layers", [])):
                                if isinstance(layer, dict) and layer.get("name") == "dense_18":
                                    out["dense_18_layer_index"] = layer_idx
                                    if layer_idx < len(nodes):
                                        out["dense_18_node_index"] = layer_idx
                                        out["dense_18_node_raw"] = nodes[layer_idx]
                                        out["dense_18_node_length"] = len(nodes[layer_idx]) if isinstance(nodes[layer_idx], list) else None
                                    break
                    else:
                        out["nodes_exists"] = False
                        # In Keras 3, connections might be in layers themselves or build_config
                        # Check if layers have inbound_nodes at the layer level (not in .config)
                        if "layers" in cfg:
                            for i, layer in enumerate(cfg["layers"]):
                                if isinstance(layer, dict) and layer.get("name") == "dense_18":
                                    # Check top-level keys of layer (not layer.config)
                                    out["dense_18_layer_top_keys"] = [k for k in layer.keys() if k != "config"]
                                    if "inbound_nodes" in layer:
                                        out["dense_18_inbound_nodes_at_layer"] = layer["inbound_nodes"]
                                    break
                return out
    except Exception as e:
        return {"error": str(e)}
    return None


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
        out["model_path_is_lfs_pointer"] = _is_lfs_pointer(p)
    if MODEL_DIR.exists():
        out["model_dir_listing"] = [x.name for x in MODEL_DIR.iterdir()]
    else:
        out["model_dir_listing"] = []
    cfg = _dump_keras_config_for_dense_18(p)
    if cfg:
        out["keras_config_dense_18"] = cfg
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
