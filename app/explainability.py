import base64
import io

import numpy as np
import tensorflow as tf
from PIL import Image

from app.model import load_model, get_input_shape, get_class_names, preprocess_image, predict


def _last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer | None:
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
            return layer
    return None


def _gradcam(model: tf.keras.Model, img_input: np.ndarray, pred_idx: int) -> np.ndarray:
    conv_layer = _last_conv_layer(model)
    if conv_layer is None:
        return np.zeros_like(img_input[0])

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_input)
        class_channel = preds[:, pred_idx]

    grads = tape.gradient(class_channel, conv_output)
    if grads is None:
        return np.zeros_like(img_input[0])

    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(weights * conv_output, axis=-1)
    cam = tf.nn.relu(cam)
    cam = tf.squeeze(cam).numpy()
    h, w = img_input.shape[1], img_input.shape[2]
    cam = tf.image.resize(
        np.expand_dims(np.expand_dims(cam, -1), 0), (h, w)
    ).numpy().squeeze()
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)
    return cam


def gradcam_image(image: Image.Image) -> tuple[np.ndarray, int, float]:
    model = load_model()
    x = preprocess_image(image)
    pred_idx, probs = predict(x)
    conf = float(probs[pred_idx])
    cam = _gradcam(model, x, pred_idx)
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap = np.stack([cam_uint8, cam_uint8, cam_uint8], axis=-1)
    img_arr = np.array(image.convert("RGB").resize((x.shape[2], x.shape[1])))
    overlay = (0.5 * img_arr + 0.5 * heatmap).astype(np.uint8)
    return overlay, pred_idx, conf


def saliency_map(image: Image.Image) -> tuple[np.ndarray, int, float]:
    model = load_model()
    x = preprocess_image(image)
    x_var = tf.Variable(x)

    with tf.GradientTape() as tape:
        tape.watch(x_var)
        preds = model(x_var, training=False)
        top_class = tf.argmax(preds[0])
        top_score = preds[0, top_class]

    grads = tape.gradient(top_score, x_var)
    if grads is None:
        sal = np.zeros_like(x[0])
    else:
        sal = tf.reduce_max(tf.abs(grads), axis=-1).numpy().squeeze()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    sal_uint8 = (sal * 255).astype(np.uint8)
    sal_rgb = np.stack([sal_uint8, sal_uint8, sal_uint8], axis=-1)
    pred_idx, probs = predict(x)
    conf = float(probs[pred_idx])
    return sal_rgb, pred_idx, conf


def encode_png(arr: np.ndarray) -> str:
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
