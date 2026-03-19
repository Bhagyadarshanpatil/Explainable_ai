"""Grad-CAM utilities for the two-channel spatial + frequency CNN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - exercised only in environments without TensorFlow
    tf = None

from .model import LAST_CONV_LAYER_NAME


@dataclass
class GradCAMResult:
    heatmap: np.ndarray
    normalized_input: np.ndarray
    predicted_index: int
    predicted_score: float
    target_layer: str


def _require_tensorflow() -> None:
    if tf is None:
        raise ImportError(
            "TensorFlow is required for Grad-CAM support. Install tensorflow before using these helpers."
        )


def _prepare_input(input_tensor: np.ndarray) -> np.ndarray:
    array = np.asarray(input_tensor, dtype=np.float32)
    if array.ndim == 3:
        array = array[np.newaxis, ...]
    if array.ndim != 4:
        raise ValueError("Expected a 3D sample tensor or a 4D batch tensor.")
    if array.shape[-1] != 2:
        raise ValueError("Grad-CAM expects the model input to have two channels.")
    return array


def compute_gradcam(
    model,
    input_tensor: np.ndarray,
    class_index: Optional[int] = None,
    layer_name: str = LAST_CONV_LAYER_NAME,
) -> GradCAMResult:
    """Generate a Grad-CAM heatmap for the requested class.

    The heatmap is computed from the final convolutional feature map while preserving
    the two-channel [spatial, frequency] input convention.
    """
    _require_tensorflow()
    normalized_input = _prepare_input(input_tensor)

    target_layer = model.get_layer(layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    inputs = tf.convert_to_tensor(normalized_input)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, class_index]

    gradients = tape.gradient(class_channel, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if float(max_value) > 0:
        heatmap = heatmap / max_value

    return GradCAMResult(
        heatmap=heatmap.numpy(),
        normalized_input=normalized_input[0],
        predicted_index=int(class_index),
        predicted_score=float(predictions[0, class_index]),
        target_layer=layer_name,
    )


def overlay_heatmap(
    spatial_channel: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a Grad-CAM heatmap with the spatial channel for visualization."""
    spatial_channel = np.asarray(spatial_channel, dtype=np.float32)
    heatmap = np.asarray(heatmap, dtype=np.float32)

    if spatial_channel.ndim == 3 and spatial_channel.shape[-1] == 1:
        spatial_channel = np.squeeze(spatial_channel, axis=-1)
    if spatial_channel.ndim != 2 or heatmap.ndim != 2:
        raise ValueError("overlay_heatmap expects 2D spatial and heatmap arrays.")

    if spatial_channel.shape != heatmap.shape:
        heatmap = tf.image.resize(heatmap[..., np.newaxis], spatial_channel.shape, method="bilinear").numpy()
        heatmap = np.squeeze(heatmap, axis=-1)

    spatial_min = spatial_channel.min()
    spatial_max = spatial_channel.max()
    if spatial_max > spatial_min:
        spatial_norm = (spatial_channel - spatial_min) / (spatial_max - spatial_min)
    else:
        spatial_norm = np.zeros_like(spatial_channel)

    heatmap = np.clip(heatmap, 0.0, 1.0)
    overlay = np.stack([spatial_norm, spatial_norm, spatial_norm], axis=-1)
    overlay[..., 0] = np.clip(overlay[..., 0] + alpha * heatmap, 0.0, 1.0)
    overlay[..., 1] = np.clip(overlay[..., 1] * (1.0 - 0.5 * alpha), 0.0, 1.0)
    overlay[..., 2] = np.clip(overlay[..., 2] * (1.0 - alpha), 0.0, 1.0)
    return overlay
