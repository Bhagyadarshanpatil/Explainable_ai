"""Model helpers for a two-channel spatial + frequency CNN."""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
except ImportError:  # pragma: no cover - exercised only in environments without TensorFlow
    tf = None
    Model = object
    Conv2D = Dense = Flatten = Input = MaxPooling2D = None


LAST_CONV_LAYER_NAME = "gradcam_target_conv"


def _require_tensorflow() -> None:
    if tf is None:
        raise ImportError(
            "TensorFlow is required for this project. Install tensorflow before using the model helpers."
        )


def compute_fft_channel(spatial_tensor: np.ndarray) -> np.ndarray:
    """Create the log-magnitude FFT channel for a grayscale image batch or sample."""
    spatial_tensor = np.asarray(spatial_tensor, dtype=np.float32)

    if spatial_tensor.ndim == 2:
        spatial_tensor = spatial_tensor[..., np.newaxis]
    if spatial_tensor.ndim not in (3, 4):
        raise ValueError("Expected a 2D image, 3D sample tensor, or 4D batch tensor.")
    if spatial_tensor.shape[-1] != 1:
        raise ValueError("The spatial tensor must contain exactly one channel before FFT expansion.")

    squeezed = np.squeeze(spatial_tensor, axis=-1)
    fft = np.fft.fft2(squeezed, axes=(-2, -1))
    shifted = np.fft.fftshift(fft, axes=(-2, -1))
    magnitude = np.log1p(np.abs(shifted)).astype(np.float32)
    return magnitude[..., np.newaxis]


def stack_spatial_frequency_channels(spatial_tensor: np.ndarray) -> np.ndarray:
    """Build the expected two-channel tensor: [spatial, log-magnitude FFT]."""
    spatial_tensor = np.asarray(spatial_tensor, dtype=np.float32)
    if spatial_tensor.ndim == 2:
        spatial_tensor = spatial_tensor[..., np.newaxis]
    fft_channel = compute_fft_channel(spatial_tensor)
    return np.concatenate([spatial_tensor, fft_channel], axis=-1)


def create_two_channel_model(
    input_shape: Tuple[int, int, int] = (256, 256, 2),
    num_classes: int = 5,
):
    """Recreate the repository CNN, but consume an explicit two-channel input tensor.

    The first channel is the spatial grayscale image and the second channel is the
    frequency-domain representation, which keeps the model compatible with Grad-CAM.
    """
    _require_tensorflow()

    if input_shape[-1] != 2:
        raise ValueError("Expected a two-channel input shaped as [spatial, frequency].")

    inputs = Input(shape=input_shape, name="spatial_frequency_input")
    x = Conv2D(16, (3, 3), activation="relu", name="conv_1")(inputs)
    x = MaxPooling2D(name="pool_1")(x)
    x = Conv2D(32, (3, 3), activation="relu", name="conv_2")(x)
    x = MaxPooling2D(name="pool_2")(x)
    x = Conv2D(16, (3, 3), activation="relu", name=LAST_CONV_LAYER_NAME)(x)
    x = MaxPooling2D(name="pool_3")(x)
    x = Flatten(name="flatten_features")(x)
    x = Dense(256, activation="relu", name="dense_features")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="spatial_spectral_fft_cnn")
