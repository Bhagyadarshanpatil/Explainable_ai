# Spatial + Spectral FFT CNN with Grad-CAM

This repository packages the core CNN idea from `Bhagyadarshanpatil/Spatial_and_Spectral_FFT_CNN` into reusable Python helpers and adds **Grad-CAM** support for a model that consumes a **two-channel input tensor**:

1. channel 0 = spatial-domain grayscale image
2. channel 1 = frequency-domain log-magnitude FFT image

## What's included

- `spatial_spectral_fft_cnn/model.py`
  - `stack_spatial_frequency_channels(...)` to build the two-channel tensor from a grayscale image.
  - `create_two_channel_model(...)` to define the CNN with a named final convolution layer for explainability.
- `spatial_spectral_fft_cnn/gradcam.py`
  - `compute_gradcam(...)` to produce a Grad-CAM heatmap.
  - `overlay_heatmap(...)` to blend the heatmap with the spatial channel for display.
- `tests/test_two_channel_helpers.py`
  - lightweight regression tests for the two-channel preprocessing helpers.

## Example usage

```python
import numpy as np
from spatial_spectral_fft_cnn import (
    compute_gradcam,
    create_two_channel_model,
    overlay_heatmap,
    stack_spatial_frequency_channels,
)

# spatial_image shape: (256, 256, 1)
spatial_image = np.random.rand(256, 256, 1).astype("float32")
combined_input = stack_spatial_frequency_channels(spatial_image)

model = create_two_channel_model(input_shape=(256, 256, 2), num_classes=5)
result = compute_gradcam(model, combined_input)
overlay = overlay_heatmap(combined_input[..., 0], result.heatmap)
```

## Notes for your project

- Grad-CAM should target the **last convolutional layer**, which is named `gradcam_target_conv` here.
- Because your model input is already a **two-channel spatial+frequency tensor**, the FFT preprocessing is kept **outside** the model architecture.
- If your training notebook already constructs the two channels upstream, feed that tensor directly into `create_two_channel_model(...)` and then call `compute_gradcam(...)` on the same tensor shape during inference.
