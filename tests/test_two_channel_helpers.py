import numpy as np

from spatial_spectral_fft_cnn.model import compute_fft_channel, stack_spatial_frequency_channels


def test_stack_spatial_frequency_channels_creates_two_channel_tensor():
    sample = np.ones((8, 8, 1), dtype=np.float32)
    combined = stack_spatial_frequency_channels(sample)

    assert combined.shape == (8, 8, 2)
    np.testing.assert_allclose(combined[..., 0], sample[..., 0])
    assert np.all(combined[..., 1] >= 0.0)


def test_compute_fft_channel_rejects_non_single_channel_input():
    bad_sample = np.ones((8, 8, 2), dtype=np.float32)

    try:
        compute_fft_channel(bad_sample)
    except ValueError as exc:
        assert "exactly one channel" in str(exc)
    else:
        raise AssertionError("Expected ValueError for multi-channel FFT input")
