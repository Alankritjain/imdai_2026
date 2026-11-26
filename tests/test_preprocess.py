import numpy as np
import pytest
from PIL import Image

from src.preprocess import adjust_brightness, crop_image, flip_image, load_image, preprocess_image


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_output_shape_rgb_type_and_range(mode):
    processed = preprocess_image(Image.new(mode, (320, 160), 128))
    assert processed.shape == (80, 160, 3)
    assert processed.dtype == np.float32
    assert 0 <= processed.min() <= processed.max() <= 1


def test_crop_removes_expected_regions():
    pixels = np.zeros((160, 320, 3), dtype=np.uint8)
    pixels[:56] = (255, 0, 0)
    pixels[56:144] = (0, 255, 0)
    pixels[144:] = (0, 0, 255)
    image = Image.fromarray(pixels)
    assert crop_image(image).size == (320, 88)
    assert np.allclose(preprocess_image(image), [0, 1, 0])


@pytest.mark.parametrize("top,bottom", [(0.6, 0.5), (-0.1, 0), (float("nan"), 0), (1, 0)])
def test_invalid_crop_fractions(top, bottom):
    with pytest.raises(ValueError, match="Crop fractions"):
        crop_image(Image.new("RGB", (320, 160)), top, bottom)


def test_tiny_image_cannot_have_empty_crop():
    with pytest.raises(ValueError, match="empty"):
        crop_image(Image.new("RGB", (1, 1)))


def test_flip_reverses_pixels_and_steering():
    pixels = np.zeros((10, 20, 3), dtype=np.uint8)
    pixels[:, :10] = 255
    flipped, steering = flip_image(Image.fromarray(pixels), 0.4)
    assert steering == -0.4
    assert np.array_equal(np.asarray(flipped), pixels[:, ::-1])


def test_brightness_and_disk_loading(tmp_path):
    path = tmp_path / "fixture.png"
    Image.new("L", (20, 20), 100).save(path)
    image = load_image(path)
    assert image.mode == "RGB"
    assert np.asarray(adjust_brightness(image, 0.5)).mean() == 50
    with pytest.raises(ValueError):
        adjust_brightness(image, -1)
