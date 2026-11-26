"""RGB conversion, percentage cropping, resizing and [0, 1] normalization."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from config import CROP_BOTTOM, CROP_TOP, IMAGE_HEIGHT, IMAGE_WIDTH


def load_image(image_path):
    """Read one image and close its file handle immediately."""
    with Image.open(Path(image_path)) as image:
        return image.convert("RGB")


def crop_image(image, crop_top=CROP_TOP, crop_bottom=CROP_BOTTOM):
    """Remove sky and dashboard using fractions of the source image height."""
    if not (0 <= crop_top < 1 and 0 <= crop_bottom < 1 and crop_top + crop_bottom < 1):
        raise ValueError("Crop fractions must be nonnegative and add up to less than 1.")
    image = image.convert("RGB")
    top_pixel = int(image.height * crop_top)
    bottom_pixel = int(image.height * (1 - crop_bottom))
    if image.width < 1 or bottom_pixel <= top_pixel:
        raise ValueError("Crop would produce an empty image; reduce the crop fractions.")
    return image.crop((0, top_pixel, image.width, bottom_pixel))


def preprocess_image(image, crop_top=CROP_TOP, crop_bottom=CROP_BOTTOM):
    """Return an RGB float32 array of shape (80, 160, 3), normalized to [0, 1]."""
    cropped_image = crop_image(image, crop_top, crop_bottom)
    resized_image = cropped_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
    return np.asarray(resized_image, dtype=np.float32) / 255.0


def flip_image(image, steering_value):
    """Training-only augmentation: a mirror reverses the steering sign."""
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), -float(steering_value)


def adjust_brightness(image, factor=1.0):
    """Training-only augmentation; a factor of 1 leaves brightness unchanged."""
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("Brightness factor must be finite and nonnegative.")
    return ImageEnhance.Brightness(image.convert("RGB")).enhance(factor)
