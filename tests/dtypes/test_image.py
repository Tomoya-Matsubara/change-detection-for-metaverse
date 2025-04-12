"""Test module for the `_image` module."""

import numpy as np
import pytest
from numpy._typing import _shape

from mcd.dtypes import _image


class TestValidateGrayscaleImage:
    """Test suite for the `_validate_grayscale_image()` function."""

    def test_validate_grayscale_image(self) -> None:
        """Test the `_validate_grayscale_image()` function with a valid input."""
        random_generator = np.random.default_rng(0)
        image = random_generator.random((10, 10))
        try:
            _image._validate_grayscale_image(image)
        except (TypeError, ValueError):
            pytest.fail("Exception raised unexpectedly.")

    def test_validate_grayscale_image_invalid_shape(self) -> None:
        """Test the `_validate_grayscale_image()` function with an invalid shape."""
        random_generator = np.random.default_rng(0)
        image = random_generator.random((10, 10, 3))
        with pytest.raises(ValueError, match="Input image must be a 2D array."):
            _image._validate_grayscale_image(image)


class TestValidateColoredImage:
    """Test suite for the `_validate_colored_image()` function."""

    def test_validate_colored_image(self) -> None:
        """Test the `_validate_colored_image()` function with a valid input."""
        random_generator = np.random.default_rng(0)
        image = random_generator.random((10, 10, 3))
        try:
            _image._validate_colored_image(image)
        except (TypeError, ValueError):
            pytest.fail("Exception raised unexpectedly.")

    @pytest.mark.parametrize(("shape"), [((10, 10)), ((10, 10, 10, 3)), ((10, 10, 4))])
    def test_validate_colored_image_invalid_shape(
        self, shape: _shape._ShapeLike
    ) -> None:
        """Test the `_validate_colored_image()` function with an invalid shape."""
        random_generator = np.random.default_rng(0)
        image = random_generator.random(shape)
        with pytest.raises(
            ValueError, match="Input image must be a 3D array with 3 channels."
        ):
            _image._validate_colored_image(image)
