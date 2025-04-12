"""Test module for the `_array` module."""

import numpy as np
import pytest
from numpy._typing import _shape

from mcd.dtypes import _array


class TestCheckArrayShape:
    """Test suite for the `_check_array_shape()` function."""

    @pytest.mark.parametrize(("shape"), [((3,)), ((3, 3, 3)), ((3, 4))])
    def test_check_array_shape(self, shape: _shape._ShapeLike) -> None:
        """Test the `_check_array_shape()` function with a valid input."""
        random_generator = np.random.default_rng(0)
        array = random_generator.random(shape)
        try:
            _array._check_array_shape(array, shape=shape)
        except (TypeError, ValueError):
            pytest.fail("Exception raised unexpectedly.")

    @pytest.mark.parametrize(("shape"), [((3,)), ((3, 3, 3)), ((3, 4))])
    def test_check_array_shape_invalid_shape(self, shape: _shape._ShapeLike) -> None:
        """Test the `_check_array_shape()` function with an invalid shape."""
        random_generator = np.random.default_rng(0)
        array = random_generator.random(shape)
        with pytest.raises(ValueError, match="Input array must have shape") as exc_info:
            _array._check_array_shape(array, shape=(3, 3))
        assert str(exc_info.value) == "Input array must have shape (3, 3)."


class TestCheckBatchArrayShape:
    """Test suite for the `_check_batch_array_shape()` function."""

    @pytest.mark.parametrize(("shape"), [((3,)), ((3, 3, 3)), ((3, 4))])
    def test_check_batch_array_shape(self, shape: tuple[int, ...]) -> None:
        """Test the `_check_batch_array_shape()` function with a valid input."""
        random_generator = np.random.default_rng(0)
        first_dimension = random_generator.integers(1, 100)
        array = random_generator.random((first_dimension, *shape))
        try:
            _array._check_batch_array_shape(array, shape=shape)
        except (TypeError, ValueError):
            pytest.fail("Exception raised unexpectedly.")

    @pytest.mark.parametrize(("shape"), [((3,)), ((3, 3, 3)), ((3, 4))])
    def test_check_batch_array_shape_invalid_shape(
        self, shape: tuple[int, ...]
    ) -> None:
        """Test the `_check_batch_array_shape()` function with an invalid shape."""
        random_generator = np.random.default_rng(0)
        array = random_generator.random(shape)
        with pytest.raises(ValueError, match="Input array must have shape") as exc_info:
            _array._check_batch_array_shape(array, shape=shape)
        assert (
            str(exc_info.value) == f"Input array must have shape (batch_size, {shape})."
        )
