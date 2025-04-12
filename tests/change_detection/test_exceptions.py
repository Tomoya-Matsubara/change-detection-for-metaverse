"""Unit tests for the `exceptions` module."""

import pytest

from mcd.change_detection import exceptions


class TestLabelInconsistentError:
    """Test suite for the `LabelInconsistentError` class."""

    def test_str(self) -> None:
        """Test the `__str__()` method."""
        with pytest.raises(exceptions.LabelInconsistentError) as exec_info:
            raise exceptions.LabelInconsistentError(
                label_type="label_id", label1=1, label2=None
            )
        assert str(exec_info.value) == (
            "Two labels must be labeled in the same way: 1 and None. If label_id is "
            "provided for one label, it must be provided for the other label as well."
        )


class TestTooManyDatasetsError:
    """Test suite for the `TooManyDatasetsError` class."""

    def test_str(self) -> None:
        """Test the `__str__()` method."""
        with pytest.raises(exceptions.TooManyDatasetsError) as exec_info:
            raise exceptions.TooManyDatasetsError
        assert str(exec_info.value) == (
            "Failed to detect the 'after' dataset because more than two datasets are "
            "present in the directory."
        )


class TestAfterDatasetNotFoundError:
    """Test suite for the `AfterDatasetNotFoundError` class."""

    def test_str(self) -> None:
        """Test the `__str__()` method."""
        with pytest.raises(exceptions.AfterDatasetNotFoundError) as exec_info:
            raise exceptions.AfterDatasetNotFoundError
        assert str(exec_info.value) == (
            "Failed to detect the 'after' dataset because there is only a single "
            "dataset present in the directory."
        )
