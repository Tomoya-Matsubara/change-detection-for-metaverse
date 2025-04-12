"""Unit tests for the `base` module."""

import pathlib
import tempfile

import pytest

from mcd.change_detection import exceptions
from mcd.loader import base


class TestDataLoaderBase:
    """Test suite for the `_DataLoaderBase` class."""

    class TestGetAfterDatasetPath:
        """Tests for the `get_after_dataset_path()` method."""

        def test_get_after_dataset_path_normal(self) -> None:
            """Test the `get_after_dataset_path()` method with a normal case."""
            with tempfile.TemporaryDirectory() as datasets_directory_str:
                datasets_directory = pathlib.Path(datasets_directory_str)
                pathlib.Path(datasets_directory / "before").mkdir()
                after_dataset_path = pathlib.Path(datasets_directory) / "after"
                after_dataset_path.mkdir()

                after_dataset_path_actual = (
                    base._DataLoaderBase().get_after_dataset_path(
                        datasets_directory, "before"
                    )
                )
                assert after_dataset_path_actual == after_dataset_path

        def test_get_after_dataset_path_too_many_datasets(self) -> None:
            """Test the `get_after_dataset_path()` method with too many datasets."""
            with tempfile.TemporaryDirectory() as datasets_directory_str:
                datasets_directory = pathlib.Path(datasets_directory_str)
                pathlib.Path(datasets_directory / "before").mkdir()
                pathlib.Path(datasets_directory / "after1").mkdir()
                pathlib.Path(datasets_directory / "after2").mkdir()

                with pytest.raises(exceptions.TooManyDatasetsError):
                    base._DataLoaderBase().get_after_dataset_path(
                        datasets_directory, "before"
                    )

        def test_get_after_dataset_path_no_after_dataset(self) -> None:
            """Test the `get_after_dataset_path()` method with no after dataset."""
            with tempfile.TemporaryDirectory() as datasets_directory_str:
                datasets_directory = pathlib.Path(datasets_directory_str)
                (pathlib.Path(datasets_directory) / "before").mkdir()
                pathlib.Path(datasets_directory / "text.txt").write_text("placeholder")

                with pytest.raises(exceptions.AfterDatasetNotFoundError):
                    base._DataLoaderBase().get_after_dataset_path(
                        datasets_directory, "before"
                    )
