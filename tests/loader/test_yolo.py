"""Unit tests for the `yolo` module."""

import pathlib
import typing

from mcd.change_detection import label
from mcd.loader import yolo

_TEST_DATA_DIRECTORY: typing.Final[pathlib.Path] = (
    pathlib.Path(__file__).parents[1] / "test_data"
)
"""Path to the test data directory."""


class TestYoloObjectDetectionDataLoader:
    """Test suite for the `YoloObjectDetectionDataLoader` class."""

    def test_get_images_path(self) -> None:
        """Test the `_get_images_path()` method."""
        dataset_path = _TEST_DATA_DIRECTORY / "dataset1"
        images_path = list(
            yolo.YoloObjectDetectionDataLoader().get_images_path(dataset_path)
        )
        assert images_path == [dataset_path / "image_123" / "image_123.jpg"]

    def test_get_labels_path(self) -> None:
        """Test the `_get_labels_path()` method."""
        labels_path = list(
            yolo.YoloObjectDetectionDataLoader().get_labels_path(
                _TEST_DATA_DIRECTORY / "dataset1"
            )
        )
        assert labels_path == [
            _TEST_DATA_DIRECTORY / "dataset1" / "image_123" / "labels" / "image_123.txt"
        ]

    def test_read_labels(self) -> None:
        """Test the `_read_labels()` method."""
        change_detector = yolo.YoloObjectDetectionDataLoader()
        labels = change_detector.read_labels(
            _TEST_DATA_DIRECTORY / "dataset1" / "image_123" / "labels" / "image_123.txt"
        )
        expected_labels = [
            label.LabelInfo(
                label_id=0,
                bounding_box=label.BoundingBox(
                    x=0.159226, y=0.570961, width=0.316666, height=0.44264
                ),
            ),
            label.LabelInfo(
                label_id=0,
                bounding_box=label.BoundingBox(
                    x=0.0817239, y=0.675018, width=0.163448, height=0.51431
                ),
            ),
        ]
        assert labels == expected_labels
