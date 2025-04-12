"""Unit tests for the `detector` module."""

import json
import pathlib
import tempfile
import typing

import pytest

from mcd.change_detection import detector, label, result
from mcd.loader import yolo

_TEST_DATA_DIRECTORY: typing.Final[pathlib.Path] = (
    pathlib.Path(__file__).parents[1] / "test_data"
)
"""Path to the test data directory."""


class TestChangeDetectorBase:
    """Tests for the ChangeDetectorBase class."""

    def test_run(self) -> None:
        """Test the `run()` method."""
        datasets_path = _TEST_DATA_DIRECTORY / "datasets"
        before_label_path = (
            datasets_path / "before" / "image_0" / "labels" / "image_0.txt"
        )
        after_label_path = (
            datasets_path / "after" / "image_0" / "labels" / "image_0.txt"
        )
        actual_result = detector.ChangeDetector().run(
            before_label_path, after_label_path, yolo.YoloObjectDetectionDataLoader()
        )

        assert actual_result == result.SinglePairResult(
            added={
                label.LabelInfo(
                    label_id=0,
                    bounding_box=label.BoundingBox(x=0.9, y=0.9, width=0.1, height=0.1),
                )
            },
            removed={
                label.LabelInfo(
                    label_id=0,
                    bounding_box=label.BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2),
                )
            },
            unchanged={
                label.LabelInfo(
                    label_id=1,
                    bounding_box=label.BoundingBox(x=0.8, y=0.8, width=0.4, height=0.4),
                )
            },
        )

    class TestRunAll:
        """Test suite for the `run_all()` method."""

        def test_run_all_file_not_found_error(self) -> None:
            """Test the `run_all()` method when a file is not found."""
            change_detector = detector.ChangeDetector()
            with pytest.raises(FileNotFoundError):
                change_detector.run_all(
                    _TEST_DATA_DIRECTORY / "datasets" / "before",
                    before_name="not_found",
                    loader=yolo.YoloObjectDetectionDataLoader(),
                )

        def test_run_all(self) -> None:
            """Test the `run_all()` method for a successful run."""
            change_detector = detector.ChangeDetector()
            actual_result = change_detector.run_all(
                _TEST_DATA_DIRECTORY / "datasets",
                before_name="before",
                loader=yolo.YoloObjectDetectionDataLoader(),
            )

            assert actual_result == result.ChangeDetectionResults(
                result={
                    "image_0": result.SinglePairResult(
                        added={
                            label.LabelInfo(
                                label_id=0,
                                bounding_box=label.BoundingBox(
                                    x=0.9, y=0.9, width=0.1, height=0.1
                                ),
                            )
                        },
                        removed={
                            label.LabelInfo(
                                label_id=0,
                                bounding_box=label.BoundingBox(
                                    x=0.1, y=0.1, width=0.2, height=0.2
                                ),
                            )
                        },
                        unchanged={
                            label.LabelInfo(
                                label_id=1,
                                bounding_box=label.BoundingBox(
                                    x=0.8, y=0.8, width=0.4, height=0.4
                                ),
                            )
                        },
                    )
                },
                image_height=1920,
                image_width=1440,
            )

    @pytest.mark.parametrize(
        ("description", "label2", "expected"),
        [
            (
                "Labels are different",
                label.LabelInfo(
                    label_id=1,
                    bounding_box=label.BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
                ),
                False,
            ),
            (
                "Intersected",
                label.LabelInfo(
                    label_id=0,
                    bounding_box=label.BoundingBox(
                        x=0.25, y=0.25, width=0.5, height=0.5
                    ),
                ),
                True,
            ),
            (
                "Not intersected",
                label.LabelInfo(
                    label_id=0,
                    bounding_box=label.BoundingBox(
                        x=0.125, y=0.125, width=0.125, height=0.125
                    ),
                ),
                False,
            ),
        ],
    )
    def test_check_intersection(
        self, description: str, label2: label.LabelInfo, *, expected: bool
    ) -> None:
        """Test the `check_intersection()` method."""
        change_detector = detector.ChangeDetector()
        label1 = label.LabelInfo(
            label_id=0,
            bounding_box=label.BoundingBox(x=0.5, y=0.5, width=1.0, height=0.5),
        )
        assert change_detector._check_intersection(label1, label2) == expected, (
            description
        )

    def test_compare_bounding_boxes(self) -> None:
        """Test the `compare_bounding_boxes()` method."""
        before_labels = [
            # Expected to be removed
            label.LabelInfo(
                label_id=0,
                bounding_box=label.BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2),
            ),
            # Expected to intersect with the after_labels[1]
            label.LabelInfo(
                label_id=1,
                bounding_box=label.BoundingBox(x=0.8, y=0.8, width=0.4, height=0.4),
            ),
        ]
        after_labels = [
            # Expected to be added
            label.LabelInfo(
                label_id=0,
                bounding_box=label.BoundingBox(x=0.9, y=0.9, width=0.1, height=0.1),
            ),
            # Expected to intersect with the before_labels[1]
            label.LabelInfo(
                label_id=1,
                bounding_box=label.BoundingBox(x=0.7, y=0.7, width=0.1, height=0.1),
            ),
        ]

        actual_result = detector.ChangeDetector()._compare_bounding_boxes(
            before_labels, after_labels
        )

        expected_result = result.SinglePairResult(
            added={after_labels[0]},
            removed={before_labels[0]},
            unchanged={before_labels[1]},
        )
        assert actual_result == expected_result

    def test_export_result(self) -> None:
        """Test the `export_result()` method."""
        bounding_box = label.BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        results = result.ChangeDetectionResults(
            result={
                "image1": result.SinglePairResult(
                    added={label.LabelInfo(label_id=0, bounding_box=bounding_box)},
                    removed=set(),
                    unchanged=set(),
                ),
                "image2": result.SinglePairResult(
                    added=set(),
                    removed={label.LabelInfo(label_id=1, bounding_box=bounding_box)},
                    unchanged=set(),
                ),
                "image3": result.SinglePairResult(
                    added=set(),
                    removed=set(),
                    unchanged={label.LabelInfo(label_id=2, bounding_box=bounding_box)},
                ),
            },
            image_height=100,
            image_width=100,
        )
        with tempfile.TemporaryDirectory() as output_directory_str:
            output_directory = pathlib.Path(output_directory_str)
            detector.ChangeDetector()._export_result(results, output_directory)

            result_path = output_directory / "change_detection_result.json"
            with result_path.open("r") as file:
                actual_result = result.ChangeDetectionResults.model_validate(
                    json.load(file)
                )

        assert actual_result == results
