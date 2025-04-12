"""Unit tests for the `label` module."""

import pytest

from mcd.change_detection import exceptions, label


class TestBoundingBox:
    """Test suite for the `BoundingBox` class."""

    @pytest.fixture
    def bounding_box(self) -> label.BoundingBox:
        """Create a bounding box."""
        return label.BoundingBox(x=0.5, y=0.5, width=1.0, height=0.5)

    def test_area(self, bounding_box: label.BoundingBox) -> None:
        """Test the `area` property."""
        assert bounding_box.area == 1.0 * 0.5

    class TestXYXY:
        """Test suite for the `xyxy` property."""

        def test_xyxy_normal(self, bounding_box: label.BoundingBox) -> None:
            """Test the `xyxy` property."""
            assert bounding_box.xyxy == (0.0, 0.25, 1.0, 0.75)

        def test_xyxy_lower_bound_warning(self) -> None:
            """Test the `xyxy` property with lower bound warning."""
            with pytest.warns() as record:
                _ = label.BoundingBox(x=0, y=0, width=1, height=1).xyxy

            expected_warning_targets = [("x1", -0.5), ("y1", -0.5)]
            for i, record_item in enumerate(record):
                target, value = expected_warning_targets[i]
                assert (
                    str(record_item.message)
                    == f"Bounding box {target} is less than 0 ({value})."
                )

        def test_xyxy_upper_bound_warning(self) -> None:
            """Test the `xyxy` property with upper bound warning."""
            with pytest.warns() as record:
                _ = label.BoundingBox(x=1, y=1, width=1, height=1).xyxy

            expected_warning_targets = [("x2", 1.5), ("y2", 1.5)]
            for i, record_item in enumerate(record):
                target, value = expected_warning_targets[i]
                assert (
                    str(record_item.message)
                    == f"Bounding box {target} is greater than 1 ({value})."
                )

    @pytest.mark.parametrize(
        ("other", "expected"),
        [
            (label.BoundingBox(x=0.5, y=0.5, width=1.0, height=0.5), 1.0),
            (label.BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5), 0.2),
            (label.BoundingBox(x=0.5, y=0.5, width=1.0, height=1.0), 0.5),
            (label.BoundingBox(x=0.125, y=0.125, width=0.125, height=0.125), 0),
        ],
    )
    def test_compute_iou(
        self, bounding_box: label.BoundingBox, other: label.BoundingBox, expected: float
    ) -> None:
        """Test the `compute_iou()` method."""
        assert bounding_box.compute_iou(other) == expected


class TestLabelInfo:
    """Test suite for the `LabelInfo` class."""

    @pytest.fixture
    def bounding_box(self) -> label.BoundingBox:
        """Create a bounding box."""
        return label.BoundingBox(x=0.5, y=0.5, width=1.0, height=0.5)

    class TestCheckIfLabelIsProvided:
        """Test suite for the `_check_if_label_is_provided()` method."""

        def test_check_if_label_is_provided_for_label_id(
            self, bounding_box: label.BoundingBox
        ) -> None:
            """Test the `_check_if_label_is_provided()` method for `label_id`."""
            try:
                label.LabelInfo(label_id=1, bounding_box=bounding_box)
            except ValueError:
                pytest.fail("ValueError raised unexpectedly.")

        def test_check_if_label_is_provided_for_label_name(
            self, bounding_box: label.BoundingBox
        ) -> None:
            """Test the `_check_if_label_is_provided()` method for `label_name`."""
            try:
                label.LabelInfo(label_name="label", bounding_box=bounding_box)
            except ValueError:
                pytest.fail("ValueError raised unexpectedly.")

        def test_check_if_label_is_provided_invalid(
            self, bounding_box: label.BoundingBox
        ) -> None:
            """Test the `_check_if_label_is_provided()` method for invalid input."""
            with pytest.raises(
                ValueError, match="Either `label_id` or `label_name` must be provided."
            ):
                label.LabelInfo(bounding_box=bounding_box)

    class TestIsLabelSame:
        """Test suite for the `is_label_same()` method."""

        @pytest.mark.parametrize(("label2_id", "expected"), [(1, True), (2, False)])
        def test_is_label_same_for_label_id(
            self, bounding_box: label.BoundingBox, label2_id: int, *, expected: bool
        ) -> None:
            """Test the `is_label_same()` method for `label_id`."""
            label1 = label.LabelInfo(label_id=1, bounding_box=bounding_box)
            label2 = label.LabelInfo(label_id=label2_id, bounding_box=bounding_box)
            assert label1.is_label_same(label2) == expected

        @pytest.mark.parametrize(
            ("label2_name", "expected"), [("label", True), ("name", False)]
        )
        def test_is_label_same_for_label_name(
            self, bounding_box: label.BoundingBox, label2_name: str, *, expected: bool
        ) -> None:
            """Test the `is_label_same()` method for `label_name`."""
            label1 = label.LabelInfo(label_name="label", bounding_box=bounding_box)
            label2 = label.LabelInfo(label_name=label2_name, bounding_box=bounding_box)
            assert label1.is_label_same(label2) == expected

        def test_is_label_same_for_invalid_label_id(
            self, bounding_box: label.BoundingBox
        ) -> None:
            """Test the `is_label_same()` method for invalid `label_id`."""
            label1 = label.LabelInfo(label_id=1, bounding_box=bounding_box)
            label2 = label.LabelInfo(label_name="label", bounding_box=bounding_box)
            with pytest.raises(exceptions.LabelInconsistentError):
                label1.is_label_same(label2)

        def test_is_label_same_for_invalid_label_name(
            self, bounding_box: label.BoundingBox
        ) -> None:
            """Test the `is_label_same()` method for invalid `label_name`."""
            label1 = label.LabelInfo(label_name="label", bounding_box=bounding_box)
            label2 = label.LabelInfo(label_id=1, bounding_box=bounding_box)
            with pytest.raises(exceptions.LabelInconsistentError):
                label1.is_label_same(label2)
