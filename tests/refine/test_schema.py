"""Unit tests for the `schema` module."""

import pydantic
import pytest

from mcd import dtypes
from mcd.change_detection import label, result
from mcd.refine import schema


class TestChangePoints:
    """Test suite for the `ChangePoints` class."""

    def test_serialize_model(self) -> None:
        """Test the `_serialize_model()` method."""
        change_point_list = [
            schema.ChangePoint(
                image_id="image_0",
                change=result.Change.ADDED,
                point=dtypes.Point(x=0.0, y=0.0, z=0.0),
                pixel=dtypes.Pixel(x=0, y=0),
            ),
            schema.ChangePoint(
                image_id="image_1",
                change=result.Change.REMOVED,
                point=dtypes.Point(x=1.0, y=1.0, z=1.0),
                pixel=dtypes.Pixel(x=1, y=1),
            ),
        ]
        change_points = schema.ChangePoints(root=change_point_list)
        assert change_points._serialize_model() == change_point_list


def _get_example_label_info_3d() -> schema.LabelInfo3d:
    """Get an example `LabelInfo3d` instance."""
    return schema.LabelInfo3d(
        label_id=0,
        pixel=dtypes.Pixel(x=0, y=0),
        point=dtypes.Point(x=0.0, y=0.0, z=0.0),
    )


class TestLabelInfo3d:
    """Test suite for the `LabelInfo3d` class."""

    class TestCheckIfLabelIsProvided:
        """Test suite for the `_check_if_label_is_provided()` method."""

        def test_check_if_label_is_provided_normal(self) -> None:
            """Test the `_check_if_label_is_provided()` method with normal behavior."""
            try:
                _get_example_label_info_3d()
            except pydantic.ValidationError:
                pytest.fail("Validation failed unexpectedly.")

        def test_check_if_label_is_provided_validation_error(self) -> None:
            """Test the `_check_if_label_is_provided()` method with validation error."""
            with pytest.raises(pydantic.ValidationError):
                schema.LabelInfo3d(
                    pixel=dtypes.Pixel(x=0, y=0),
                    point=dtypes.Point(x=0.0, y=0.0, z=0.0),
                )

    class TestLabel:
        """Test suite for the `label` property."""

        @pytest.mark.parametrize(
            ("label_id", "label_name", "expected"),
            [(0, None, 0), (None, "label_0", "label_0"), (0, "label_0", "label_0")],
        )
        def test_label_normal(
            self,
            label_id: label.LabelId | None,
            label_name: label.LabelName | None,
            expected: label.LabelId | label.LabelName,
        ) -> None:
            """Test the `label` property with normal behavior."""
            label_info = schema.LabelInfo3d(
                label_id=label_id,
                label_name=label_name,
                pixel=dtypes.Pixel(x=0, y=0),
                point=dtypes.Point(x=0.0, y=0.0, z=0.0),
            )
            assert label_info.label == expected

        def test_label_value_error(self) -> None:
            """Test the `label` property with a value error."""
            label_info = schema.LabelInfo3d.model_construct(
                pixel=dtypes.Pixel(x=0, y=0), point=dtypes.Point(x=0.0, y=0.0, z=0.0)
            )
            with pytest.raises(ValueError, match="Label ID or name must be provided."):
                _ = label_info.label


class TestSinglePairResult3d:
    """Test suite for the `SinglePairResult3d` class."""

    def test_serialize_model(self) -> None:
        """Test the `_serialize_model()` method."""
        added_item, removed_item, unchanged_item = [
            _get_example_label_info_3d() for _ in range(3)
        ]
        single_pair_result = schema.SinglePairResult3d(
            added={added_item}, removed={removed_item}, unchanged={unchanged_item}
        )
        assert single_pair_result._serialize_model() == {
            "added": [added_item],
            "removed": [removed_item],
            "unchanged": [unchanged_item],
        }


class TestChangeDetection3dResults:
    """Test suite for the `ChangeDetection3dResults` class."""

    def test_serialize_model(self) -> None:
        """Test the `_serialize_model()` method."""
        result_3d = schema.SinglePairResult3d(
            added={_get_example_label_info_3d()},
            removed={_get_example_label_info_3d()},
            unchanged={_get_example_label_info_3d()},
        )
        change_detection_results = schema.ChangeDetection3dResults(
            root={"image_0": result_3d}
        )
        assert change_detection_results._serialize_model() == {"image_0": result_3d}

    def test_items(self) -> None:
        """Test the `items()` method."""
        result_3d = schema.SinglePairResult3d(
            added={_get_example_label_info_3d()},
            removed={_get_example_label_info_3d()},
            unchanged={_get_example_label_info_3d()},
        )
        change_detection_results = schema.ChangeDetection3dResults(
            root={"image_0": result_3d}
        )

        assert dict(change_detection_results.items()) == {"image_0": result_3d}
