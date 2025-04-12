"""Unit tests for the `result` module."""

import pytest

from mcd.change_detection import result


class TestResult:
    """Test suite for the `result` module."""

    @pytest.mark.parametrize(
        ("change", "expected_str"),
        [
            (result.Change.ADDED, "added"),
            (result.Change.REMOVED, "removed"),
            (result.Change.UNCHANGED, "unchanged"),
        ],
    )
    def test__str__(self, change: result.Change, expected_str: str) -> None:
        """Test the `__str__()` method."""
        assert str(change) == expected_str

    @pytest.mark.parametrize(
        ("change", "expected_color"),
        [
            (result.Change.ADDED, "green"),
            (result.Change.REMOVED, "red"),
            (result.Change.UNCHANGED, "black"),
        ],
    )
    def test_color(self, change: result.Change, expected_color: str) -> None:
        """Test the `color` property."""
        assert change.color == expected_color
