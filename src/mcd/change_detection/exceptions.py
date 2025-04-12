"""Exceptions for the change detection module."""

import typing


class LabelInconsistentError(Exception):
    """Exception for inconsistent labels.

    Parameters
    ----------
    label_type : {"label_id", "label_name"}
        The type of the label which is inconsistent.
    label1 : int | str | None
        The first label.
    label2 : int | str | None
        The second label.
    """

    def __init__(
        self,
        label_type: typing.Literal["label_id", "label_name"],
        label1: int | str | None,
        label2: int | str | None,
    ) -> None:
        self._label_type: typing.Final[typing.Literal["label_id", "label_name"]] = (
            label_type
        )
        """The type of the label which is inconsistent."""

        self._label1: typing.Final[int | str | None] = label1
        """The first label."""

        self._label2: typing.Final[int | str | None] = label2
        """The second label."""

    def __str__(self) -> str:
        """Return the error message."""
        return (
            f"Two labels must be labeled in the same way: {self._label1} and "
            f"{self._label2}. If {self._label_type} is provided for one label, it must "
            "be provided for the other label as well."
        )


class TooManyDatasetsError(Exception):
    """Exception for too many datasets in a directory for change detection."""

    def __str__(self) -> str:
        """Return the error message."""
        return (
            "Failed to detect the 'after' dataset because more than two datasets are "
            "present in the directory."
        )


class AfterDatasetNotFoundError(Exception):
    """Exception for not finding the 'after' dataset for change detection."""

    def __str__(self) -> str:
        """Return the error message."""
        return (
            "Failed to detect the 'after' dataset because there is only a single "
            "dataset present in the directory."
        )
