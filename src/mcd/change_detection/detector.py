"""Module for the change detection."""

import json
import logging
import pathlib
import typing

import cv2
from rich import progress

from mcd import log
from mcd.change_detection import label, result
from mcd.loader import base as loader_base

_LOGGER: typing.Final[logging.Logger] = log.setup_logger(__name__)
"""Logger for the module."""


class ChangeDetector:
    """Change detector.

    This class is responsible for detecting changes between two sets of object detection
    results.
    """

    def run(
        self,
        before_path: pathlib.Path,
        after_path: pathlib.Path,
        loader: loader_base.ObjectDetectionDataLoaderBase,
    ) -> result.SinglePairResult:
        """Run change detection on a single pair of images.

        Parameters
        ----------
        before_path : pathlib.Path
            Path to the label file of the first image.
        after_path : pathlib.Path
            Path to the label file of the second image.
        loader : ObjectDetectionDataLoaderBase
            Loader for the object detection results.

        Returns
        -------
        SingleResult
            Result of the change detection.
        """
        before_labels = loader.read_labels(before_path)
        after_labels = loader.read_labels(after_path)
        return self._compare_bounding_boxes(before_labels, after_labels)

    def run_all(
        self,
        datasets_path: pathlib.Path,
        loader: loader_base.ObjectDetectionDataLoaderBase,
        before_name: str = "before",
    ) -> result.ChangeDetectionResults:
        """Run change detection on all pairs of images in a directory.

        The results are stored in a JSON file in the same directory.

        Parameters
        ----------
        datasets_path : pathlib.Path
            Path to the directory containing the datasets. The directory should have
            exactly two subdirectories, one for the dataset before the change and one
            for the dataset after the change, whose structure should be as follows:

            ```
            datasets_path/
            ├── dataset_after/
            │   ├── image0/
            │   │   └── ...
            │   └── ...
            └── dataset_before/
                ├── image0/
                │   └── ...
                └── ...
            ```

            The contents under each image directory vary depending on the model.

        loader : ObjectDetectionDataLoaderBase
            Loader for the object detection results.
        before_name : str, default "before"
            Name of the dataset that represents the scene before the change. This should
            be the name of the directory containing the images before the change.

        Returns
        -------
        ChangeDetectionResults
            Results of the change detection.
        """
        if not (before_dataset_path := datasets_path / before_name).exists():
            raise FileNotFoundError(before_dataset_path)

        after_dataset_path = loader.get_after_dataset_path(datasets_path, before_name)

        before_image_path = next(loader.get_images_path(before_dataset_path))
        image = cv2.imread(before_image_path.as_posix())
        height, width = image.shape[:2]

        results = result.ChangeDetectionResults(
            result={}, image_height=height, image_width=width
        )
        before_dataset_items_path = list(loader.get_labels_path(before_dataset_path))
        with progress.Progress() as progress_bar:
            task = progress_bar.add_task(
                "[cyan]Detecting changes", total=len(before_dataset_items_path)
            )
            for before_item_path in before_dataset_items_path:
                image_id = before_item_path.stem
                after_item_path = after_dataset_path / before_item_path.relative_to(
                    before_dataset_path
                )

                progress_bar.update(
                    task,
                    advance=1,
                    description="[green]Detecting changes[/green] in "
                    f"{before_item_path.name}",
                )

                # Skip if the corresponding item in the 'after' dataset does not exist
                if not after_item_path.exists():
                    continue

                results.result[image_id] = self.run(
                    before_item_path, after_item_path, loader
                )

        self._export_result(results, datasets_path)
        return results

    def _check_intersection(
        self, label1: label.LabelInfo, label2: label.LabelInfo
    ) -> bool:
        """Check if two bounding boxes with the same label intersect.

        If they have different labels, they are considered to not intersect.

        Parameters
        ----------
        label1 : LabelInfo
            The first label.
        label2 : LabelInfo
            The second label.

        Returns
        -------
        bool
            True if the bounding boxes intersect, False otherwise.
        """
        if not label1.is_label_same(label2):
            return False

        return label1.bounding_box.compute_iou(label2.bounding_box) > 0

    def _compare_bounding_boxes(
        self, before_labels: list[label.LabelInfo], after_labels: list[label.LabelInfo]
    ) -> result.SinglePairResult:
        """Compare two bounding boxes to detect changes.

        Parameters
        ----------
        before_labels : list[LabelInfo]
            List of bounding boxes in the first image (before).
        after_labels : list[LabelInfo]
            List of bounding boxes in the second image (after).

        Returns
        -------
        SingleResult
            Result of the change detection.
        """
        detection_result = result.SinglePairResult(
            added=set(), removed=set(), unchanged=set()
        )
        intersected_after_labels: set[label.LabelInfo] = set()

        # Check if the bounding boxes in the first image (before) are present in the
        # second image (after). If they are not, they are considered removed.
        for before_label in before_labels:
            for after_label in after_labels:
                if not self._check_intersection(before_label, after_label):
                    continue
                detection_result.unchanged.add(before_label)
                intersected_after_labels.add(after_label)
            if before_label not in detection_result.unchanged:
                # If the loop completes without breaking, the bounding box is not
                # present in the second image. Therefore, it is considered removed.
                detection_result.removed.add(before_label)

        # Check for added bounding boxes
        for after_label in after_labels:
            if after_label not in intersected_after_labels:
                detection_result.added.add(after_label)

        return detection_result

    def _export_result(
        self, results: result.ChangeDetectionResults, datasets_path: pathlib.Path
    ) -> None:
        """Export the change detection result.

        Parameters
        ----------
        results : ChangeDetectionResults
            Result of the change detection.
        datasets_path : pathlib.Path
            Path to the directory containing the datasets.
        """
        result_path = datasets_path / "change_detection_result.json"
        with result_path.open("w") as f:
            json.dump(results.model_dump(), f, indent=4)

        _LOGGER.info("Change detection results saved to %s", result_path)
