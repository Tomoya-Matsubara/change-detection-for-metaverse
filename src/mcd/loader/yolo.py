"""Data loader for object detection results created by YOLO."""

import pathlib
import typing

from mcd.change_detection import label
from mcd.loader import base


class YoloObjectDetectionDataLoader(base.ObjectDetectionDataLoaderBase):
    """Data loader for object detection results created by YOLO."""

    def get_images_path(
        self, dataset_path: pathlib.Path
    ) -> typing.Generator[pathlib.Path, None, None]:
        """Get the paths to the images in a dataset.

        The dataset directory for YOLO is expected to have the following structure:

        ```
        dataset_path/
        ├── image0/
        │   ├── labels/
        │   │   └── image0.txt
        │   └── image0.jpg
        ├── image1/
        │   └── ...
        └── ...
        ```

        Parameters
        ----------
        dataset_path : pathlib.Path
            Path to the dataset directory.

        Yields
        ------
        pathlib.Path
            Path to the image.
        """
        for file_path in dataset_path.iterdir():
            yield from file_path.glob("*.jpg")

    def get_labels_path(
        self, dataset_path: pathlib.Path
    ) -> typing.Generator[pathlib.Path, None, None]:
        """Get the paths to the labels in a dataset.

        The dataset directory for YOLO is expected to have the following structure:

        ```
        dataset_path/
        ├── image0/
        │   ├── labels/
        │   │   └── image0.txt
        │   └── image0.jpg
        ├── image1/
        │   └── ...
        └── ...
        ```

        Parameters
        ----------
        dataset_path : pathlib.Path
            Path to the dataset directory.

        Yields
        ------
        pathlib.Path
            Path to the label.
        """
        for file_path in dataset_path.iterdir():
            yield from (file_path / "labels").glob("*.txt")

    def read_labels(self, path: pathlib.Path) -> list[label.LabelInfo]:
        """Read labels from a file.

        Parameters
        ----------
        path : pathlib.Path
            Path to the label file.

        Returns
        -------
        list[label.LabelInfo]
            Labels read from the file.
        """
        labels: list[label.LabelInfo] = []
        with path.open("r") as file:
            for line in file:
                _label, x, y, width, height = line.split()
                labels.append(
                    label.LabelInfo(
                        label_id=int(_label) if _label.isdigit() else None,
                        label_name=_label if not _label.isdigit() else None,
                        bounding_box=label.BoundingBox(
                            x=float(x),
                            y=float(y),
                            width=float(width),
                            height=float(height),
                        ),
                    )
                )
        return labels
