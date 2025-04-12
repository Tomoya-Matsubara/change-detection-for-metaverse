"""Abstraction layer for object detection models."""

import abc
import pathlib

import numpy as np

from mcd import dtypes


class ObjectDetectorBase(abc.ABC):
    """Object detection model interface."""

    @abc.abstractmethod
    def detect(
        self,
        image: dtypes.ColoredImageType[np.uint8],
        dataset_id: str,
        image_id: str,
        results_path: pathlib.Path,
    ) -> None:
        """Detect objects in an image.

        The results are saved in the specified dataset and image with the following
        structure:

        ```
        results_path/
        └── dataset_id/
            └── image_id/
                ├── labels/
                │   └── image0.txt
                └── image0.jpg
        ```

        where the contents under `image_id` vary depending on the model.

        Parameters
        ----------
        image : dtypes.ColoredImageType[np.uint8]
            Image to detect objects in.
        dataset_id : str
            Identifier for the dataset.
        image_id : str
            Identifier for the image being processed.
        results_path : pathlib.Path
            Path to save the results.
        """

    @abc.abstractmethod
    def detect_all(
        self, dataset_path: pathlib.Path, results_path: pathlib.Path
    ) -> None:
        """Detect objects in all images in a dataset.

        The results are saved in the specified dataset with the following structure:

        ```
        results_path/
        └── dataset_id/
            ├── image0/
            │   ├── labels/
            │   │   └── image0.txt
            │   └── image0.jpg
            ├── image1/
            │   ├── labels/
            │   │   └── image1.txt
            │   └── image1.jpg
            └── ...
        ```

        where the contents under each image directory vary depending on the
        model.

        Parameters
        ----------
        dataset_path : pathlib.Path
            Path to the dataset with the following structure:

            ```
            dataset_id/
                └── images/
                    ├── image0.png
                    ├── image1.png
                    └── ...
            ```

            Note that the `images` directory name is fixed.
        results_path : pathlib.Path
            Path to save the results.
        """
