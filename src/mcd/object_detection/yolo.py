"""Object detection using YOLO."""

import itertools
import logging
import pathlib
import typing

import cv2
import numpy as np
import ultralytics
from rich import progress

from mcd import dtypes, log
from mcd.object_detection import base

_LOGGER: typing.Final[logging.Logger] = log.setup_logger(__name__)
"""Logger for this module."""


class YoloObjectDetector(base.ObjectDetectorBase):
    """YOLO object detection model.

    Parameters
    ----------
    model_path : pathlib.Path
        Path to the YOLO model file.
    """

    def __init__(self, model_path: pathlib.Path) -> None:
        self._model_path: typing.Final[pathlib.Path] = model_path
        """Path to the YOLO model file."""

        self._model: typing.Final[ultralytics.YOLO] = ultralytics.YOLO(model_path)
        """YOLO model."""

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

        Parameters
        ----------
        image : dtypes.ColoredImageType[np.uint8]
            Image to detect objects in.
        dataset_id : str
            Identifier for the dataset.
        image_id : str
            Identifier for the image being processed.
        results_path : pathlib.Path
            Path where the results will be saved.
        """
        result = self._model.predict(
            source=image,
            project=(results_path / dataset_id).as_posix(),
            name=image_id,
            save=True,
            save_txt=True,
            exist_ok=True,
            verbose=False,
        )[0]

        # YOLO saves the results with the name `image0` by default, e.g., `image0.jpg`,
        # `image0.txt`. For consistency, we rename these files to match the image ID.
        result_path = results_path / dataset_id / image_id
        for file_path in itertools.chain(
            result_path.glob("image0.*"), result_path.glob("labels/image0.*")
        ):
            new_file_name = f"{image_id}{file_path.suffix}"
            _LOGGER.debug("Renaming %s to %s.", file_path, new_file_name)
            file_path.rename(file_path.parent / new_file_name)

        if (boxes := result.boxes) is None or len(boxes.cls) == 0:
            _LOGGER.debug("No objects detected in image %s.", image_id)
            return
        _LOGGER.debug("Detected %d objects in image %s.", len(boxes.cls), image_id)

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
        image_paths = list(dataset_path.glob("images/*.png"))

        with progress.Progress() as progress_bar:
            task = progress_bar.add_task(
                "[cyan]Detecting objects", total=len(image_paths)
            )

            for i, image_path in enumerate(image_paths):
                progress_bar.update(
                    task,
                    advance=1,
                    description=f"[green]Detecting objects[/green] in {image_path.name}"
                    f" ({i + 1}/{len(image_paths)})",
                )

                self.detect(
                    image=cv2.imread(image_path.as_posix()).astype(np.uint8),
                    dataset_id=dataset_path.name,
                    image_id=image_path.stem,
                    results_path=results_path,
                )
