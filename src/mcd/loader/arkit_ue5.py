"""Change detection loader for ARKit and Unreal Engine 5."""

import json
import pathlib

import cv2
import numpy as np
import polars as pl

from mcd import dtypes
from mcd.loader import arkit as arkit_loader
from mcd.loader import base
from mcd.utils import arkit


class ArkitUe5RefinementDataLoader(base.RefinementDataLoaderBase):
    """Data loader for ARKit and Unreal Engine 5 datasets to refine change detection.

    The loader expects the following datasets configuration:

    - Before: Scenes captured in Unreal Engine 5.
    - After: Scenes captured in ARKit.

    ```
    datasets/
    ├── after/
    │   ├── images/
    │   └── depth/
    │       ├── image_0.json
    │       └── ...
    └── before/
        ├── images/
        └── depth/
            ├── image_0.csv
            └── ...
    ```
    """

    def __init__(self) -> None:
        super().__init__(correction_matrix=arkit.ARKIT_MATRIX)

    def get_depth_map_path_pair(
        self, datasets_path: pathlib.Path, image_id: str, before_name: str = "before"
    ) -> tuple[pathlib.Path, pathlib.Path]:
        """Get the pair of paths to the depth maps of an image.

        Parameters
        ----------
        datasets_path : pathlib.Path
            Path to the datasets directory containing "before" and "after" directories.

            ```
            datasets/
            ├── after/
            │   ├── images/
            │   └── depth/
            │       ├── image_0.json
            │       └── ...
            └── before/
                ├── images/
                └── depth/
                    ├── image_0.csv
                    └── ...
            ```

            Note that the depth map of the "before" dataset is in CSV format created by
            Unreal Engine 5, and the depth map of the "after" dataset is in JSON format
            captured by ARKit.
        image_id : str
            Identifier of the image.

        Returns
        -------
        tuple[pathlib.Path, pathlib.Path]
            Before and after depth map paths.
        """
        after_dataset_path = self.get_after_dataset_path(datasets_path, before_name)
        return (
            datasets_path / before_name / "depth" / f"{image_id}.csv",
            after_dataset_path / "depth" / f"{image_id}.json",
        )

    def _get_depth_map_from_csv(
        self, depth_path: pathlib.Path
    ) -> dtypes.GrayscaleImageType[np.float32]:
        """Get the depth map from a CSV file created by Unreal Engine 5.

        Parameters
        ----------
        depth_path : pathlib.Path
            Path to the depth map file.

        Returns
        -------
        dtypes.GrayscaleImageType[np.float32]
            Depth map.
        """
        dataframe = pl.read_csv(depth_path, has_header=False)
        return dataframe.to_numpy().squeeze()

    def _get_depth_map_from_json(
        self, depth_path: pathlib.Path
    ) -> dtypes.GrayscaleImageType[np.float32]:
        """Get the depth map from a JSON file captured by ARKit.

        Parameters
        ----------
        depth_path : pathlib.Path
            Path to the depth map file.

        Returns
        -------
        dtypes.GrayscaleImageType[np.float32]
            Depth map.
        """
        with depth_path.open("r") as file:
            frame = arkit_loader.ArkitFrame.model_validate(json.load(file))
        depth_map = frame.depth_map.to_numpy()

        # ARKit depth maps are smaller than the image resolution
        return cv2.resize(
            depth_map, (int(frame.resolution[1]), int(frame.resolution[0]))
        ).astype(np.float32)

    def get_depth_map(
        self, depth_path: pathlib.Path
    ) -> dtypes.GrayscaleImageType[np.float32]:
        """Get the depth map from a file.

        Parameters
        ----------
        depth_path : pathlib.Path
            Path to the depth map file. The accepted formats are CSV (from Unreal Engine
            5) and JSON (from ARKit).

        Returns
        -------
        dtypes.GrayscaleImageType[np.float32]
            Depth map.
        """
        match depth_path.suffix:
            case ".csv":
                return self._get_depth_map_from_csv(depth_path)
            case ".json":
                return self._get_depth_map_from_json(depth_path)
            case _:
                message = f"Unsupported depth map format: {depth_path.suffix}"
                raise ValueError(message)

    def load_camera_parameters(
        self, datasets_path: pathlib.Path, image_id: str, before_name: str = "before"
    ) -> dtypes.Camera:
        """Load the camera parameters of an image.

        Parameters
        ----------
        datasets_path : pathlib.Path
            Path to the datasets directory containing "before" and "after" directories.

            ```
            datasets/
            ├── after/
            │   ├── images/
            │   └── depth/
            │       ├── image_0.json
            │       └── ...
            └── before/
                ├── images/
                └── depth/
                    ├── image_0.csv
                    └── ...
            ```

            Although from which dataset the camera parameters are loaded is not
            important, this method loads the camera parameters from the "after" (ARKit)
            dataset.
        image_id : str
            Identifier of the image.
        before_name : str, default "before"
            Name of the dataset that represents the scene before the change.

        Returns
        -------
        dtypes.Camera
            Camera parameters.
        """
        _, depth_path_after = self.get_depth_map_path_pair(
            datasets_path, image_id, before_name
        )
        with depth_path_after.open("r") as file:
            frame = arkit_loader.ArkitFrame.model_validate(json.load(file))
        return dtypes.Camera(intrinsic=frame.intrinsic, view_matrix=frame.view_matrix)
