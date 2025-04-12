"""Module for reconstructing scenes from images.

Disclaimer
----------
The reconstruction process is **NOT** required for the change detection task.

Purpose
-------
The change detection pipeline requires the data loader to implement each loading method
correctly. However, the implementation of these methods can be tricky due to differences
in data formats and conventions.

For example, camera parameters can be stored in many different ways, such as:

- row-major or column-major,
- inverted or not,
- right-handed or left-handed,
- etc.

The reconstruction process helps verify if the data loader is implemented correctly.

Usage
-----

First, create a data loader that inherits from the `ImageLoaderBase` class:

```python
from mcd.loader import base


class MyDataLoader(base.ImageLoaderBase):
    # Implement the required methods
    ...


loader = MyDataLoader()
```

Then, create a reconstructor:

```python
reconstructor = Reconstructor()
```

Finally, reconstruct the scene:

```python
reconstructor.reconstruct_all(loader, results_directory)
```
"""

import logging
import pathlib
import typing

import numpy as np
from open3d import geometry, io, utility
from rich import progress

from mcd import dtypes, log
from mcd.loader import base

_LOGGER: typing.Final[logging.Logger] = log.setup_logger(__name__)
"""Logger for the module."""


def unproject(
    camera_parameters: dtypes.Camera,
    x: dtypes.NpArray1dType[np.uint32],
    y: dtypes.NpArray1dType[np.uint32],
    depth_map: dtypes.NpArray1dType[np.float32],
    correction_matrix: dtypes.NpArray4x4Type[np.float32] | None = None,
) -> dtypes.NpArrayNx3Type[np.float32]:
    """Unproject the pixel to 3D space.

    Parameters
    ----------
    camera_parameters : Camera
        Camera parameters.
    x : NpArray1dType[np.uint32]
        X coordinate of the pixels.
    y : NpArray1dType[np.uint32]
        Y coordinate of the pixels.
    depth_map : NpArray1dType[np.float32]
        Depth map.
    correction_matrix : NpArray4x4Type[np.float32] | None, default None
        Matrix to correct the position in camera space, by default None.

    Returns
    -------
    NpArrayNx3Type[np.float32]
        Unprojected points in 3D space.
    """
    depth_values = depth_map[y, x]  # (num_samples,)

    # Shape (3, num_samples) = (3, 3) @ (3, num_samples)
    position_in_camera_space = (
        np.linalg.inv(camera_parameters.intrinsic)
        @ np.array([x, y, np.ones_like(x)])
        * depth_values
    )

    if correction_matrix is None:
        correction_matrix = np.eye(4, dtype=np.float32)

    # Shape (4, num_samples) = (4, 4) @ (4, num_samples)
    corrected_position_in_camera_space = correction_matrix @ np.vstack(
        [position_in_camera_space, np.ones_like(x)]
    )
    # Shape (4, num_samples) = (4, 4) @ (4, num_samples)
    position_in_world_space = (
        np.linalg.inv(camera_parameters.view_matrix)
        @ corrected_position_in_camera_space
    )

    return typing.cast(
        "dtypes.NpArrayNx3Type[np.float32]", position_in_world_space[:3].T
    )


class Reconstructor:
    """Class for reconstructing scenes from images."""

    def reconstruct(  # noqa: PLR0913
        self,
        image: dtypes.ColoredImageType[np.uint8],
        depth_map: dtypes.GrayscaleImageType[np.float32],
        camera_parameters: dtypes.Camera,
        confidence_map: dtypes.GrayscaleImageType[np.uint8] | None = None,
        max_confidence: int | None = None,
        max_samples: int | None = None,
        correction_matrix: dtypes.NpArray4x4Type[np.float32] | None = None,
    ) -> tuple[dtypes.NpArrayNx3Type[np.float32], dtypes.NpArrayNx3Type[np.float32]]:
        """Reconstruct the scene from the given image and depth map.

        Parameters
        ----------
        image : ColoredImageType[np.uint8]
            Image.
        depth_map : GrayscaleImageType[np.float32]
            Depth map.
        camera_parameters : Camera
            Camera parameters.
        confidence_map : GrayscaleImageType[np.uint8] | None, default None
            Confidence map of the depth map, by default None.
        max_confidence : int | None, default None
            Maximum confidence value, by default None. If provided, only the points with
            the maximum confidence value are used.
        max_samples : int | None, default None
            Maximum number of samples to use, by default None.
        correction_matrix : NpArray4x4Type[np.float32] | None, default None
            Matrix to correct the position in camera space, by default None.
        """
        y, x = np.where(depth_map > 0)
        if confidence_map is not None and max_confidence is not None:
            y, x = np.where(confidence_map == max_confidence)

        if max_samples is not None and (num_samples := len(x)) > max_samples:
            _random_generator = np.random.default_rng()
            indices = _random_generator.choice(num_samples, max_samples, replace=False)
            x, y = x[indices], y[indices]

        points = unproject(
            camera_parameters,
            x=x.astype(np.uint32),
            y=y.astype(np.uint32),
            depth_map=depth_map,
            correction_matrix=correction_matrix,
        )
        colors = image[y, x] / 255

        return points, colors

    def _create_point_cloud(
        self,
        points: dtypes.NpArrayNx3Type[np.float32],
        colors: dtypes.NpArrayNx3Type[np.float32],
    ) -> geometry.PointCloud:
        """Create a point cloud from the points and colors.

        Parameters
        ----------
        points : NpArrayNx3Type[np.float32]
            Points in 3D space.
        colors : NpArrayNx3Type[np.float32]
            Colors of the points.

        Returns
        -------
        PointCloud
            Point cloud.
        """
        point_cloud = geometry.PointCloud()
        point_cloud.points = utility.Vector3dVector(points)
        point_cloud.colors = utility.Vector3dVector(colors)

        return point_cloud

    def reconstruct_all(
        self,
        loader: base.ImageLoaderBase,
        results_directory: pathlib.Path,
        max_samples_per_frame: int | None = None,
    ) -> None:
        """Reconstruct the scene from all the images in the loader.

        Parameters
        ----------
        loader : ImageLoaderBase
            Data loader for the images.
        results_directory : Path
            Directory to save the results.
        max_samples_per_frame : int | None, default None
            Maximum number of samples to use per frame, by default None.
        """
        points: dtypes.NpArrayNx3Type[np.float32] = np.empty((0, 3), dtype=np.float32)
        colors: dtypes.NpArrayNx3Type[np.float32] = np.empty((0, 3), dtype=np.float32)
        with progress.Progress() as progress_bar:
            task = progress_bar.add_task("[cyan]Reconstructing", total=len(loader))

            for i, (
                image_id,
                image,
                depth_map,
                confidence_map,
                camera_parameters,
            ) in enumerate(loader):
                progress_bar.update(
                    task,
                    advance=1,
                    description=f"[green]Reconstructing {image_id} "
                    f"{i + 1}/{len(loader)}",
                )
                frame_points, frame_colors = self.reconstruct(
                    image=image,
                    depth_map=depth_map,
                    camera_parameters=camera_parameters,
                    confidence_map=confidence_map,
                    max_samples=max_samples_per_frame,
                    correction_matrix=loader.correction_matrix,
                )

                points = np.vstack([points, frame_points])
                colors = np.vstack([colors, frame_colors])

        point_cloud = self._create_point_cloud(points, colors)
        save_path = results_directory / "point_cloud.ply"
        io.write_point_cloud(
            save_path.as_posix(),
            point_cloud,
            write_ascii=False,
            compressed=True,
            print_progress=True,
        )
        _LOGGER.info("Point cloud saved at: %s", save_path)
