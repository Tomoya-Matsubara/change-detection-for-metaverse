"""Module for refining the change detection results."""

import collections
import functools
import json
import logging
import math
import pathlib
import typing

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich import progress
from sklearn import cluster

from mcd import dtypes, log, reconstruct
from mcd.change_detection import label
from mcd.change_detection import result as cd_result
from mcd.loader import base as loader_base
from mcd.refine import result, schema

sns.set_theme()

_LOGGER: typing.Final[logging.Logger] = log.setup_logger(__name__)
"""Logger for the module."""


class Refiner:
    """Refiner for the change detection results."""

    def _convert_label_2d_to_3d(
        self,
        label_2d: label.LabelInfo,
        depth_map: dtypes.GrayscaleImageType[np.float32],
        image_size: tuple[int, int],
        camera_parameters: dtypes.Camera,
        correction_matrix: dtypes.NpArray4x4Type[np.float32] | None = None,
    ) -> schema.LabelInfo3d:
        """Convert a 2D label to a 3D label.

        Parameters
        ----------
        label_2d : label.LabelInfo
            2D label to convert.
        depth_map : dtypes.GrayscaleImageType[np.float32]
            Depth map of the image.
        image_size : tuple[int, int]
            Size of the image in the format (width, height).
        camera_parameters : dtypes.Camera
            Camera parameters.
        correction_matrix : dtypes.NpArray4x4Type[np.float32] | None, default None
            Matrix to correct the position in camera space.

        Returns
        -------
        LabelInfo3d
            3D label.
        """
        width, height = image_size
        pixel = dtypes.Pixel(
            x=int(label_2d.bounding_box.x * width),
            y=int(label_2d.bounding_box.y * height),
        )
        return schema.LabelInfo3d(
            label_id=label_2d.label_id,
            label_name=label_2d.label_name,
            pixel=pixel,
            point=self._unproject(
                camera_parameters, pixel, depth_map, correction_matrix
            ),
        )

    def _get_object_label_to_points(
        self, results_3d: schema.ChangeDetection3dResults
    ) -> dict[label.LabelId | label.LabelName, schema.ChangePoints]:
        """Get the object label to points mapping from the 3D results.

        Parameters
        ----------
        results_3d : ChangeDetection3dResults
            3D change detection results.

        Returns
        -------
        dict[label.LabelId | label.LabelName, schema.ChangePoints]
            Mapping from object label to points.
        """
        object_label_to_points: dict[
            label.LabelId | label.LabelName, list[schema.ChangePoint]
        ] = collections.defaultdict(list)

        for image_id, result_3d in results_3d.items():
            for label_info_3d in result_3d.added:
                object_label_to_points[label_info_3d.label].append(
                    schema.ChangePoint(
                        image_id=image_id,
                        change=cd_result.Change.ADDED,
                        point=label_info_3d.point,
                        pixel=label_info_3d.pixel,
                    )
                )

            for label_info_3d in result_3d.removed:
                object_label_to_points[label_info_3d.label].append(
                    schema.ChangePoint(
                        image_id=image_id,
                        change=cd_result.Change.REMOVED,
                        point=label_info_3d.point,
                        pixel=label_info_3d.pixel,
                    )
                )

            for label_info_3d in result_3d.unchanged:
                object_label_to_points[label_info_3d.label].append(
                    schema.ChangePoint(
                        image_id=image_id,
                        change=cd_result.Change.UNCHANGED,
                        point=label_info_3d.point,
                        pixel=label_info_3d.pixel,
                    )
                )

        for object_label, points in object_label_to_points.items():
            unique_changes = {point.change for point in points}
            _LOGGER.debug(
                "[%s] Changes: %s",
                object_label,
                [f"{change}" for change in unique_changes],
            )
            for change in unique_changes:
                _LOGGER.debug(
                    "[%s] Number of points: %d",
                    change,
                    len([point for point in points if point.change == change]),
                )

        return {
            object_label: schema.ChangePoints(root=points)
            for object_label, points in object_label_to_points.items()
        }

    def _plot_change_points(
        self,
        label_to_points: dict[label.LabelId | label.LabelName, schema.ChangePoints],
        result_path: pathlib.Path,
        *,
        refined: bool = False,
    ) -> None:
        """Plot the change points and save it to a file.

        Parameters
        ----------
        label_to_points : dict[label.LabelId | label.LabelName, schema.ChangePoints]
            Mapping from object label to points.
        result_path : pathlib.Path
            Path to save the plot.
        refined : bool, default False
            Indicates whether `label_to_points` is the result of refinement.
        """
        num_object_categories = len(label_to_points)
        num_columns = math.ceil(math.sqrt(num_object_categories))
        num_rows = math.ceil(num_object_categories / num_columns)

        fig = plt.figure(figsize=(num_columns * 5, num_rows * 5))

        with progress.Progress() as progress_bar:
            task = progress_bar.add_task(
                "[cyan]Plotting change points", total=num_object_categories
            )
            for i, (object_label, points) in enumerate(label_to_points.items()):
                progress_bar.update(
                    task,
                    advance=1,
                    description=f"[green]Plotting points for {object_label}[/green]",
                )
                ax = fig.add_subplot(num_rows, num_columns, i + 1, projection="3d")
                ax.set_title(f"{object_label}")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")  # type: ignore[attr-defined]

                inner_task = progress_bar.add_task(
                    "[cyan]Plotting points", total=len(points.root)
                )
                for point in points.root:
                    progress_bar.update(inner_task, advance=1)

                    if (color := point.change.color) == "black":
                        ax.scatter(
                            point.point.x,
                            point.point.y,
                            point.point.z,
                            color=color,
                            alpha=0.2,
                        )
                    else:
                        ax.scatter(
                            point.point.x, point.point.y, point.point.z, color=color
                        )

                progress_bar.remove_task(inner_task)

        file_name = "refined_change_points.svg" if refined else "change_points.svg"
        save_path = result_path / file_name
        fig.savefig(save_path)
        _LOGGER.info("Change points plot saved to %s", save_path)

        plt.close(fig)

    def convert_result_2d_to_3d(
        self,
        datasets_path: pathlib.Path,
        results: cd_result.ChangeDetectionResults,
        loader: loader_base.RefinementDataLoaderBase,
        before_name: str = "before",
    ) -> schema.ChangeDetection3dResults:
        """Convert 2D change detection results to 3D.

        Parameters
        ----------
        datasets_path : pathlib.Path
            Path to the datasets directory containing "before" and "after" directories.
        results : cd_result.ChangeDetectionResults
            2D change detection results.

        Returns
        -------
        schema.ChangeDetection3dResults
            3D change detection results.
        """
        results_3d = schema.ChangeDetection3dResults(root={})

        with progress.Progress() as progress_bar:
            task = progress_bar.add_task(
                "[cyan]Converting 2D results to 3D", total=len(results.result)
            )
            for i, (image_id, image_result) in enumerate(results.result.items()):
                progress_bar.update(
                    task,
                    advance=1,
                    description="[green]Converting 2D results to 3D[/green] for "
                    f"{image_id} ({i + 1}/{len(results.result)})",
                )

                depth_path_before, depth_path_after = loader.get_depth_map_path_pair(
                    datasets_path, image_id, before_name
                )
                depth_map_before = loader.get_depth_map(depth_path_before)
                depth_map_after = loader.get_depth_map(depth_path_after)

                convert_label_2d_to_3d_partial = functools.partial(
                    self._convert_label_2d_to_3d,
                    image_size=(results.image_width, results.image_height),
                    camera_parameters=loader.load_camera_parameters(
                        datasets_path, image_id, before_name
                    ),
                    correction_matrix=loader.correction_matrix,
                )

                results_3d.root[image_id] = schema.SinglePairResult3d(
                    added={
                        convert_label_2d_to_3d_partial(
                            label_2d=label_2d, depth_map=depth_map_after
                        )
                        for label_2d in image_result.added
                    },
                    removed={
                        convert_label_2d_to_3d_partial(
                            label_2d=label_2d, depth_map=depth_map_before
                        )
                        for label_2d in image_result.removed
                    },
                    unchanged={
                        convert_label_2d_to_3d_partial(
                            label_2d=label_2d, depth_map=depth_map_after
                        )
                        for label_2d in image_result.unchanged
                    },
                )

        self._export_result(results_3d, datasets_path)
        object_label_to_points = self._get_object_label_to_points(results_3d)

        _LOGGER.info("Plotting change points")
        self._plot_change_points(object_label_to_points, datasets_path)

        return results_3d

    def _unproject(
        self,
        camera_parameters: dtypes.Camera,
        pixel: dtypes.Pixel,
        depth_map: dtypes.GrayscaleImageType[np.float32],
        correction_matrix: dtypes.NpArray4x4Type[np.float32] | None = None,
    ) -> dtypes.Point:
        """Unproject a pixel to a 3D point.

        Parameters
        ----------
        camera_parameters : dtypes.Camera
            Camera parameters.
        pixel : dtypes.Pixel
            Pixel to unproject.
        depth_map : dtypes.GrayscaleImageType[np.float32]
            Depth map of the image.
        correction_matrix : dtypes.NpArray4x4Type[np.float32] | None, default None
            Matrix to correct the position in camera space.

        Returns
        -------
        Point
            3D point in world space.
        """
        position_in_world_space = reconstruct.unproject(
            camera_parameters=camera_parameters,
            x=np.array([pixel.x], dtype=np.uint32),
            y=np.array([pixel.y], dtype=np.uint32),
            depth_map=depth_map,
            correction_matrix=correction_matrix,
        ).squeeze()
        return dtypes.Point(
            x=position_in_world_space[0],
            y=position_in_world_space[1],
            z=position_in_world_space[2],
        )

    def _export_result(
        self,
        results: schema.ChangeDetection3dResults,
        results_path: pathlib.Path,
        *,
        refined: bool = False,
    ) -> None:
        """Export the 3D change detection result.

        Parameters
        ----------
        results : ChangeDetection3dResults
            Result of the 3D change detection.
        results_path : pathlib.Path
            Path to the directory where the results will be saved.
        refined : bool, default False
            Indicates whether the results are refined.
        """
        file_name = (
            "refined_change_detection_result_3d.json"
            if refined
            else "change_detection_result_3d.json"
        )
        result_path = results_path / file_name
        with result_path.open("w") as f:
            json.dump(results.model_dump(), f, indent=4)

        _LOGGER.info("3D change detection result saved to %s", result_path)

    def _cluster_points(
        self,
        change_points: schema.ChangePoints,
        epsilon: float = 0.5,
        min_samples: int = 10,
    ) -> result.ClusteredPoints:
        """Cluster the change points.

        Parameters
        ----------
        change_points : ChangePoints
            Change points to cluster.
        epsilon : float, default 0.5
            Maximum distance between two samples for one to be considered as in the
            neighborhood of the other.
        min_samples : int, default 10
            Number of samples in a neighborhood for a point to be considered as a core
            point.

        Returns
        -------
        ClusteredPoints
            Clustered points.
        """
        clusters = cluster.DBSCAN(eps=epsilon, min_samples=min_samples).fit_predict(
            change_points.coordinates
        )
        return result.ClusteredPoints(
            root=[
                result.ClusteredPoint(
                    image_id=change_points.root[i].image_id,
                    change=change_points.root[i].change,
                    point=change_points.root[i].point,
                    pixel=change_points.root[i].pixel,
                    cluster_id=cluster_id,
                )
                for i, cluster_id in enumerate(clusters)
            ]
        )

    def _voting(
        self,
        cluster_id: result.ClusterId,
        clustered_points: result.ClusteredPoints,
        object_label: label.LabelId | label.LabelName,
    ) -> schema.ChangeDetection3dResults:
        """Voting for the dominant change in the cluster.

        All points in the cluster are assigned the dominant change.

        Parameters
        ----------
        cluster_id : ClusterId
            Cluster ID to vote for.
        clustered_points : ClusteredPoints
            Clustered points.
        object_label : LabelId | LabelName
            Object label.

        Returns
        -------
        ChangeDetection3dResults
            Results of the 3D change detection after voting.
        """
        points_in_cluster = clustered_points.get_points_in_cluster(cluster_id)
        unique_change_values = points_in_cluster.unique_change_values
        change_counts: list[tuple[cd_result.Change, int]] = []
        _LOGGER.debug("Cluster ID: %s", cluster_id)
        _LOGGER.debug(
            "Unique changes: %s",
            [cd_result.Change(change) for change in unique_change_values],
        )

        for change in sorted(unique_change_values):
            num_points = len(
                [_change for _change in points_in_cluster.changes if _change == change]
            )
            _change = cd_result.Change(change)
            change_counts.append((_change, num_points))
            _LOGGER.debug("[%s] Number of points: %s", _change, f"{num_points:,}")

        dominant_change, _ = max(change_counts, key=lambda x: x[1])
        center = np.mean(points_in_cluster.points, axis=0)
        _LOGGER.debug("Dominant change: %s, Center: %s", dominant_change, center)

        label_id = (
            object_label if isinstance(object_label, label.LabelId.__value__) else None
        )
        label_name = (
            object_label
            if isinstance(object_label, label.LabelName.__value__)
            else None
        )

        results_3d = schema.ChangeDetection3dResults(root={})
        for point, pixel, _, image_id in points_in_cluster.iter():
            label_info_3d = schema.LabelInfo3d(
                label_id=label_id, label_name=label_name, pixel=pixel, point=point
            )
            result_3d = results_3d.root.setdefault(
                image_id,
                schema.SinglePairResult3d(added=set(), removed=set(), unchanged=set()),
            )
            if cluster_id != -1:
                match dominant_change:
                    case cd_result.Change.ADDED:
                        result_3d.added.add(label_info_3d)
                    case cd_result.Change.REMOVED:
                        result_3d.removed.add(label_info_3d)
                    case cd_result.Change.UNCHANGED:
                        result_3d.unchanged.add(label_info_3d)
                    case _:
                        typing.assert_never(_)
            else:
                result_3d.unchanged.add(label_info_3d)

        return results_3d

    def refine(
        self,
        results_3d: schema.ChangeDetection3dResults,
        results_path: pathlib.Path,
        epsilon: float = 0.5,
        min_samples: int = 10,
    ) -> schema.ChangeDetection3dResults:
        """Refine the change detection results.

        Parameters
        ----------
        results_3d : ChangeDetection3dResults
            3D change detection results.
        results_path : pathlib.Path
            Path to save the refined results.
        epsilon : float, default 0.5
            Maximum distance between two samples for one to be considered as in the
            neighborhood of the other.
        min_samples : int, default 10
            Number of samples in a neighborhood for a point to be considered as a core
            point.

        Returns
        -------
        ChangeDetection3dResults
            Refined 3D change detection results.
        """
        object_label_to_points = self._get_object_label_to_points(results_3d)
        refined_results_3d = schema.ChangeDetection3dResults(root={})
        with progress.Progress() as progress_bar:
            total = len(object_label_to_points)
            task = progress_bar.add_task(
                "[cyan]Refining change detection results", total=total
            )

            for i, (object_label, points) in enumerate(object_label_to_points.items()):
                progress_bar.update(
                    task,
                    advance=1,
                    description="[green]Refining change detection results for "
                    f"{object_label}[/green] ({i + 1}/{total})",
                )
                clustered_points = self._cluster_points(points, epsilon, min_samples)
                unique_cluster_ids = clustered_points.unique_cluster_ids

                inner_task = progress_bar.add_task(
                    "[cyan]Voting for the dominant change",
                    total=len(unique_cluster_ids),
                )
                for cluster_id in unique_cluster_ids:
                    progress_bar.update(inner_task, advance=1)
                    _results_3d = self._voting(
                        cluster_id, clustered_points, object_label
                    )
                    for image_id, _result_3d in _results_3d.root.items():
                        if image_id in refined_results_3d.root:
                            refined_results_3d.root[image_id].added.update(
                                _result_3d.added
                            )
                            refined_results_3d.root[image_id].removed.update(
                                _result_3d.removed
                            )
                            refined_results_3d.root[image_id].unchanged.update(
                                _result_3d.unchanged
                            )
                        else:
                            refined_results_3d.root[image_id] = _result_3d
                progress_bar.remove_task(inner_task)

        self._export_result(refined_results_3d, results_path, refined=True)
        object_label_to_points_refined = self._get_object_label_to_points(
            refined_results_3d
        )

        _LOGGER.info("Plotting refined change points")
        self._plot_change_points(
            object_label_to_points_refined, results_path, refined=True
        )
        _LOGGER.info("Refinement complete")

        return refined_results_3d
