"""Abstraction layer for data loaders."""

import abc
import pathlib
import typing

import numpy as np

from mcd import dtypes
from mcd.change_detection import exceptions, label


class _DataLoaderBase:
    """Base class for data loaders."""

    @typing.final
    def get_after_dataset_path(
        self, datasets_path: pathlib.Path, before_name: str
    ) -> pathlib.Path:
        """Get the path to the 'after' dataset.

        Parameters
        ----------
        datasets_path : pathlib.Path
            Path to the directory containing the datasets.
        before_name : str
            Name of the dataset that represents the scene before the change.

        Returns
        -------
        pathlib.Path
            Path to the 'after' dataset.

        Raises
        ------
        TooManyDatasetsError
            If more than two datasets are present in the directory.
        AfterDatasetNotFoundError
            If the 'after' dataset is not found.
        """
        after_name = ""
        for dataset_path in datasets_path.iterdir():
            if not dataset_path.is_dir():
                continue
            if dataset_path.stem == before_name:
                continue
            if len(after_name) > 0:
                raise exceptions.TooManyDatasetsError
            after_name = dataset_path.stem
        if len(after_name) == 0:
            raise exceptions.AfterDatasetNotFoundError
        return datasets_path / after_name


class ImageLoaderBase(abc.ABC, _DataLoaderBase):
    """Data loader interface for images.

    Parameters
    ----------
    dataset_path : pathlib.Path
        Path to the dataset directory containing the images.
    correction_matrix : dtypes.NpArray4x4Type[np.float32] | None, default None
        Matrix to correct the position in camera space.

    Attributes
    ----------
    correction_matrix : dtypes.NpArray4x4Type[np.float32] | None
        Matrix to correct the position in camera space.
    """

    def __init__(
        self,
        dataset_path: pathlib.Path,
        correction_matrix: dtypes.NpArray4x4Type[np.float32] | None = None,
    ) -> None:
        self._dataset_path: typing.Final[pathlib.Path] = dataset_path
        """Path to the dataset directory."""

        self.correction_matrix: typing.Final[
            dtypes.NpArray4x4Type[np.float32] | None
        ] = correction_matrix
        """Matrix to correct the position in camera space."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Get the number of images in the dataset.

        Returns
        -------
        int
            Number of images.
        """

    @abc.abstractmethod
    def __iter__(
        self,
    ) -> typing.Iterator[
        tuple[
            dtypes.ImageId,
            dtypes.ColoredImageType[np.uint8],
            dtypes.GrayscaleImageType[np.float32],
            dtypes.GrayscaleImageType[np.uint8] | None,
            dtypes.Camera,
        ],
    ]:
        """Iterate over the images in the dataset in chronological order.

        Yields
        ------
        tuple[
            dtypes.ImageId,
            dtypes.ColoredImageType[np.uint8],
            dtypes.GrayscaleImageType[np.float32],
            dtypes.GrayscaleImageType[np.uint8] | None,
            dtypes.Camera,
        ]
            Image ID, image, depth map, and confidence map.
        """

    @abc.abstractmethod
    def _load_image(
        self, image_id: dtypes.ImageId
    ) -> dtypes.ColoredImageType[np.uint8]:
        """Load an image.

        Parameters
        ----------
        image_id : dtypes.ImageId
            Identifier of the image.

        Returns
        -------
        dtypes.ColoredImageType[np.uint8]
            Image.
        """

    @abc.abstractmethod
    def _load_depth_map(
        self, image_id: dtypes.ImageId
    ) -> dtypes.GrayscaleImageType[np.float32]:
        """Load a depth map.

        Parameters
        ----------
        image_id : dtypes.ImageId
            Identifier of the image.

        Returns
        -------
        dtypes.GrayscaleImageType[np.float32]
            Loaded depth map.
        """

    def _get_confidence_map(
        self, image_id: dtypes.ImageId
    ) -> dtypes.GrayscaleImageType[np.uint8] | None:
        """Get the confidence map of the image.

        Parameters
        ----------
        image_id : dtypes.ImageId
            Identifier of the image.

        Returns
        -------
        dtypes.GrayscaleImageType[np.uint8] | None
            Confidence map or None if not available.
        """

    @abc.abstractmethod
    def _load_camera_parameters(self, image_id: dtypes.ImageId) -> dtypes.Camera:
        """Load the camera parameters of an image.

        Parameters
        ----------
        image_id : dtypes.ImageId
            Identifier of the image.

        Returns
        -------
        dtypes.Camera
            Camera parameters.
        """

    @typing.final
    def __getitem__(
        self, image_id: dtypes.ImageId
    ) -> tuple[
        dtypes.ColoredImageType[np.uint8],
        dtypes.GrayscaleImageType[np.float32],
        dtypes.GrayscaleImageType[np.uint8] | None,
        dtypes.Camera,
    ]:
        """Get the frame data of the specified image.

        Parameters
        ----------
        image_id : dtypes.ImageId
            Identifier of the image.

        Returns
        -------
        tuple[
            dtypes.ColoredImageType[np.uint8],
            dtypes.GrayscaleImageType[np.float32],
            dtypes.GrayscaleImageType[np.uint8] | None,
            dtypes.Camera,
        ]
            Loaded image, depth map, and confidence map.
        """
        return (
            self._load_image(image_id),
            self._load_depth_map(image_id),
            self._get_confidence_map(image_id),
            self._load_camera_parameters(image_id),
        )


class ObjectDetectionDataLoaderBase(abc.ABC, _DataLoaderBase):
    """Data loader interface for object detection results."""

    @abc.abstractmethod
    def get_images_path(
        self, dataset_path: pathlib.Path
    ) -> typing.Generator[pathlib.Path, None, None]:
        """Get the paths to the images in a dataset.

        Parameters
        ----------
        dataset_path : pathlib.Path
            Path to the dataset directory.

        Yields
        ------
        pathlib.Path
            Path to the image.
        """

    @abc.abstractmethod
    def get_labels_path(
        self, dataset_path: pathlib.Path
    ) -> typing.Generator[pathlib.Path, None, None]:
        """Get the paths to the labels in a dataset.

        Parameters
        ----------
        dataset_path : pathlib.Path
            Path to the dataset directory.

        Yields
        ------
        pathlib.Path
            Path to the label.
        """

    @abc.abstractmethod
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
