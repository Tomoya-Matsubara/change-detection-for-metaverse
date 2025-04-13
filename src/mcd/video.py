"""Utilities to visualize change detection results in videos."""

import logging
import pathlib
import typing
import warnings

import cv2
import numpy as np
from rich import progress

from mcd import dtypes, log
from mcd.change_detection import result
from mcd.refine import schema

_LOGGER: typing.Final[logging.Logger] = log.setup_logger(__name__)
"""Logger for the module."""


class Color:
    """Color constants in BGR format."""

    RED = (0, 0, 255)
    """Red color."""

    BLUE = (255, 0, 0)
    """Blue color."""


def _draw_changed_pixel_from_result_3d(
    result_3d: schema.SinglePairResult3d,
    image_before: dtypes.ColoredImageType[np.uint8],
    image_after: dtypes.ColoredImageType[np.uint8],
) -> None:
    """Draw the changed pixels on the before and after images based on the 3D result.

    This function modifies the input images in place.

    Parameters
    ----------
    result_3d : SinglePairResult3d
        Result of the change detection in 3D space.
    image_before : ColoredImageType[np.uint8]
        Image before the change.
    image_after : ColoredImageType[np.uint8]
        Image after the change.
    """
    for removed_label_info in result_3d.removed:
        change_pixel = (removed_label_info.pixel.x, removed_label_info.pixel.y)
        cv2.circle(image_before, change_pixel, 20, Color.BLUE, -1)
    for added_label_info in result_3d.added:
        change_pixel = (added_label_info.pixel.x, added_label_info.pixel.y)
        cv2.circle(image_after, change_pixel, 20, Color.RED, -1)


def _draw_changed_pixel_from_result_2d(
    result: result.SinglePairResult,
    width: int,
    height: int,
    image_before: dtypes.ColoredImageType[np.uint8],
    image_after: dtypes.ColoredImageType[np.uint8],
) -> None:
    """Draw the changed pixels on the before and after images based on the result.

    This function modifies the input images in place.

    Parameters
    ----------
    result : SinglePairResult
        Result of the change detection.
    width : int
        Width of the image.
    height : int
        Height of the image.
    image_before : ColoredImageType[np.uint8]
        Image before the change.
    image_after : ColoredImageType[np.uint8]
        Image after the change.
    """
    for removed_label_info in result.removed:
        change_pixel = (
            int(removed_label_info.bounding_box.x * width),
            int(removed_label_info.bounding_box.y * height),
        )
        cv2.circle(image_before, change_pixel, 20, Color.BLUE, -1)
    for added_label_info in result.added:
        change_pixel = (
            int(added_label_info.bounding_box.x * width),
            int(added_label_info.bounding_box.y * height),
        )
        cv2.circle(image_after, change_pixel, 20, Color.RED, -1)


def _draw_changed_pixel(
    result_info: result.ChangeDetectionResults | schema.ChangeDetection3dResults,
    image_id: dtypes.ImageId,
    image_before: dtypes.ColoredImageType[np.uint8],
    image_after: dtypes.ColoredImageType[np.uint8],
) -> None:
    """Draw the changed pixels on the before and after images based on the result.

    This function modifies the input images in place.

    Parameters
    ----------
    result_info : ChangeDetectionResults | ChangeDetection3dResults
        Change detection result.
    image_id : ImageId
        Image ID.
    image_before : ColoredImageType[np.uint8]
        Image before the change.
    image_after : ColoredImageType[np.uint8]
        Image after the change.
    """
    result_for_image: result.SinglePairResult | schema.SinglePairResult3d | None
    if isinstance(result_info, schema.ChangeDetection3dResults):
        if (result_for_image := result_info.root.get(image_id)) is None:
            return
        _draw_changed_pixel_from_result_3d(result_for_image, image_before, image_after)
    else:
        if (result_for_image := result_info.result.get(image_id)) is None:
            return
        _draw_changed_pixel_from_result_2d(
            result_for_image,
            result_info.image_width,
            result_info.image_height,
            image_before,
            image_after,
        )


def visualize_change_detection_result(
    result_info: result.ChangeDetectionResults | schema.ChangeDetection3dResults,
    before_images_paths: list[pathlib.Path],
    after_images_paths: list[pathlib.Path],
    video_directory: pathlib.Path,
    fps: int = 5,
) -> None:
    """Visualize the change detection result in a video.

    There must be the same number of images before and after the change.

    Parameters
    ----------
    result_info : ChangeDetectionResults
        Change detection result.
    before_images_paths : list[pathlib.Path]
        Paths to the images before the change in chronological order.
    after_images_paths : list[pathlib.Path]
        Paths to the images after the change in chronological order.
    video_directory : pathlib.Path
        Path to the directory where the video will be saved.
    fps : int, default 5
        Frames per second of the video.
    """
    video: cv2.VideoWriter | None = None
    file_name = (
        "change_detection_result.mp4"
        if isinstance(result_info, result.ChangeDetectionResults)
        else "refined_change_detection_result.mp4"
    )
    save_path = video_directory / file_name

    with progress.Progress() as progress_bar:
        task = progress_bar.add_task(
            "[cyan]Creating video", total=len(before_images_paths)
        )

        for i, (before_image_path, after_image_path) in enumerate(
            zip(before_images_paths, after_images_paths, strict=True)
        ):
            progress_bar.update(
                task,
                advance=1,
                description="[green]Creating video[/green] for "
                f"{before_image_path.name} ({i + 1}/{len(before_images_paths)})",
            )

            if (image_before := cv2.imread(before_image_path.as_posix())) is None:
                warnings.warn(
                    f"Could not read image: {before_image_path}", stacklevel=2
                )
                continue

            if (image_after := cv2.imread(after_image_path.as_posix())) is None:
                warnings.warn(f"Could not read image: {after_image_path}", stacklevel=2)
                continue

            _draw_changed_pixel(
                result_info,
                before_image_path.stem,
                image_before=typing.cast(
                    "dtypes.ColoredImageType[np.uint8]", image_before
                ),
                image_after=typing.cast(
                    "dtypes.ColoredImageType[np.uint8]", image_after
                ),
            )

            concatenated_image = cv2.hconcat([image_before, image_after])
            if video is None:
                video = cv2.VideoWriter(
                    save_path.as_posix(),
                    cv2.VideoWriter.fourcc("m", "p", "4", "v"),
                    fps,
                    concatenated_image.shape[:2][::-1],
                )
            video.write(concatenated_image)

    if video is not None:
        video.release()
        _LOGGER.info("Video saved at: %s", save_path)
