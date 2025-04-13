"""Runner of the whole pipeline of change detection."""

import json
import pathlib
import typing

import hydra
import omegaconf

from mcd import conf, video
from mcd.change_detection import detector, result
from mcd.conf import change_detection, dataset, object_detection, refinement
from mcd.loader import arkit_ue5
from mcd.loader import base as loader_base
from mcd.loader import yolo as yolo_loader
from mcd.object_detection import base as object_detector_base
from mcd.object_detection import yolo
from mcd.refine import refiner, schema

_PROJECT_ROOT_DIRECTORY: typing.Final[pathlib.Path] = pathlib.Path(__file__).parents[2]
"""Path to the root directory of the project."""


def _run_object_detection(
    dataset_path: pathlib.Path,
    results_path: pathlib.Path,
    object_detector: object_detector_base.ObjectDetectorBase,
) -> None:
    """Run object detection on a dataset.

    Parameters
    ----------
    dataset_path : pathlib.Path
        Path to the dataset directory.
    results_path : pathlib.Path
        Path where the results will be saved.
    object_detector : ObjectDetectorBase
        Object detector to use.
    """
    object_detector.detect_all(dataset_path, results_path)


def _run_object_detection_all(
    datasets_path: pathlib.Path,
    results_path: pathlib.Path,
    object_detector: object_detector_base.ObjectDetectorBase,
) -> None:
    """Run object detection on all datasets in a directory.

    Parameters
    ----------
    datasets_path : pathlib.Path
        Path to the directory containing the datasets.
    results_path : pathlib.Path
        Path where the results will be saved.
    object_detector : ObjectDetectorBase
        Object detector to use.
    """
    for dataset_path in datasets_path.iterdir():
        if not dataset_path.is_dir():
            continue
        _run_object_detection(dataset_path, results_path, object_detector)


def _get_image_path_list(
    datasets_path: pathlib.Path, before_name: str, *, after: bool = False
) -> list[pathlib.Path]:
    """Get the list of image paths in the datasets.

    Parameters
    ----------
    datasets_path : pathlib.Path
        Path to the datasets directory.
    before_name : str
        Name of the directory containing the images before the change.
    after : bool, default False
        Whether to get the image paths after the change.

    Returns
    -------
    list[pathlib.Path]
        List of image paths sorted by name. If `after` is True, the paths are for the
        images after the change. Otherwise, the paths are for the images before the
        change.
    """
    if after:
        dataset_path = arkit_ue5.ArkitUe5RefinementDataLoader().get_after_dataset_path(
            datasets_path, before_name
        )
    else:
        dataset_path = datasets_path / before_name
    return sorted(
        dataset_path.glob("**/*.jpg"), key=lambda path: int(path.stem.split("_")[-1])
    )


def _run_change_detection(
    datasets_path: pathlib.Path,
    before_name: str,
    loader: loader_base.ObjectDetectionDataLoaderBase,
    results_path: pathlib.Path,
) -> None:
    """Run change detection on datasets.

    Parameters
    ----------
    datasets_path : pathlib.Path
        Path to the datasets directory.
    before_name : str
        Name of the directory containing the images before the change.
    loader : ObjectDetectionDataLoaderBase
        Loader for the object detection results.
    results_path : pathlib.Path
        Path to the directory where the results will be saved.
    """
    change_detector = detector.ChangeDetector()
    change_detector.run_all(datasets_path, loader=loader, before_name=before_name)

    json_path = datasets_path / "change_detection_result.json"
    with json_path.open() as file:
        results = result.ChangeDetectionResults.model_validate(json.load(file))
    video.visualize_change_detection_result(
        results,
        _get_image_path_list(results_path, before_name),
        _get_image_path_list(results_path, before_name, after=True),
        datasets_path,
    )


def _run_refinement(
    datasets_path: pathlib.Path,
    results_path: pathlib.Path,
    before_name: str,
    loader: loader_base.RefinementDataLoaderBase,
) -> None:
    """Run refinement on the results of change detection.

    Parameters
    ----------
    datasets_path : pathlib.Path
        Path to the directory containing the datasets.
    results_path : pathlib.Path
        Path to the directory containing the change detection results.
    before_name : str
        Name of the directory containing the images before the change.
    loader : RefinementDataLoaderBase
        Loader for the refinement data.
    """
    json_path = results_path / "change_detection_result.json"
    with json_path.open() as file:
        results = result.ChangeDetectionResults.model_validate(json.load(file))
    refine = refiner.Refiner()
    results_3d = refine.convert_result_2d_to_3d(
        datasets_path, results, loader=loader, before_name=before_name
    )
    refine.refine(results_3d, datasets_path)

    json_path = datasets_path / "refined_change_detection_result_3d.json"
    with json_path.open() as file:
        refined_results_3d = schema.ChangeDetection3dResults.model_validate(
            json.load(file)
        )
    video.visualize_change_detection_result(
        refined_results_3d,
        _get_image_path_list(results_path, before_name),
        _get_image_path_list(results_path, before_name, after=True),
        results_path,
    )


def _execute_change_detection_pipeline(
    dataset_config: dataset.DatasetConfig,
    object_detection_config: object_detection.ObjectDetectionConfig,
    change_detection_config: change_detection.ChangeDetectionConfig,
    refinement_config: refinement.RefinementConfig,
) -> None:
    """Run the whole pipeline of change detection.

    Parameters
    ----------
    dataset_config : DatasetConfig
        Configuration for the datasets.
    object_detection_config : ObjectDetectionConfig
        Configuration for the object detection process.
    change_detection_config : ChangeDetectionConfig
        Configuration for the change detection process.
    refinement_config : RefinementConfig
        Configuration for the refinement process.
    """
    match object_detection_config.object_detector_id:
        case "yolo":
            object_detector = yolo.YoloObjectDetector(
                object_detection_config.model_path
            )
        case _ as unreachable:
            typing.assert_never(unreachable)

    match change_detection_config.loader_id:
        case "yolo":
            object_detection_loader = yolo_loader.YoloObjectDetectionDataLoader()
        case _ as unreachable:
            typing.assert_never(unreachable)

    match refinement_config.loader_id:
        case "arkit_ue5":
            refine_loader = arkit_ue5.ArkitUe5RefinementDataLoader()
        case _ as unreachable_:
            typing.assert_never(unreachable_)

    _run_object_detection_all(
        dataset_config.datasets_path, dataset_config.results_path, object_detector
    )
    _run_change_detection(
        dataset_config.results_path,
        dataset_config.before_name,
        object_detection_loader,
        dataset_config.results_path,
    )
    _run_refinement(
        dataset_config.datasets_path,
        dataset_config.results_path,
        dataset_config.before_name,
        refine_loader,
    )


@hydra.main(
    version_base=None,
    config_path=(_PROJECT_ROOT_DIRECTORY / "config").as_posix(),
    config_name="example",
)
def _main(_config: omegaconf.DictConfig) -> None:
    """Parse the configuration and run the change detection pipeline.

    Parameters
    ----------
    _config : DictConfig
        Configuration for the change detection pipeline.
    """
    config = conf.Config.from_omegaconf(_config)
    _execute_change_detection_pipeline(
        config.dataset,
        config.object_detection,
        config.change_detection,
        config.refinement,
    )


if __name__ == "__main__":
    _main()
