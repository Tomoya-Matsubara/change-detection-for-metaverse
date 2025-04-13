"""Dataset configuration."""

import pydantic

from mcd.conf import _utils


class DatasetConfig(pydantic.BaseModel, frozen=True, strict=True):
    """Dataset configuration.

    Attributes
    ----------
    datasets_path : pathlib.Path
        Path to the datasets directory.

        The directory should have exactly two subdirectories, one for the dataset before
        the change and one for the dataset after the change.
    before_name : str
        Name of the directory containing the images before the change. `datasets_path`
        should have a subdirectory with this name.
    results_path : pathlib.Path
        Path to the directory where the results will be saved.
    """

    datasets_path: _utils.Path
    """Path to the datasets directory.

    The directory should have exactly two subdirectories, one for the dataset before the
    change and one for the dataset after the change.
    """

    before_name: str = "before"
    """Name of the directory containing the images before the change.

    `datasets_path` should have a subdirectory with this name.
    """

    results_path: _utils.Path
    """Path to the directory where the results will be saved."""
