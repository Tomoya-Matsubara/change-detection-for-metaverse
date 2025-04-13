"""Utility functions for configuration handling."""

import pathlib
import typing

import pydantic


def _validate_path(path: str | pathlib.Path) -> pathlib.Path:
    """Validate a path.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to validate.

    Returns
    -------
    pathlib.Path
        Validated path.
    """
    if isinstance(path, str):
        return pathlib.Path(path)
    return path


type Path = typing.Annotated[pathlib.Path, pydantic.BeforeValidator(_validate_path)]
"""Path type.

This type is equivalent to `pathlib.Path` but can be instantiated from a string in
Pydantic models. This is useful for loading configurations from YAML files with Hydra.
"""
