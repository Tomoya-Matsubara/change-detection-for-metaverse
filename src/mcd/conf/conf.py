"""The top-level configuration."""

import typing

import omegaconf
import pydantic

from mcd.conf import change_detection, dataset, object_detection, refinement


class Config(pydantic.BaseModel, frozen=True, strict=True):
    """The top-level configuration.

    Attributes
    ----------
    object_detection : ObjectDetectionConfig
        Object detection configuration.
    dataset : DatasetConfig
        Dataset configuration.
    change_detection : ChangeDetectionConfig
        Change detection configuration.
    refinement : RefinementConfig
        Refinement configuration.
    """

    object_detection: object_detection.ObjectDetectionConfig
    """Object detection configuration."""

    dataset: dataset.DatasetConfig
    """Dataset configuration."""

    change_detection: change_detection.ChangeDetectionConfig
    """Change detection configuration."""

    refinement: refinement.RefinementConfig
    """Refinement configuration."""

    @classmethod
    def from_omegaconf(cls, config: omegaconf.DictConfig) -> typing.Self:
        """Create a configuration from an OmegaConf configuration.

        Parameters
        ----------
        config : omegaconf.DictConfig
            OmegaConf configuration.

        Returns
        -------
        Config
            Configuration.
        """
        return cls.model_validate(
            omegaconf.OmegaConf.to_container(config, resolve=True)
        )
