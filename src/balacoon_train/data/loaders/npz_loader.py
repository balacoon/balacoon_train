"""
Copyright 2022 Balacoon

npz archive loader
"""

import logging
from dataclasses import dataclass, field
from typing import List, cast

import numpy as np

from balacoon_train.config import Config
from balacoon_train.data.container import Container
from balacoon_train.data.loaders.loader import Loader, LoaderConfig


class NpzLoader(Loader):
    """
    loader that loads npz archives - dictionaries with named numpy arrays.
    this is generic data structure suitable for any acoustic features, embeddings, etc.
    """

    def __init__(self, config: Config):
        """
        simple constructor of `NpzLoader` as a configurable object.
        """
        config.add_missing(NpzLoaderConfig)
        super().__init__(config)

    def load(self, container: Container, validate: bool = False) -> bool:
        """
        Loads npz archive into container, so it can be accessed
        by various extractors. gets utterance id from container,
        loads npz archive and puts it to container under
        `config.name`. User has to specify name in configuration,
        since there could be multiple npz loaders.

        Parameters
        ----------
        container: Container
            data container, to get utterance id from and to put loaded
            npz archive to
        validate: bool
            whether to additionally validate during loading

        Returns
        -------
        flag: bool
            whether loading is successful
        """
        path = self._get_path(container)
        if validate and path is None:
            logging.warning(f"{container.get_id()} is missing for {self._config.name}, skip")
            return False
        container[self._config.name] = np.load(cast(str, path))
        return True

    def unload(self, container: Container):
        """
        additionally should close npz file that was open
        """
        cast(np.lib.npyio.NpzFile, container[self._config.name]).close()
        super().unload(container)


@dataclass
class NpzLoaderConfig(LoaderConfig):
    """
    configuration of npz loader.
    defines location of npz, and how to store loaded npz archives in data container.
    """

    cls: str = NpzLoader.__module__ + "." + NpzLoader.__name__
    name: str = "???"  # name under which npz archive is stored in the data container
    locations: List[str] = field(default_factory=lambda: [])  # directory with "*.npz" files
    extension: str = "npz"  # extension of npz files
