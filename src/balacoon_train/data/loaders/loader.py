"""
Copyright 2022 Balacoon

abstract loader of the data used in dataset
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from balacoon_train.config import Config, ConfigurableConfig
from balacoon_train.data.container import Container


class Loader(ABC):
    """
    Interface that specifies the functions that data loader should implement.
    """

    def __init__(self, config: Config):
        self._config = config
        # simplify access because those fields are requested for each utt in dataset
        self._locations = self._config.locations.to_list()
        # for each location checks if it has subdirectories
        self._locations_have_subdirs = [self._has_subdirs(x) for x in self._locations]
        self._ext = self._config.extension
        self._alternative = self._config.alternative

        # cheks if id mapping is specified for loader
        self._id_mapping: Dict[str, str] = dict()
        if "id_mapping" in config and config.id_mapping:
            # check if id mapping exists
            if not os.path.isfile(config.id_mapping):
                raise RuntimeError(
                    "id mapping [{}] is specified for loader, but can't open".format(
                        config.id_mapping
                    )
                )
            # read id mapping to be used when selecting utterance to load
            with open(config.id_mapping, "r") as fp:
                for line in fp:
                    parts = line.strip().split()
                    assert len(parts) == 2, "Invalid id_mapping line: [{}] in {}".format(
                        line, config.id_mapping
                    )
                    self._id_mapping[parts[0]] = parts[1]

    @staticmethod
    def _has_subdirs(data_dir: str) -> bool:
        """
        For big datasets, number of files is too large to put them into
        single directory. In this case data dir has subdirectories, that
        contain manageable subset of files.
        Convention is that sub dirs are named via intengers, or subdir should be
        part of the id
        """
        return os.path.isdir(os.path.join(data_dir, "1"))

    def _get_path(self, container: Container) -> Optional[str]:
        """
        Helper function that creates path to npz archive,
        given container, gets utterance id from it, iterates over directories specified as possible
        locations in the config and checks if such an utterance id exist there
        """
        assert "locations" in self._config, "Loader doesn't have location in a config"

        # get uttid from container
        if self._alternative:
            uttid = container.get_alternative_id(self._alternative)
        else:
            uttid = container.get_id()

        if self._id_mapping:
            if uttid not in self._id_mapping:
                logging.warning("No id mapping for {}".format(uttid))
                return None
            # map uttid to some other to be loaded
            uttid = self._id_mapping[uttid]
        for possible_dir, has_subdirs in zip(self._locations, self._locations_have_subdirs):
            # if there is no sub-directories, i.e. files are directly in data dir, without being nested,
            # then statement has no effect
            subdirs = [""]
            if has_subdirs:
                subdirs.extend(os.listdir(possible_dir))
            for subdir_name in subdirs:
                path = os.path.join(possible_dir, subdir_name, uttid + "." + self._ext)
                if os.path.isfile(path):
                    return path
        return None

    @abstractmethod
    def load(self, container: Container, validate: bool = False) -> bool:
        """
        gets utterance id from container, loads the data based on utterance name
        and puts it back to container.

        Parameters
        ----------
        container: Container
            data storage to load the data to
        validate: bool
            whether to perform additional validation during loading

        Returns
        -------
        flag: bool
            true if loading is successful.
        """
        pass

    def unload(self, container: Container):
        """
        after extractors are applied, loaded data structures, archives
        are no longer relevant. this method removes from container what was
        added at `load` and closes all resources that were open
        """
        assert (
            self._config.name in container
        ), "Can't remove loaded data from container {}, its not there".format(self._config.name)
        container.pop(self._config.name)


@dataclass
class LoaderConfig(ConfigurableConfig):
    """
    configuration of a loader.
    gives a hint what should be in a typical loader configuration
    """

    cls: str = "???"
    name: str = "???"  # name of the loader, which is used as a key to store data in container
    locations: List[str] = field(default_factory=lambda: [])  # directories with files to load
    extension: str = "???"  # extension of the files to load
    id_mapping: str = ""  # path to id mapping if any (alters id that is returned)
    alternative: str = ""  # alternative utterance id key to get from container
