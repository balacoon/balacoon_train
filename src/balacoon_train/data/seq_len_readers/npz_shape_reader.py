"""
Copyright 2022 Balacoon

reading sequence length from npz
"""

from dataclasses import dataclass, field
from typing import List, cast

import numpy as np

from balacoon_train.config import Config
from balacoon_train.data.container import Container
from balacoon_train.data.loaders.npz_loader import NpzLoader, NpzLoaderConfig
from balacoon_train.data.seq_len_readers.sequence_length_reader import SequenceLengthReader


class NpzShapeReader(NpzLoader, SequenceLengthReader):
    """
    simple class that reads a sequence length of an utterance
    from a npz file, using utterance name and configuration.
    Largely reuses functionality from `NpzLoader`.
    """

    def __init__(self, config: Config):
        config.add_missing(NpzShapeReaderConfig)
        super().__init__(config)

    def get_sequence_length(self, name: str) -> int:
        """
        Gets a sequence name for the given utterance,
        by loading npz archive (Loader functionality is reused)
        and checking shape across sequence axis

        Parameters
        ----------
        name: str
            utterance id for which to get sequence length

        Returns
        -------
        seq_len: int
            sequence length of the given utterance, which can be used to
            group utterances together.
        """
        container = Container(name)
        self.load(container)
        # this should be run after validation so it is already checked
        # that all npz archives contain the stream
        npz = cast(np.lib.npyio.NpzFile, container[self._config.name])
        seq_len = npz[self._config.stream].shape[self._config.axis]
        self.unload(container)
        return seq_len


@dataclass
class NpzShapeReaderConfig(NpzLoaderConfig):
    """
    Configuration of npz shape reader.
    defines name to use as a key for loaded npz
    """

    cls: str = NpzShapeReader.__module__ + "." + NpzShapeReader.__name__
    locations: List[str] = field(
        default_factory=lambda: []
    )  # directory with "*.npz" files
    name: str = "npz"  # name under which to store a loaded npz in a container
    stream: str = (
        "???"  # which stream in npz archive to use as a source of sequence length
    )
    axis: int = 1  # what is the sequence axis in the array of npz archive
