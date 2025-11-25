"""
Copyright 2022 Balacoon

implementation for torch.utils.data.Dataset
"""

from dataclasses import dataclass, field
from typing import Dict, List, cast

import torch

from balacoon_train.config import Config, create_configurable
from balacoon_train.data.container import Container
from balacoon_train.data.loaders.loader import Loader, LoaderConfig
from balacoon_train.data.loaders.other_utterance_selector import OtherUtteranceSelector
from balacoon_train.data.loaders.previous_utterance_selector import PreviousUtteranceSelector
from balacoon_train.data.processors.processor import Processor, ProcessorConfig


@dataclass
class DatasetConfig:
    loaders: List[LoaderConfig] = field(
        default_factory=lambda: []
    )  # list of data loaders executed in dataset
    processors: List[ProcessorConfig] = field(
        default_factory=lambda: []
    )  # list of data processors executed in dataset


class Dataset(torch.utils.data.Dataset):
    """
    borysthenes dataset is a configurable data loader,
    which combines multiple sources of data specified in a config file.
    `Dataset` is built on top of 2 abstractions that are provided
    as ordered lists in a config:

        - data loader - simply loads the data from files based on utterance id.
          For example, loads linguistic utterance or npz archive from files.
        - data processor - gets actual training data from outputs of data loaders or
          other data processors. order in which data processors are specified
          matters, because one processor may depend on the other. Another possible
          functionality of data processors - is to adjust existing data, for example
          adjust durations and melspectrogram to have same number of frames.

    Dataset implements two main methods:

        - `__getitem__` which is executed on a randomly sampled utterance and returns
          a data container with all the data in it according to config
        - collate - method that combines a list of data containers (one per utterance)
          into a single container with a batch of data.

    Additionally, `Dataset` provides a helper function `validate_utterance`, which is
    used inside of Validator.
    """

    def __init__(self, config: Config, idx2id: Dict[int, str], idx2len: Dict[int, int]):
        """
        constructor of dataset

        Parameters
        ----------
        config: Config
            configuration of dataset, containing info on
            data loaders and processors in use
        idx2id: Dict[int, str]
            mapping between utterance index and id (name).
            Samplers operate indices, so dataset has to convert
            an index to a name of the utterance to load
        """
        config.add_missing(DatasetConfig)
        self._config = config
        self._idx2id = idx2id
        self._idx2len = idx2len
        self._loaders: List[Loader] = [
            cast(Loader, create_configurable(c)) for c in self._config.loaders if c
        ]
        assert len(self._loaders) > 0, "No loaders where created in dataset, check config"
        self._processors: List[Processor] = [
            cast(Processor, create_configurable(c)) for c in self._config.processors if c
        ]

        # propagate utterance ids to loaders/processors that need them
        ids = list(self._idx2id.values())
        for loader in self._loaders:
            if isinstance(loader, OtherUtteranceSelector):
                cast(OtherUtteranceSelector, loader).set_alternatives(ids)
            elif isinstance(loader, PreviousUtteranceSelector):
                cast(PreviousUtteranceSelector, loader).find_predecessors(ids)

    def get_idx2id(self) -> Dict[int, str]:
        """
        access method, getter that returns mapping
        between utterance index and id
        """
        return self._idx2id

    def get_idx2len(self) -> Dict[int, int]:
        """
        access method, getter that returns
        mapping between utterance index and utterance length
        """
        return self._idx2len

    def validate_utterance(self, name: str) -> bool:
        """
        validate a single utterance. This is executed
        by sub-Dataset created in `ValidationHelper`

        Parameters
        ----------
        name: str
            name of the utterance to validate

        Returns
        -------
        is_valid: bool
            True if the utterance is valid and can be used
        """
        container = Container(name)
        for loader in self._loaders:
            if not loader.load(container, validate=True):
                return False
        for processor in self._processors:
            if not processor.process(container, validate=True):
                return False
        for loader in self._loaders:
            loader.unload(container)
        return True

    def __getitem__(self, idx: int) -> Container:
        """
        Primary method called by `torch.utils.data.DataLoader`.
        It gets the name of the utterance to load from the
        sampler and loads all the required information into a
        data container. Later on, those containers are grouped into
        batches.

        Parameters
        ----------
        idx: int
            utterance index, that is further converted to an
            utterance name with `self._idx2id`

        Returns
        -------
        container: Container
            data container with all the data loaded/extracted
            for the requested utterance
        """
        name = self._idx2id[idx]
        container = Container(name)
        # load the staff into container
        for loader in self._loaders:
            loader.load(container)
        # extract useful information from loaded data
        for processor in self._processors:
            processor.process(container)
        # drop loaded things (npz, linguistic utterance),
        # keeping only extracted data
        for loader in self._loaders:
            loader.unload(container)
        return container

    def __len__(self) -> int:
        """
        returns the total size of the dataset
        """
        return len(self._idx2id)

    def get_config(self) -> Config:
        """
        simple getter for dataset configuration,
        in case there is dataset wrapper
        """
        return self._config

    def collate(self, batch_elements: List[Container]) -> Container:
        """
        Combines goes around data processors combines data extracted
        with those processors for multiple utterances into batches.
        That requires processors to know sequence axis of the data,
        because sequences should be padded to maximum length in the batch.
        Resulting batch is put to the `batch` container.

        Parameters
        ----------
        batch_elements: List[Container]
            list of data containers, each of which is obtained with `__getitem__` of `Dataset`.
            i.e. each batch element contains data for a single utterance.

        Returns
        -------
        batch: Container
            a data container where batched data is put to. data is padded and stacked into batches
            by data processors.
        """
        ids = [x.get_id() for x in batch_elements]
        batch = Container(ids)
        for processor in self._processors:
            processor.collate(batch_elements, batch)
        return batch
