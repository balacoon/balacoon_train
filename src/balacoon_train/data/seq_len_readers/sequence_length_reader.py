"""
Copyright 2022 Balacoon

Abstract class that defines
sequence length reader
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from balacoon_train.config import ConfigurableConfig


class SequenceLengthReader(ABC):
    @abstractmethod
    def get_sequence_length(self, name: str) -> int:
        """
        Given name of the utterance, returns sequence length for it.
        Is used by `SequenceLengthCollector` to read lengths of utterances
        in dataset to do bucketing.
        """
        pass


@dataclass
class SequenceLengthReaderConfig(ConfigurableConfig):
    cls: str = "???"  # should be a name of the sequence length reader to instantiate
