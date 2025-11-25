"""
Copyright 2022 Balacoon

class that collects lengths of sequences for bucketing batch sampler
"""

import logging
import tempfile
import time
from typing import Dict, List, cast

from balacoon_train.config import Config, create_configurable
from balacoon_train.data.parallel import parallel_execution, split_ids_to_subsets
from balacoon_train.data.seq_len_readers.sequence_length_reader import SequenceLengthReader


class SequenceLengthCollector:
    """
    Class that collects lengths of sequences in the utterances.
    Those lengths are used by `BucketingBatchSampler` to group
    utterances into buckets, before creating batches.
    This class takes dedicated configuration, which is used to create
    a sequence length reader. Multiple sequence length readers
    are created and sequence lengths are collected in parallel.
    Typical usage would be:

        if seq_len in config:
            id2len = SequenceLengthCollector(config.seq_len).get_seq_len(ids)

    """

    def __init__(self, config: Config):
        """
        constructor of validator

        Parameters
        ----------
        config: Config
            configuration for `Dataset`, because its used to create
            separate instances of `Dataset` for each parallel process
        """
        self._config = config

    def get_seq_len(self, ids: List[str]) -> Dict[str, int]:
        """
        Collects sequence lengths in parallel. Serializes
        sequence length reader config, then spans multiple processes,
        where a dedicated sequence length reader is instantiated and applied
        to a subset of utterances. Then results of multiple processes are merged
        and a mapping between utterance id and correspondent sequence length is
        returned.

        Parameters
        ----------
        ids: List[str]
            list of utterance ids to get sequence lengths for

        Returns
        -------
        id2len: Dict[str, int]
            mapping between utterance id and correspondent sequence length
        """
        with tempfile.NamedTemporaryFile() as serialized_conf:
            seq_len_start = time.time()
            # save config to be loaded in separate processes
            self._config.save(serialized_conf.name, store_subconfig=True)
            # split the ids into separate subsets
            subsets = split_ids_to_subsets(ids)
            # create helper to be invoked from separate processes
            helper = SequenceLengthHelper(serialized_conf.name)
            results = parallel_execution(helper, subsets)
            # merge id2len mapping from difference processes
            id2len: Dict[str, int] = dict()
            for r in results:
                id2len.update(r)
            seq_len_took = time.time() - seq_len_start
            logging.info(
                "Extraction of sequence lengths for {} utterances with {} processes, took {}".format(
                    len(ids), len(subsets), seq_len_took
                )
            )
            return id2len


class SequenceLengthHelper:
    """
    helper class that allows to run a parallel sequence length reading.
    each helper call creates separate `Reader` and runs sequence length extraction
    on a subset of utterance ids.
    """

    def __init__(self, config_path: str):
        """
        constructor of sequence length helper

        Parameters
        ----------
        config_path: str
            serialized configuration of sequence length reader,
            for ex. `NpzSequenceLengthReader`. It is created separately in each process
        """
        self._config_path = config_path

    def __call__(self, ids: List[str]) -> Dict[str, int]:
        """
        creates an instance of sequence length reader based on
        serialized configuration and reads sequence lengths
        for subset of ids provided.

        Parameters
        ----------
        ids: List[str]
            subset of utterance ids to get sequence lengths for
            in current process

        Returns
        -------
        id2len: Dict[str, int]
            mapping between utterance id and its sequence length
        """
        config = Config.load(self._config_path)
        # TODO: once there are multiple possible sequence length readers,
        # create abstraction to inherit and use it here
        reader = cast(SequenceLengthReader, create_configurable(config))
        id2len: Dict[str, int] = dict()
        logging.info("Collecting sequence lengths for a subset")
        for name in ids:
            seq_len = reader.get_sequence_length(name)
            id2len[name] = seq_len
        return id2len
