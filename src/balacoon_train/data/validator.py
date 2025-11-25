"""
Copyright 2022 Balacoon

class that validates if utterances should be used
"""

import itertools
import logging
import tempfile
import time
from typing import List

from balacoon_train.config import Config
from balacoon_train.data.dataset import Dataset
from balacoon_train.data.parallel import parallel_execution, split_ids_to_subsets


class Validator:
    """
    Class that runs validation of the utterances, i.e.
    checks if they can be used. It splits list of utterances
    into subsets, for each subset creates a `Dataset` object
    and validates it in a separate process using
    loaders and processors from `Dataset`. Typical usage is:

        valid_ids = Validator(config.dataset).get_valid_ids(ids)

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

    def get_valid_ids(self, ids: List[str]) -> List[str]:
        """
        Actually runs validation of passed utterances.
        Spawns as many processes as possible.

        Parameters
        ----------
        ids: List[str]
            list of utterance ids to validate

        Returns
        -------
        valid_ids: List[str]
            list of utterance ids that can be actually used
        """
        with tempfile.NamedTemporaryFile() as serialized_conf:
            validation_start = time.time()
            # save config to be loaded in separate processes
            self._config.save(serialized_conf.name, store_subconfig=True)
            # split the ids into separate subsets
            subsets = split_ids_to_subsets(ids)
            # create helper to be invoked from separate processes
            helper = ValidationHelper(serialized_conf.name)
            results = parallel_execution(helper, subsets)
            # concat valid utterances from different processes
            valid_ids = list(itertools.chain(*results))
            validation_took = time.time() - validation_start
            logging.info(
                "Validation of {} utterances with {} processes, took {}".format(
                    len(ids), len(subsets), validation_took
                )
            )
            logging.info(
                "Validation discarded {} utterances, reducing dataset from {} to {}".format(
                    len(ids) - len(valid_ids), len(ids), len(valid_ids)
                )
            )
            return valid_ids


class ValidationHelper:
    """
    helper class that allows to run a parallel validation in several processes.
    each validation helper call creates separate `Dataset` and runs validation on a subset
    of utterance ids.
    """

    def __init__(self, config_path: str):
        """
        constructor of validation helper

        Parameters
        ----------
        config_path: str
            serialized configuration of `Dataset`,
            which is loaded separately in each process
        """
        self._config_path = config_path

    def __call__(self, ids: List[str]) -> List[str]:
        """
        creates an instance of `Dataset` based on
        serialized configuration and runs validation
        on subset of ids provided.

        Parameters
        ----------
        ids: List[str]
            subset of utterance ids to run validation
            on in current process

        Returns
        -------
        valid_ids: List[str]
            utterance ids that passed validation.
            those are returned via internal queue of multiprocessing:
            https://stackoverflow.com/questions/9908781/sharing-a-result-queue-among-several-processes
        """
        config = Config.load(self._config_path)
        idx2id = {i: name for i, name in enumerate(ids)}
        dummy_idx2len = {i: 1 for i in range(len(ids))}
        dataset = Dataset(config, idx2id, dummy_idx2len)
        valid_ids = []
        logging.info("Validating a subset")
        for name in ids:
            is_valid = dataset.validate_utterance(name)
            if is_valid:
                valid_ids.append(name)
        return valid_ids
