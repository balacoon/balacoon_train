"""
Copyright 2023 Balacoon

selects previous utterance for the given utterance
"""

import logging
import re
import random
from dataclasses import dataclass
from typing import Dict, List

from balacoon_train.config import Config
from balacoon_train.data.container import Container
from balacoon_train.data.loaders.other_utterance_selector import OtherUtteranceSelector, OtherUtteranceSelectorConfig


class PreviousUtteranceSelector(OtherUtteranceSelector):
    """
    Selects previous utterance for the given utterance in the container.
    Which utterance is a previous one is defined based on the uttid.
    Regex is used to define which part of the utterance can be used
    to define the order.
    """

    def __init__(self, config: Config):
        """
        simple constructor of `OtherUtteranceSelector` as a configurable object.
        """
        config.add_missing(PreviousUtteranceSelectorConfig)
        super().__init__(config)
        self._order_re = re.compile(self._config.order_regex)
        # maps utteranceid to its predecessor
        self._name2previous: Dict[str, str] = dict()

    def find_predecessors(self, ids: List[str]):
        """
        Should be called just once to fill in map of predecessors.
        This method should be called after validation to actually
        populate map of utterance predecessors. Makes two loops over
        ids and tries to find predecessor for each. There are 3 rules:
            * select previous utterance in order
            * if there is no predecessor, select successor
            * if there is no predecessor/successor - use any utterance from same speaker.
              for this one - fall back to parent OtherUtteranceSelector
        """
        
        # set alternatives of parent
        super().set_alternatives(ids)
        
        # to check if such an id exists
        ids_set = set(ids)

        for name in ids:
            # it is checked during validation that order can be extracted
            order_re_match = self._order_re.search(name)
            prefix, idx_str = order_re_match.group(1), order_re_match.group(2)
            idx = int(idx_str)

            # compose list of possible suffixes to check
            suffixes = []
            for sign in [1, -1]:
                for i in range(1, 10):
                    prev_idx = idx - sign * i
                    if prev_idx < 0:
                        break
                    prev_idx_str = str(prev_idx)
                    suffixes.append(prev_idx_str)
                    suffixes.append(prev_idx_str.zfill(len(idx_str)))
            
            # go through suffixes and check if there is any name like that
            for suffix in suffixes:
                prev_name = prefix + suffix
                if prev_name in ids_set:
                    self._name2previous[name] = prev_name
                    break
            # if such a name was not found, alternative will be used

    def load(self, container: Container, validate: bool = False) -> bool:
        """
        Gets previous utterance id for the utterance id in the container
        and puts it back into container as alternative under `name`.
        During validation - mapping names to their predecessors is not ready,
        so putting back itself for further validation

        Parameters
        ----------
        container: Container
            data container, which has utterance id for which
            the previous should be found
        validate: bool
            whether to additionally validate during loading

        Returns
        -------
        flag: bool
            whether loading is successful
        """
        name = container.get_id()
        if validate:
            if self._order_re.search(name) is None:
                logging.warning(
                    f"Can't extract order from [{name}] to define previous utterance"
                )
                return False
            if self._find_category(name) < 0:
                logging.warning(
                    "There is no category for [{}], skip since no alternative".format(
                        name
                    )
                )
                return False
            # put back same name as previous so other dependent data loaders/processors can be validated
            other_name = name
        else:
            if name in self._name2previous:
                other_name = self._name2previous[name]
            else:
                # use random utterance for the given speaker
                cat = self._find_category(name)
                other_name = random.choice(self._category2alternatives[cat])

        container.set_alternative_id(self._name, other_name)
        return True

    def unload(self, container: Container):
        """
        no actual data is loaded in this one, nothing to unload
        """
        pass


@dataclass
class PreviousUtteranceSelectorConfig(OtherUtteranceSelectorConfig):
    """
    configuration of selector of previous utterance.
    defines regexes for part of utterance that corresponds to order
    """

    cls: str = (
        PreviousUtteranceSelector.__module__ + "." + PreviousUtteranceSelector.__name__
    )
    name: str = (
        "previous"  # is put to data container together with previous utterance id
    )
    order_regex: str = "^(.+_)([0-9]+)$"  # regex to apply to utterance id to get order
    extension: str = ""  # dummy entry to comply with parent class
