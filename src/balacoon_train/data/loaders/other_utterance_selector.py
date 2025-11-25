"""
Copyright 2022 Balacoon

selects alternative (contrastive or similar)
utterance for the given utterance
"""

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List

from balacoon_train.config import Config
from balacoon_train.data.container import Container
from balacoon_train.data.loaders.loader import Loader, LoaderConfig


class OtherUtteranceSelector(Loader):
    """
    Select another utterance for the given utterance in container.
    It can be a contrastive pair or pair from the same speaker but
    with different content. It should be done just once, in case multiple
    other data loaders has to process this another utterance.
    This object doesn't actually loads anything but selects a
    counterpart utterance id and adds it into container so other
    data loaders can use it.
    Alternative utterance ids are selected via regexes.
    `OtherUtteranceSelector` has list of regexes, which represents categories.
    When input falls into particular category, any other utterance from
    this category is returned as an alternative.
    """

    def __init__(self, config: Config):
        """
        simple constructor of `OtherUtteranceSelector` as a configurable object.
        """
        config.add_missing(OtherUtteranceSelectorConfig)
        super().__init__(config)
        self._name = self._config.name
        self._categories = [re.compile(x) for x in self._config.categories]
        # maps category to utterances that can be selected as alternative for this category
        self._category2alternatives: Dict[int, List[str]] = dict()
        for i in range(len(self._categories)):
            # init with empty list
            self._category2alternatives[i] = []

    def _find_category(self, name: str) -> int:
        """
        helper function that matches utterance id with
        category regexes and returns index of matching category.
        Returns "-1" if there is no category for the given name
        """
        for i, regex in enumerate(self._categories):
            if regex.match(name):
                return i
        return -1

    def set_alternatives(self, ids: List[str]):
        """
        Should be called just once to fill in map of alternatives.
        Configuration defines how group the utterances together.
        This method should be called after validation to actually
        populate map of utterance alternatives.
        """
        for name in ids:
            cat = self._find_category(name)
            self._category2alternatives[cat].append(name)

    def load(self, container: Container, validate: bool = False) -> bool:
        """
        Finds alternative for utterance id stored in container and puts
        it back to container as alternative under `name`.
        During validation - checks that all utterances actually belong
        to some category, as well as filling alternatives dictionary

        Parameters
        ----------
        container: Container
            data container, which has utterance id for which
            the alternative should be found
        validate: bool
            whether to additionally validate during loading

        Returns
        -------
        flag: bool
            whether loading is successful
        """
        name = container.get_id()
        cat = self._find_category(name)
        if validate:
            if cat < 0:
                logging.warning(
                    "There is no category for [{}], skip since no alternative".format(name)
                )
                return False
            # put back same name as alternative so other dependent data loaders/processors can be validated
            other_name = name
        else:
            # randomly select alternative
            other_name = random.choice(self._category2alternatives[cat])

        container.set_alternative_id(self._name, other_name)
        return True

    def unload(self, container: Container):
        """
        no actual data is loaded in this one, nothing to unload
        """
        pass


@dataclass
class OtherUtteranceSelectorConfig(LoaderConfig):
    """
    configuration of selector of other utterances.
    defines regexes to for categories into which to split the utterances.
    """

    cls: str = OtherUtteranceSelector.__module__ + "." + OtherUtteranceSelector.__name__
    name: str = "alternative"  # is put to data container together with alternative utterance id
    categories: List[str] = field(default_factory=lambda: [])  # regexes for categories
    extension: str = ""  # dummy entry to comply with parent class
