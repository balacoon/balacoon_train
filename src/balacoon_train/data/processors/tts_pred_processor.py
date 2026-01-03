"""
Copyright 2025 Balacoon

combines data streams from npz processors, aligning them
and preparing data for text-to-speech training
"""

import logging
import zipfile
from dataclasses import dataclass
from typing import cast
import os
import json

import numpy as np
import torch

from balacoon_train.config import Config
from balacoon_train.data.container import Container
from balacoon_train.data.processors.processor import Processor, ProcessorConfig
from balacoon_train.data.processors.tts_train_processor import (
    TTSTrainProcessor,
    TTSTrainProcessorConfig,
)
from balacoon_train.data.processors.tensor_mock import TensorMock
from balacoon_train.data.processors.vc_ops import resample_phonemes, resample_pitch


class TTSPredProcessor(TTSTrainProcessor):
    def __init__(self, config: Config):
        super().__init__(config)
        self._config.add_missing(TTSPredProcessorConfig)

    def process(self, container: Container, validate: bool = False) -> bool:
        """
        takes container with all data streams loaded for both prompt and target utterances.
        """
        if validate:
            # check that all input streams are actually in the container
            if any(
                stream not in container
                for stream in [
                    self._config.prompt_prefix + self._config.acoustic_tokens_name,
                    self._config.phonemes_name,
                    self._config.prompt_prefix + self._config.phonemes_name,
                    self._config.prompt_prefix + self._config.phonemes_start_name,
                    self._config.prompt_prefix + self._config.phonemes_end_name,
                ]
            ):
                return False

        phonemes_str = container[self._config.phonemes_name]  # num_phonemes
        prompt_phonemes_str = container[
            self._config.prompt_prefix + self._config.phonemes_name
        ]  # num_phonemes

        # WARNING: checks in this processor relies on actual values,
        # so mocking should be disabled for phoneme npz processor
        if validate:
            for phonemes_lst in [phonemes_str, prompt_phonemes_str]:
                for p in phonemes_lst:
                    if p not in self._phoneme_mapping:
                        logging.warning(f"Phoneme {p} not found in phoneme mapping")
                        return False

        phonemes = self.encode_phonemes(phonemes_str)
        prompt_phonemes = self.encode_phonemes(prompt_phonemes_str)

        # create alignment for the prompt
        prompt_acoustic_tokens = container[
            self._config.prompt_prefix + self._config.acoustic_tokens_name
        ]  # frames x vocabs
        prompt_starts = container[
            self._config.prompt_prefix + self._config.phonemes_start_name
        ]  # num_phonemes
        prompt_ends = container[
            self._config.prompt_prefix + self._config.phonemes_end_name
        ]  # num_phonemes
        prompt_acoustic_tokens, prompt_phonemes, prompt_phoneme_indices, is_valid = (
            self.create_alignment(
                prompt_acoustic_tokens, prompt_phonemes, prompt_starts, prompt_ends
            )
        )
        if validate and not is_valid:
            return False

        # put everything back to container for collation
        # prompt data first
        container[self._config.prompt_prefix + self._config.acoustic_tokens_name] = (
            prompt_acoustic_tokens
        )
        container[self._config.prompt_prefix + self._config.phonemes_name] = prompt_phonemes
        container[self._config.prompt_prefix + self._config.name] = prompt_phoneme_indices
        # data to generate
        container[self._config.phonemes_name] = phonemes
        return True

    def collate(self, batch_elements: list[Container], batch: Container):
        """
        collate prompt and target utterance data.
        concat phonemes of prompt and target utterance so they are processed together.
        """

        # prepare the prompt data
        (
            batched_prompt_acoustic_tokens,
            batched_prompt_phonemes,
            batched_prompt_phoneme_indices,
            _,
            prompt_tokens_len,
            prompt_phonemes_len,
        ) = self.collate_prompt(
            batch_elements,
            self._config.prompt_prefix + self._config.acoustic_tokens_name,
            self._config.prompt_prefix + self._config.phonemes_name,
            self._config.prompt_prefix + self._config.name,
            None,
        )

        # prepare the target text data
        phonemes_lst = [element[self._config.phonemes_name] for element in batch_elements]
        batched_phonemes, phonemes_len = self.pad_and_stack(
            phonemes_lst, axis=0, val=0, on_right=True
        )

        # put everything to the batch container
        batch[self._config.prompt_prefix + self._config.acoustic_tokens_name] = (
            batched_prompt_acoustic_tokens
        )
        batch[self._config.prompt_prefix + self._config.phonemes_name] = batched_prompt_phonemes
        batch[self._config.prompt_prefix + self._config.name] = batched_prompt_phoneme_indices
        batch[self._config.phonemes_name] = batched_phonemes
        batch["prompt_tokens_len"] = prompt_tokens_len
        batch["prompt_phonemes_len"] = prompt_phonemes_len
        batch["phonemes_len"] = phonemes_len


@dataclass
class TTSPredProcessorConfig(TTSTrainProcessorConfig):
    cls: str = TTSPredProcessor.__module__ + "." + TTSPredProcessor.__name__
    prompt_prefix: str = "prompt_"
