"""
Copyright 2025 Balacoon

combines data streams from npz processors, aligning them
and preparing data for voice conversion training
"""

import logging
import zipfile
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from balacoon_train.config import Config
from balacoon_train.data.container import Container
from balacoon_train.data.processors.processor import Processor, ProcessorConfig
from balacoon_train.data.processors.tensor_mock import TensorMock
from balacoon_train.data.processors.vc_ops import resample_phonemes, resample_pitch
from balacoon_train.data.processors.vc_train_processor import (
    VCTrainProcessor,
    VCTrainProcessorConfig,
    PITCH_ACOUSTIC_RATIO,
    PHONEME_ACOUSTIC_RATIO,
)


class VCPredProcessor(VCTrainProcessor):
    def __init__(self, config: Config):
        config.add_missing(VCPredProcessorConfig)
        super().__init__(config)

    def process(self, container: Container, validate: bool = False) -> bool:
        """
        prepares reference acoustic tokens / phonemes / pitch.
        this is done using parent class `VCTrainProcessor` implementation.
        Then we process pitch and phonemes for target.
        """
        is_valid = super().process(container, validate)
        if not is_valid:
            return False

        if validate:
            # check that target streams are in the container
            if any(
                stream not in container
                for stream in [self._config.tgt_text_name, self._config.tgt_pitch_name]
            ):
                return False

        phonemes = container[self._config.tgt_text_name]  # frames
        pitch = container[self._config.tgt_pitch_name]  # frames
        # compute the target length of acoustic tokens
        expected_len = int(
            min(
                round(phonemes.shape[0] / PHONEME_ACOUSTIC_RATIO),
                round(pitch.shape[0] / PITCH_ACOUSTIC_RATIO),
            )
        )

        if validate:
            # check that the total length of prompt + tokens to be generated is not bigger
            # than max length
            total_len = container[self._config.name].shape[0] + expected_len
            return total_len < self._config.max_seq_len

        # align phonemes and pitch
        # Resampling is necessary because phonemes/pitch and acoustic tokens operate at different frame rates.
        # We need to align them to the same length (acoustic tokens length) for the model.
        phonemes = torch.from_numpy(
            resample_phonemes(phonemes.numpy().astype(np.int32), expected_len)
        )
        pitch = torch.from_numpy(
            resample_pitch(pitch.numpy().astype(np.float32), expected_len)
        ).int()
        pitch = self.shift_pitch(pitch)
        # verify that resampling worked
        assert phonemes.shape[0] == expected_len and pitch.shape[0] == expected_len

        # put back tgt pitch and tgt phonemes to container
        container[self._config.tgt_text_name] = phonemes
        container[self._config.tgt_pitch_name] = pitch
        return True

    def collate(self, batch_elements: list[Container], batch: Container):
        """
        During collation we need to concatenate reference and target phonemes and pitch.
        In that way total prompt_len would be the same for all sequences in the batch,
        the sequences would be padded both on the left and on the right.

        pitch: x x x 1 2 3 | 6 1 x x
        ref_pitch_len[i] = 3 (1, 2, 3 - is a reference pitch values without padding)
        ref_acoustic_tokens.shape[0] = 6 (x x x 1 2 3 - reference and the padding)
        pitch_len[i] = 2 (6, 1 - target pitch values without padding)
        """
        ref_pitch = cast(torch.Tensor, batch[self._config.pitch_name])
        tgt_pitch = cast(torch.Tensor, batch[self._config.tgt_pitch_name])
        pitch = torch.cat([ref_pitch, tgt_pitch], dim=1)

        ref_phonemes = cast(torch.Tensor, batch[self._config.text_name])
        tgt_phonemes = cast(torch.Tensor, batch[self._config.tgt_text_name])
        phonemes = torch.cat([ref_phonemes, tgt_phonemes], dim=1)

        batch[self._config.tgt_pitch_name] = pitch
        batch[self._config.tgt_text_name] = phonemes


@dataclass
class VCPredProcessorConfig(VCTrainProcessorConfig):
    cls: str = VCPredProcessor.__module__ + "." + VCPredProcessor.__name__
    name: str = "ref_acoustic_tokens"  # primary data stream, which we align to
    text_name: str = "ref_phoneme"  # stream of phonemes from phoneme recognizer
    pitch_name: str = "ref_pitch"  # stream of pitch from pitch estimator

    max_seq_len: int = 768
    tgt_text_name: str = "phoneme"
    tgt_pitch_name: str = "pitch"
