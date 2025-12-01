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


PITCH_ACOUSTIC_RATIO = 100 / 21.5  # ratio between pitch frame rate and acoustic tokens frame rate
PHONEME_ACOUSTIC_RATIO = (
    50 / 21.5
)  # ratio between phoneme frame rate and acoustic tokens frame rate


class VCTrainProcessor(Processor):
    def __init__(self, config: Config):
        super().__init__(config)
        self._config.add_missing(VCTrainProcessorConfig)

    def process(self, container: Container, validate: bool = False) -> bool:
        """
        takes container with all data streams (acoustic tokens, phonemes, pitch) loaded,
        aligns them to the same length.
        """
        if validate:
            # check that all input streams are actually in the container
            if any(
                stream not in container
                for stream in [
                    self._config.acoustic_tokens_name,
                    self._config.phoneme_probs_name,
                    self._config.phoneme_indices_name,
                    self._config.pitch_name,
                ]
            ):
                return False

        pitch = container[self._config.pitch_name]  # frames
        phoneme_probs = container[self._config.phoneme_probs_name]  # frames x vocabs
        phoneme_indices = container[self._config.phoneme_indices_name]  # frames x vocabs
        assert phoneme_probs.shape[0] == phoneme_indices.shape[0]
        acoustic_tokens = container[self._config.acoustic_tokens_name]  # frames x vocabs

        # check that the ratio between rates is as expected
        expected_len = acoustic_tokens.shape[0]
        if validate:
            if abs(pitch.shape[0] / float(expected_len) - PITCH_ACOUSTIC_RATIO) > 0.2:
                # log the mismatch between shapes of pitch and acoustic tokens
                logging.warning(
                    "Mismatch between shapes of pitch and acoustic tokens: {} vs {}".format(
                        pitch.shape[0], expected_len
                    )
                )
                return False
            if abs(phoneme_probs.shape[0] / float(expected_len) - PHONEME_ACOUSTIC_RATIO) > 0.2:
                logging.warning(
                    "Mismatch between shapes of phonemes and acoustic tokens: {} vs {}".format(
                        phoneme_probs.shape[0], expected_len
                    )
                )
                return False
            # nothing else to do during validation
            # TODO: maybe make sure pitch and phonemes are same size,
            # if there are subsequent processors
            return True

        # align phonemes and pitch to the same length as acoustic tokens
        # This ensures that for every acoustic token we have a corresponding phoneme and pitch value,
        # which are used as conditions in the model.
        pitch = torch.from_numpy(
            resample_pitch(pitch.numpy().astype(np.float32), expected_len)
        ).int()
        phonemes = torch.from_numpy(
            resample_phonemes(
                phoneme_probs.numpy().astype(np.float32),
                phoneme_indices.numpy().astype(np.int32),
                expected_len,
                self._config.phoneme_vocab_size,
            )
        )

        # verify that resampling worked
        assert phonemes.shape[0] == expected_len and pitch.shape[0] == expected_len
        # put tensors back to the container so they can be collated by respected npz processors
        pitch = self.shift_pitch(pitch)
        container[self._config.name] = phonemes  # T x vocab_size
        container[self._config.pitch_name] = pitch
        return True

    def shift_pitch(self, pitch: torch.Tensor) -> torch.Tensor:
        return torch.clip(
            pitch + self._config.pitch_val_shift, min=0, max=self._config.pitch_val_shift * 2 - 1
        )

    def collate(self, batch_elements: list[Container], batch: Container):
        """
        Nothing to do here, separate npz processors take care of collating data.
        Here we create a tensor which specifies prompt_len - number of acoustic tokens
        that are masked from loss computation. Those tokens are used to pick the speaker identity.
        """
        seq_len = batch["acoustic_tokens_len"]
        prompt_len = []
        for l in seq_len.tolist():
            # prompt length is a random value in the range [int(3 * 21.5), l // 2].
            # if l//2 < int(3 * 21.5) then use int(3 * 21.5), i.e. prompt should be at least
            # 3 seconds long
            min_prompt_len = int(3 * 21.5)
            max_prompt_len = max(min_prompt_len, l // 2)
            # pick random prompt length in [min_prompt_len, max_prompt_len]
            pl = np.random.randint(min_prompt_len, max_prompt_len + 1)
            prompt_len.append(pl)
        batch["prompt_len"] = torch.tensor(prompt_len, dtype=torch.int)
        super().collate(batch_elements, batch)  # collates phonemes


@dataclass
class VCTrainProcessorConfig(ProcessorConfig):
    cls: str = VCTrainProcessor.__module__ + "." + VCTrainProcessor.__name__
    acoustic_tokens_name: str = "acoustic_tokens"  # primary data stream, which we align to
    pitch_name: str = "pitch"  # stream of pitch from pitch estimator
    pitch_val_shift: int = 30  # add this to pitch values to make sure that they suit as indexes
    phoneme_vocab_size: int = 37
    pad_value: float = 0.0
    pad_on_right: bool = False

    # keys to load phoneme info from npz
    phoneme_probs_name: str = "phoneme_probs"
    phoneme_indices_name: str = "phoneme_indices"
    name: str = "phoneme"
