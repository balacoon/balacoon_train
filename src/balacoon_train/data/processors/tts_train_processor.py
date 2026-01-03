"""
Copyright 2025 Balacoon

combines data streams from npz processors, aligning them
and preparing data for text-to-speech training
"""

import logging
import zipfile
from dataclasses import dataclass
from typing import cast, Optional
import os
import json

import numpy as np
import torch

from balacoon_train.config import Config
from balacoon_train.data.container import Container
from balacoon_train.data.processors.processor import Processor, ProcessorConfig
from balacoon_train.data.processors.tensor_mock import TensorMock
from balacoon_train.data.processors.vc_ops import resample_phonemes, resample_pitch


FRAME_RATE = 21.5


class TTSTrainProcessor(Processor):
    def __init__(self, config: Config):
        super().__init__(config)
        self._config.add_missing(TTSTrainProcessorConfig)
        if not os.path.isfile(self._config.phoneme_mapping):
            raise FileNotFoundError(
                f"Phoneme mapping file not found: {self._config.phoneme_mapping}"
            )
        self._phoneme_mapping = json.load(open(self._config.phoneme_mapping))

    def encode_phonemes(self, phonemes_str) -> torch.Tensor:
        return torch.tensor(
            [self._phoneme_mapping.get(p, 0) for p in phonemes_str], dtype=torch.int32
        )

    def create_alignment(
        self,
        acoustic_tokens: torch.Tensor,
        phonemes: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        Helper function that processes starts and ends of phonemes, creating an alignment between
        phonemes and acoustic tokens.
        """
        assert starts.shape == ends.shape

        if round(starts[0].item()) != 0:
            # first phoneme does not start with 0, need to prepend artificial silence
            starts = torch.cat([torch.tensor([0.0]), starts])
            ends = torch.cat([torch.tensor([starts[1]]), ends])
            phonemes = torch.cat([torch.tensor([0]), phonemes])

        # compute how many frames each phoneme is.
        end_frames = torch.round(ends * FRAME_RATE).int()
        frame_durations = torch.diff(end_frames, prepend=torch.tensor([0]))
        total_duration = torch.sum(frame_durations).item()

        diff = abs(total_duration - acoustic_tokens.shape[0])
        is_valid = True
        if diff > 2:
            logging.warning(
                f"Total duration of phonemes {total_duration} does not match acoustic tokens {acoustic_tokens.shape[0]}"
            )
            is_valid = False

        if total_duration > acoustic_tokens.shape[0] and diff - 1 > frame_durations[-1].item():
            logging.warning(
                f"Duration mismatch {diff} can't be compensated by last phoneme {frame_durations[-1].item()}"
            )
            is_valid = False

        if total_duration > acoustic_tokens.shape[0]:
            # reduce duration of the last phoneme
            frame_durations[-1] -= diff
        elif total_duration < acoustic_tokens.shape[0]:
            # drop some frames from acoustic tokens
            acoustic_tokens = acoustic_tokens[:total_duration]

        # convert frame durations into phoneme indices that are used by torch.gather
        # e.g. frame_durations = [1, 2, 1, 3] -> phoneme_indices = [0, 1, 1, 2, 3, 3, 3]
        phoneme_indices = torch.repeat_interleave(
            torch.arange(len(frame_durations)), frame_durations
        )
        assert phoneme_indices.shape[0] == acoustic_tokens.shape[0]
        return acoustic_tokens, phonemes, phoneme_indices, is_valid

    def process(self, container: Container, validate: bool = False) -> bool:
        """
        takes container with all data streams loaded, alings them and prepares
        upsampling that connects text and acoustic tokens.
        """
        if validate:
            # check that all input streams are actually in the container
            if any(
                stream not in container
                for stream in [
                    self._config.acoustic_tokens_name,
                    self._config.phonemes_name,
                    self._config.phonemes_start_name,
                    self._config.phonemes_end_name,
                ]
            ):
                return False

        acoustic_tokens = container[self._config.acoustic_tokens_name]  # frames x vocabs
        phonemes_str = container[self._config.phonemes_name]  # num_phonemes

        # WARNING: checks in this processor relies on actual values,
        # so mocking should be disabled for phoneme npz processor
        if validate:
            for p in phonemes_str:
                if p not in self._phoneme_mapping:
                    logging.warning(f"Phoneme {p} not found in phoneme mapping")
                    return False

        phonemes = self.encode_phonemes(phonemes_str)
        starts = container[self._config.phonemes_start_name]  # num_phonemes
        ends = container[self._config.phonemes_end_name]  # num_phonemes

        acoustic_tokens, phonemes, phoneme_indices, is_valid = self.create_alignment(
            acoustic_tokens, phonemes, starts, ends
        )
        if validate:
            return is_valid

        # finally create a target for end-of-phoneme prediction
        # it's a binary flag that is 1 at the end of each phoneme
        # e.g. phoneme_indices = [0, 1, 1, 2, 3, 3, 3] -> end_of_phoneme = [1, 0, 1, 1, 0, 0, 1]
        end_of_phoneme = torch.cat(
            [
                (phoneme_indices[:-1] != phoneme_indices[1:]).int(),
                torch.tensor([1]),  # last frame is always end of phoneme
            ]
        )

        # put everything back to container for collation
        container[self._config.phonemes_name] = phonemes
        container[self._config.acoustic_tokens_name] = acoustic_tokens
        container[self._config.name] = phoneme_indices
        container[self._config.phonemes_flag_name] = end_of_phoneme
        return True

    def collate_prompt(
        self,
        batch_elements: list[Container],
        acoustic_tokens_name: str,
        phonemes_name: str,
        phoneme_indices_name: str,
        phonemes_flag_name: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        a helper function that aligns indices, acoustic tokens and phonemes of the prompt.
        collation happens with padding on the left.
        function returns:
            - batched acoustic tokens
            - batched phonemes
            - batch phoneme indices (for upsampling)
            - batched end-of-phoneme flags
            - tokens length - actual sequence length of each batch element in frames
            - phonemes length - actual sequence length of each batch element in phonemes
        """
        acoustic_tokens_lst = [element[acoustic_tokens_name] for element in batch_elements]
        batched_acoustic_tokens, tokens_len = self.pad_and_stack(
            acoustic_tokens_lst, axis=0, val=self._config.pad_value, on_right=False
        )

        # now check how phonemes and alignment should be adjusted according to padding of acoustic tokens
        phonemes_lst = [element[phonemes_name] for element in batch_elements]
        indices_lst = [element[phoneme_indices_name] for element in batch_elements]
        if phonemes_flag_name is not None:
            eop_lst = [element[phonemes_flag_name] for element in batch_elements]
        max_frames = torch.max(tokens_len).item()

        phonemes_padding = torch.zeros(len(batch_elements), dtype=torch.int32)
        for i, frames in enumerate(tokens_len.tolist()):
            if frames < max_frames:
                # add SIL phoneme at the beginning of the phonemes list
                # increment phonemes padding that will be used for masking
                phonemes_padding[i] += 1
                phonemes_lst[i] = torch.cat([torch.tensor([0]), phonemes_lst[i]])
                # we added a phoneme, adjust alignment so upsampling is still valid
                diff = max_frames - frames
                indices = indices_lst[i]
                # shift the indices because we added a phoneme at the beginning
                indices += 1
                # prepend 0s, that allows to simply stack indices
                padding = [0] * diff
                indices_lst[i] = torch.cat([torch.tensor(padding), indices])
                padding[-1] = 1
                if phonemes_flag_name is not None:
                    eop_lst[i] = torch.cat([torch.tensor(padding), eop_lst[i]])

        # now we need pad and stack phonemes, since the sequence length is different here
        batched_phonemes, phonemes_len = self.pad_and_stack(
            phonemes_lst, axis=0, val=0, on_right=False
        )

        # since we padded the phonemes, we need to shift the indices by amount of padding
        max_phonemes = torch.max(phonemes_len).item()
        for i, num_phonemes in enumerate(phonemes_len.tolist()):
            if num_phonemes < max_phonemes:
                # there was padding, need to shift indices
                diff = max_phonemes - num_phonemes
                indices_lst[i] += diff

        # if phoneme was added, it will be masked
        phonemes_len -= phonemes_padding

        # finally simply stack indices and eop, since they are already padded to have same shape
        batched_phoneme_indices = torch.stack(indices_lst)  # batch x max_frames
        batched_eop = None
        if phonemes_flag_name is not None:
            batched_eop = torch.stack(eop_lst)  # batch x max_frames

        return (
            batched_acoustic_tokens,
            batched_phonemes,
            batched_phoneme_indices,
            batched_eop,
            tokens_len,
            phonemes_len,
        )

    def collate(self, batch_elements: list[Container], batch: Container):
        """
        aligned collation of phonemes, acoustic tokens and their alignment.
        we collate acoustic tokens, then add SIL phoneme at the beginning and alter alignment
        """

        (
            batched_acoustic_tokens,
            batched_phonemes,
            batched_phoneme_indices,
            batched_eop,
            tokens_len,
            phonemes_len,
        ) = self.collate_prompt(
            batch_elements,
            self._config.acoustic_tokens_name,
            self._config.phonemes_name,
            self._config.name,
            self._config.phonemes_flag_name,
        )

        # put everything to the container
        batch[self._config.acoustic_tokens_name] = batched_acoustic_tokens
        batch["tokens_len"] = tokens_len

        batch[self._config.phonemes_name] = batched_phonemes
        batch["phonemes_len"] = phonemes_len

        batch[self._config.name] = batched_phoneme_indices  # batch x max_frames
        batch[self._config.phonemes_flag_name] = batched_eop  # batch x max_frames

        # last thing that needs to be done - create a prompt_len,
        # which specifies how much to mask during loss computation.
        # TODO: we dont respect phrase or even phoneme boundaries.
        # that means we learn to continue generating from any position in the sequence.
        seq_len = batch["tokens_len"]
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


@dataclass
class TTSTrainProcessorConfig(ProcessorConfig):
    cls: str = TTSTrainProcessor.__module__ + "." + TTSTrainProcessor.__name__
    acoustic_tokens_name: str = "acoustic_tokens"  # primary data stream, which we align to
    phonemes_name: str = "phonemes"
    phonemes_start_name: str = "starts"
    phonemes_end_name: str = "ends"
    phoneme_mapping: str = (
        "???"  # path to phoneme mapping created with scripts/create_phoneme_mapping.py
    )

    # overwrite default values
    pad_value: float = 0.0  # used to pad acoustic tokens
    pad_on_right: bool = False

    # name of the stream that is getting created by the processor
    name: str = "phoneme_indices"
    phonemes_flag_name: str = "end_of_phoneme"
