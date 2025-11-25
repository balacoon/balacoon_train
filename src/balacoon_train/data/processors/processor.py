"""
Copyright 2022 Balacoon

abstract processor of the data used in dataset
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, cast

import torch

from balacoon_train.data.container import Container
from balacoon_train.config import ConfigurableConfig, Config


class Processor(ABC):
    """
    Interface that specifies the functions that data processor should implement.
    It should be a configurable object, which has function `process`.
    """

    def __init__(self, config: Config):
        self._config = config

    @abstractmethod
    def process(self, container: Container, validate: bool = False) -> bool:
        """
        based on configuration, takes from container stream
        created by loader or other processor, adjusts or extracts
        some data and puts it back to a container

        Parameters
        ----------
        container: Container
            data container with loaded data that should be processed
        validate: bool
            whether to validate the data before/during processing.
            sometimes, when validation is enabled, processor
            creates dummy data, to avoid time-consuming IO

        Returns
        -------
        flag: bool
            whether processing is successful
        """
        pass

    @staticmethod
    def multi_pad_tensor(
        tensor: torch.Tensor,
        axis: List[int],
        pre_pad: List[int],
        post_pad: List[int],
        mode: str = "constant",
        val: float = 0.0,
    ) -> torch.Tensor:
        """
        helper function that pads given pytorch tensor. used in `pad_and_stack`, which
        in turn is used in almost all `collate` methods

        Parameters
        ----------
        tensor: torch.Tensor
            array to pad
        axis: List[int]
            list of axis across which to perform padding
        pre_pad: List[int]
            size to pad before tensor for each axis
        post_pad: List[int]
            size of pad after tensor across for each axis (typically used for batching)
        mode: str
            mode in which to pad as in https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        val: float
            defines which value to pad with if mode is `constant`

        Returns
        -------
        padded_tensor: torch.Tensor
            tensor padded across sequences, axis, so

                padded_tensor.shape[axis] == pre_pad + tensor.shape[axis] + post_pad

        """
        assert (
            len(axis) == len(pre_pad) == len(post_pad)
        ), "number of axis and pad sizes should match"
        if all([x == 0 for x in pre_pad]) and all([x == 0 for x in post_pad]):
            # no need to do padding
            return tensor
        pad_width = [0] * tensor.ndim * 2  # padding before and after for each dimension of a tensor
        for a, pre, post in zip(axis, pre_pad, post_pad):
            pad_width[a * 2 + 1] = pre
            pad_width[a * 2] = post
        pad_width.reverse()
        padded_tensor = torch.nn.functional.pad(tensor, pad_width, mode=mode, value=val)
        return padded_tensor

    @staticmethod
    def replicate_pad_tensor(tensor: torch.Tensor, axis: int, post_pad: int) -> torch.Tensor:
        """
        Perform post-padding by replicating the last frame.
        It is separated into a special case, because functional
        pad doesn't support replicate padding with arbitrary dimensions.
        Here padding is implemented with slice / expand / concat which
        likely is slower, but fine for data loading.

        Parameters
        ----------
        tensor: torch.Tensor
            tensor to pad
        axis: int
            axis across which to perform padding
        post_pad: int
            how much to add after tensor across given `axis`

        Returns
        -------
        padded_tensor: torch.Tensor
        """
        # need to slice last frame which will be replicated
        slice_indices = [slice(None)] * tensor.ndim
        slice_indices[axis] = slice(tensor.size(axis) - 1, tensor.size(axis))
        last_frame = tensor[tuple(slice_indices)]
        repeats = [-1] * tensor.ndim
        repeats[axis] = post_pad
        last_frame_replicated = last_frame.expand(tuple(repeats))
        padded_tensor = torch.cat((tensor, last_frame_replicated), dim=axis)
        return padded_tensor

    @staticmethod
    def pad_tensor(
        tensor: torch.Tensor,
        axis: int,
        pre_pad: int = 0,
        post_pad: int = 0,
        mode: str = "constant",
        val: float = 0.0,
    ) -> torch.Tensor:
        """
        Simplified interface of `multi_pad_tensor`, which pads across single axis
        """
        if mode == "replicate" and pre_pad == 0:
            return Processor.replicate_pad_tensor(tensor, axis, post_pad)
        return Processor.multi_pad_tensor(tensor, [axis], [pre_pad], [post_pad], mode, val)

    @staticmethod
    def pad_and_stack(
        tensors: List[torch.Tensor],
        axis: int,
        mode: str = "constant",
        val: float = 0.0,
        on_right: bool = True,
        multiple_of: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        helper function that takes tensors belonging to the batch, defines maximum length in the batch,
        pads all the tensors to that length across sequence axis and then stack those tensors into a single tensor.
        After padding all dimensions of the tensors are supposed to be the same. Stacking introduces extra dimension -
        0th `batch` dimension.

        Parameters
        ----------
        tensors: List[torch.Tensor]
            list of tensors to combine into batch
        axis: int
            sequence axis, which should be padded to max length in order to combine tensors
        mode: str
            mode in which to pad, see https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        val: float
            which value to pad with if mode is `constant`
        on_right: bool
            whether to pad on the right, which is usually the case
        multiple_of: int
            whether total sequence length should be a multiple of this number

        Returns
        -------
        batch: torch.Tensor
            a single tensor, where input tensors are combined
        seq_len: torch.Tensor
            original lengths of the sequences before padding.
            added to container at collate, so can be used for masking
            or restoring original sequences.
        """
        tensor_lens = torch.tensor([x.shape[axis] for x in tensors], dtype=torch.int)
        max_len = max(tensor_lens)
        if multiple_of > 1:
            # Increase max_len to be divisible by multiple_of
            max_len = ((max_len + multiple_of - 1) // multiple_of) * multiple_of
        padded_tensors = []
        for tensor in tensors:
            to_pad = max_len - tensor.shape[axis]
            post_pad = pre_pad = 0
            if on_right:
                post_pad = to_pad
            else:
                pre_pad = to_pad
            tensor = Processor.pad_tensor(
                tensor, axis, pre_pad=pre_pad, post_pad=post_pad, mode=mode, val=val
            )
            padded_tensors.append(tensor)

        batch = torch.stack(padded_tensors)
        return batch, tensor_lens

    def collate(self, batch_elements: List[Container], batch: Container):
        """
        Collates data extracted by this processor. From each batch element
        takes a pytorch tensor previously extracted with `process` method,
        and then combines them into a single tensor corresponding to a batch.
        Mostly this implementation covers the typical processor collating,
        but for some processors should be overwritten (for ex. disabled for
        data adjusters)
        """
        if not self._config.to_collate:
            # nothing to do here
            return
        # collation happens only on output of processors, i.e. torch tensors,
        # other data (NpzFile, LinguisticUtterance) should be dropped by now
        tensors = [cast(torch.Tensor, x[self._config.name]) for x in batch_elements]
        combined, seq_len = self.pad_and_stack(
            tensors,
            axis=self._config.axis,
            val=self._config.pad_value,
            on_right=self._config.pad_on_right,
        )
        # store both batch and sequence length to a container.
        # sequence length might be needed to mask textual encoders
        # in the padded regions
        batch[self._config.name] = combined
        batch[self._config.name + "_len"] = seq_len

    def _get_torch_type(self) -> torch.dtype:
        """
        helper function that gets a type from a config (py:attr:`ProcessorConfig.type`),
        converts it to a pytorch type and returns. It is used in creation of torch
        tensors during `process`.
        """
        type_str = self._config.type
        if type_str == "float32" or type_str == "float":
            return torch.float
        elif type_str == "int32" or type_str == "int":
            return torch.int
        else:
            raise RuntimeError("Can't convert [{}] to a torch type".format(type_str))


@dataclass
class ProcessorConfig(ConfigurableConfig):
    """
    configuration of a processor.
    """

    cls: str = "???"
    type: str = "float32"  # in which to create a data
    name: str = "???"  # name under which to store the extracted data
    axis: int = 0  # sequence axis in extracted data; during batching, this dimension is padded
    pad_value: float = 0.0
    pad_on_right: bool = True
    to_collate: bool = True
