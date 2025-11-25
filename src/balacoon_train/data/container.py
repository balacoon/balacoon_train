"""
Copyright 2022 Balacoon

implementation container for the data,
to be passed at data loading and between models
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Union, cast

import numpy as np
import soundfile as sf
import torch


from balacoon_train.data.processors.tensor_mock import TensorMock

"""
specifies what can be stored in container:
tensors, arrays, npz archives or linguistic utterances
"""
TensorType = Union[np.ndarray, torch.Tensor, TensorMock]
try:
    from balacoon_frontend import LinguisticUtterance
    DataType = Union[TensorType, LinguisticUtterance, np.lib.npyio.NpzFile, sf.SoundFile]
except ImportError:
    DataType = Union[TensorType, np.lib.npyio.NpzFile, sf.SoundFile]


class Container:
    """
    holds named data streams. `Container` is passed
    between data loaders/extractors/adjusters.
    Also between network models and modules.
    Usually it is used as in/out parameter,
    i.e. it provides some inputs to the object,
    and outputs are also put into it.
    """

    def __init__(self, ids: Union[str, List[str]]):
        if isinstance(ids, str):
            ids = [ids]
        self._ids: List[str] = ids
        self._data: Dict[str, DataType] = dict()
        # maps name of the `OtherUtteranceSelector`
        # to alternative utterance id
        self._other_ids: Dict[str, str] = dict()

    def get_id(self) -> str:
        """
        Returns name of the utterance that is stored in the container.
        Should be called only when container stores data for a single
        utterance. If container stores data for a batch, this method
        will trigger assertion

        Returns
        -------
        id: str
            name of the utterance stored in container
        """
        assert len(self._ids) == 1, "requested id from container with multiple ids"
        return self._ids[0]

    def set_alternative_id(self, name: str, uttid: str):
        """
        Adds an alternative utterance id for `get_id()`.
        `name` is defines which `OtherUtteranceSelector` produced
        this alternative, so data loaders/processors can retrieve
        specific one. This method can only be executed when
        data container carries a single utterance id.

        Parameters
        ----------
        name: str
            name of `OtherUtteranceSelector` that produced
            this alternative. Should be used as a key at retrieval
        uttid: str
            alternative utterance id, that should be a pair with
            current `get_id()`.
        """
        assert (
            len(self._ids) == 1
        ), "Can't set alternative id, container contains multiple utterances"
        self._other_ids[name] = uttid

    def get_alternative_id(self, name: str) -> str:
        """
        Retrieves alterntaive utterance id set by `set_alternative_id`.
        Is used by data loaders/processors that process alternative/contrastive
        utterance for the given utterance.

        Parameters
        ----------
        name: str
            name of the `OtherUtteranceSelector` that produced alternative utterance id
            and was used in `set_alternative_id`

        Returns
        -------
        uttid: str
            alternative utterance id, previously added with `set_alternative_id`
        """
        assert name in self._other_ids, "No alternative utterance id for [{}]".format(
            name
        )
        return self._other_ids[name]

    def get_ids(self) -> List[str]:
        """
        Getter for names of utterances stored in container.

        Returns
        -------
        ids: List[str]
            list of utterance ids stored in container
        """
        return self._ids

    def __contains__(self, name: str) -> bool:
        """
        checks if container has certain data stream

        Parameters
        ----------
        name: str
            name of the data stream

        Returns
        -------
        flag: bool
            True if requested data is stored in this container
        """
        return name in self._data

    def __setitem__(
        self, names: Union[List[str], str], tensors: Union[List[DataType], DataType]
    ):
        """
        Adds the data into container. There are 2 options:

            - adding multiple data streams, when both names and tensors are lists
            - adding just a single data stream and corresponding data

        Parameters
        ----------
        names: Union[List[str], str]
            single or multiple names of data streams added to the container
        tensors: tensors: Union[List[DataType], DataType]
            single or multiple tensors (acutal data) added to the container
        """
        if isinstance(names, List) and isinstance(tensors, List):
            for n, t in zip(names, tensors):
                self[n] = t
        else:
            assert isinstance(names, str) and not isinstance(
                tensors, List
            ), "names and tensors should match when added to container"
            if (
                isinstance(tensors, (torch.Tensor, np.ndarray))
                and len(self._ids) > 1
                and np.prod(tensors.shape) > 1  # type: ignore
            ):
                # check that tensor has the same batch size as number of utterances
                # stored in container
                assert tensors.shape[0] == len(
                    self._ids
                ), "Trying to add tensor {}, with batch-size {} that doesn't match number if ids {}".format(
                    names, tensors.shape[0], len(self._ids)
                )
            self._data[names] = tensors

    def __getitem__(
        self, names: Union[List[str], str]
    ) -> Union[List[DataType], DataType]:
        """
        Getter for the data stored in container

        Parameters
        ----------
        names: Union[List[str], str]
            one or multiple data stream names to get

        Returns
        -------
        tensors: Union[List[DataType], DataType]
            one or multiple data streams stored in container
        """
        if isinstance(names, str):
            assert names in self._data, "{} is missing from the container".format(names)
            return self._data[names]
        else:
            return [self._data[n] for n in names]

    def keys(self) -> List[str]:
        """
        Getter for streams in the container.
        Mainly for debug.
        """
        return list(self._data.keys())

    def get(self, name: str, default: Optional[DataType] = None) -> DataType:
        """
        getter similar to __getitem__, which can fall back to default value
        """
        return self._data.get(name, default)

    def pop(self, name: str) -> DataType:
        """
        forwards pop method for underlying data dictionary
        """
        return self._data.pop(name)

    def to(self, rank) -> Container:
        """
        Similar to torch.Tensor.to(.) copies
        container to particular device

        Parameters
        ----------
        rank: int
            device to copy the data container to.
            if negative - means CPU device

        Returns
        -------
        copy: Container
            copy of the container but all the
            torch tensors are moved to specified
            device
        """
        copy = Container(self.get_ids())
        if isinstance(rank, int):
            device = torch.device("cpu") if rank < 0 else torch.device("cuda", rank)
        else:
            device = rank
        for key, val in self._data.items():
            if isinstance(val, torch.Tensor):
                copy._data[key] = val.to(device)
            else:
                copy._data[key] = val
        return copy

    def save_to_npz(
        self,
        streams: List[str],
        out_dir: str,
        actual_lens: List[str] = None,
        seq_axis: List[int] = None,
        rename: List[str] = None,
    ):
        """
        Stores streams from container to npz archives, based on utterance ids.
        If `actual_lens` is provided, slices only effective lens for each utterance, dropping padding.
        If `rename` is not provided, stores tensors under name from `streams`. If it is provided,
        that name is used instead.

        Parameters
        ----------
        streams: List[str]
            list of data streams to save.
        out_dir: str
            directory to store npz archives to. directory should already exist
        actual_lens: List[str]
            optional names of streams with actual lens of the `streams` before padding.
            if provided, should be the same length as streams list.
        seq_axis: List[int]
            sequence axis across which the array was padded. If `actual_lens` is provided,
            arrays will be sliced across these axis
        rename: List[str]
            names to use instead of `streams` as keys to npz archives.
            if provided - should be the same length as streams.
        """
        if actual_lens:
            assert seq_axis is not None
            assert len(actual_lens) == len(
                seq_axis
            ), "mismatch between sequence axis and actual lens tensors"
            assert len(streams) == len(
                actual_lens
            ), "list of keys for sequence length tensors has mismatching length"
        if rename:
            assert len(streams) == len(
                rename
            ), "list of new names for tensors has mismatching length"
        os.makedirs(out_dir, exist_ok=True)
        for i, name in enumerate(self.get_ids()):
            path = os.path.join(out_dir, name + ".npz")

            data_dict = {}
            if os.path.isfile(path):
                # if npz archive already exists, append data streams to it
                data_dict = dict(np.load(path))

            for j, stream in enumerate(streams):
                # assumes 0th dimension is batch one
                full_arr = self[stream]
                if isinstance(full_arr, torch.Tensor):
                    full_arr = full_arr.detach().cpu().numpy()
                arr = full_arr[i]
                if actual_lens:
                    # take seq lens tensor out of container based on the name
                    seq_len = self[actual_lens[j]]
                    assert (
                        seq_len.ndim == 1
                    ), "{} should be a sequence length tensor".format(actual_lens[j])
                    # select sequence length for the utterance based on batch_idx
                    seq_len_val = int(seq_len[i])
                    assert seq_axis is not None  # since actual_lens is provided
                    slice_axis = seq_axis[j]
                    # need to slice array according to actual lens of the tensor, dropping the padding
                    slice_indices = [slice(None)] * arr.ndim
                    slice_indices[slice_axis] = slice(seq_len_val)
                    arr = arr[tuple(slice_indices)]
                if rename:
                    # change stream name to store with if rename provided
                    stream = rename[j]
                # put into data dict
                data_dict[stream] = arr
            # save the data dict as npz
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, **data_dict)

    def save_to_wav(
        self,
        name: str,
        out_dir: str,
        actual_lens: torch.Tensor = None,
        sampling_rate: int = 24000,
    ):
        """
        Saves specified stream as audio files into directory.

        Parameters
        ----------
        name: str
            name of the data stream to save. should refer to tensor of shape (batch x samples_num)
        out_dir: str
            directory to store audio to. if it doesn't exist - will be created
        actual_lens: torch.Tensor
            actual lens of audio files in samples, not to store the padding.
            if not provided, audio files will be stored as is, but at the end of each audio file
            there will be extra silence
        sampling_rate: int
            sampling rate to save audio with
        """
        assert name in self, "Can't save {} to wav, it's not in the container".format(
            name
        )
        os.makedirs(out_dir, exist_ok=True)
        batch_audio = cast(torch.Tensor, self[name]).detach().cpu().numpy()
        assert batch_audio.ndim == 2, "Tensor to save as wav should have 2 dimensions"
        for i, name in enumerate(self.get_ids()):
            out_path = os.path.join(out_dir, name + ".wav")
            audio = batch_audio[i, :]  # assumes batch dimension is 0 in audio
            if actual_lens is not None:
                audio = audio[: int(actual_lens[i])]
            sf.write(out_path, audio, sampling_rate)
