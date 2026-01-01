"""
Copyright 2022 Balacoon

npz stream extractor
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


class NpzStreamProcessor(Processor):
    """
    processor that takes loaded npz archive and extracts particular stream out of it.
    """

    def __init__(self, config: Config):
        """
        simple constructor of `NpzStreamProcessor` as a configurable object.
        """
        super().__init__(config)
        self._config.add_missing(NpzStreamProcessorConfig)

    def process(self, container: Container, validate: bool = False) -> bool:
        """
        takes loaded npz archive out of container and extracts
        particular data stream out of it, which is then put
        back to container

        Parameters
        ----------
        container: Container
            data container with :py:attr:`NpzStreamProcessorConfig.loader`
            in it. extracts :py:attr:`NpzStreamProcessorConfig.stream` out of it.
        validate: bool
            if specified, validates that stream is actually in npz. Also checks
            if size of tensor across sequence axis is within expected range.
            Lastly don't actually load the data at validation, but create a dummy one,
            to save time on IO

        Returns
        -------
        flag: bool
            if tensor from npz is loaded successfully
        """
        if validate:
            assert (
                self._config.loader in container
            ), "{} is missing from container, can't extract {}".format(
                self._config.loader, self._config.name
            )
        npz = cast(np.lib.npyio.NpzFile, container[self._config.loader])
        npz_stream_name = self._config.npz_name
        if not npz_stream_name:
            npz_stream_name = self._config.name
        if validate:
            if npz_stream_name not in npz.files:
                logging.warning(
                    "{} is not present for {} in {}. there are {}".format(
                        npz_stream_name,
                        self._config.loader,
                        container.get_id(),
                        str(npz.files),
                    )
                )
                return False
            # get shape of the array without reading the array
            assert npz.fid is not None, "can't retrieve npz file location"
            with zipfile.ZipFile(npz.fid.name) as archive:
                npy = archive.open(npz_stream_name + ".npy")
                version = np.lib.format.read_magic(npy)
                # check https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/format.py#L483
                if version[0] == 1:
                    arr_header = np.lib.format.read_array_header_1_0(npy)
                elif version[0] == 2:
                    arr_header = np.lib.format.read_array_header_2_0(npy)
                else:
                    raise RuntimeError("Unexpected numpy array version: " + str(version))
                shape = arr_header[0]
            if self._config.transpose:
                shape = shape[::-1]
            if 0 < self._config.max_len < shape[self._config.axis]:
                logging.warning(
                    "{} in {} is longer than {}, skip".format(
                        self._config.name, container.get_id(), self._config.max_len
                    )
                )
                return False
            if 0 < self._config.min_len > shape[self._config.axis]:
                logging.warning(
                    "{} in {} is shorter than {}, skip".format(
                        self._config.name, container.get_id(), self._config.min_len
                    )
                )
                return False
            # add fake data for validation
            if self._config.mock:
                container[self._config.name] = TensorMock(shape)
                return True

        arr = npz[npz_stream_name]
        if self._config.transpose:
            arr = arr.transpose()
        if self._config.torch:
            arr = torch.tensor(arr, dtype=self._get_torch_type())
        container[self._config.name] = arr
        return True


@dataclass
class NpzStreamProcessorConfig(ProcessorConfig):
    """
    configuration of npz stream processor.
    defines what is the name of npz loader to extract data for
    and stream to extract
    """

    cls: str = NpzStreamProcessor.__module__ + "." + NpzStreamProcessor.__name__
    loader: str = "???"  # name of npz loader to extract data for
    max_len: int = -1  # max sequence length allowed
    min_len: int = -1  # min sequence length allowed
    npz_name: str = ""  # if not specified `name` is used
    name: str = "???"  # data stream in npz archive to extract
    transpose: bool = False  # whether to transpose the data
    torch: bool = True  # whether to convert data to torch tensor
    mock: bool = True  # whether to mock data during validation
