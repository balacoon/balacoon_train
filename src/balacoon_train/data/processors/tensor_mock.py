"""
Copyright 2022 Balacoon

mocks the tensor during validation
"""

from typing import Tuple


class TensorMock:
    """
    mocks tensors in data container
    during validation. instead of saving
    whole tensor, store this lightweight
    object that has shape field and still
    allows to perform data validation
    """

    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape

    @property
    def ndim(self):
        return len(self.shape)

    def transpose(self, i, j):
        new_shape = list(self.shape)
        new_shape[i] = self.shape[j]
        new_shape[j] = self.shape[i]
        return TensorMock(tuple(new_shape))
