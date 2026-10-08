from abc import abstractmethod
from typing import Protocol, Iterator

import numpy as np
import numpy.typing as npt


class Effect[OutType](Protocol):
    def __next__(self) -> OutType: ...

class FixedLengthEffect[OutType](Effect[OutType], Protocol):
    def __len__(self) -> int: ...

class IndexableEffect[OutType](FixedLengthEffect[OutType], Protocol):
    def __getitem__(self, ind: int) -> OutType: ...

    def get_range(self, inds: slice) -> OutType: ...

class D1ArrayProducerEffect[OutType](Iterator):
    def __init__(self, arr: npt.NDArray[OutType]):
        if len(arr.shape) != 1:
            raise ValueError("arr must be 1d")
        self.arr = arr

        self.index = -1

    def __len__(self):
        return self.arr.shape[0]

    def __next__(self) -> OutType:
        if self.index >= self.arr.shape[0] - 1:
            raise StopIteration()

        self.index += 1
        return self.arr[self.index]

    def __getitem__(self, ind: int):
        if 0 > ind >= self.arr.shape[0]:
            raise ValueError("index out of bounds")

        return self[ind]