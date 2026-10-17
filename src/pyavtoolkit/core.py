from abc import abstractmethod
from typing import Protocol, Iterator, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class Effect[OutType](Iterator[OutType], Protocol):
    def __next__(self) -> OutType: ...

@runtime_checkable
class FixedLengthEffect[OutType](Effect[OutType], Protocol):
    def __len__(self) -> int: ...

@runtime_checkable
class IndexableEffect[OutType](FixedLengthEffect[OutType], Protocol):
    def __getitem__(self, ind: int) -> OutType: ...

    def get_range(self, inds: slice) -> list[OutType]: ...