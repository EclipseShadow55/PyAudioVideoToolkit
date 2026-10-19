from abc import ABC
from abc import abstractmethod
from typing import Iterator
from collections.abc import Sequence


class Effect[OutType](Iterator[OutType], ABC):
    @abstractmethod
    def __next__(self) -> OutType: ...

class FixedLengthEffect[OutType](Effect[OutType], ABC):
    @abstractmethod
    def __len__(self) -> int: ...


class IndexableEffect[OutType](FixedLengthEffect[OutType], ABC):
    @abstractmethod
    def __getitem__(self, ind: int | slice) -> OutType | Sequence[OutType]: ...