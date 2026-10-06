from abc import ABCMeta, abstractmethod
from typing import Protocol
from collections.abc import Iterable


class Effect[OutType](Protocol):
    @abstractmethod
    def __next__(self) -> OutType | StopIteration: ...

class FixedLengthEffect[OutType](Effect[OutType]):
    @abstractmethod
    def __len__(self) -> int: ...