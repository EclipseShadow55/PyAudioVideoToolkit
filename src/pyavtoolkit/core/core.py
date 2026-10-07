from abc import abstractmethod
from typing import Protocol


class Effect[OutType](Protocol):
    @abstractmethod
    def __next__(self) -> OutType: ...

class FixedLengthEffect[OutType](Effect[OutType], Protocol):
    @abstractmethod
    def __len__(self) -> int: ...

class IndexableEffect[OutType](Effect[OutType], Protocol):
    @abstractmethod
    def __getitem__(self, ind: int) -> OutType: ...