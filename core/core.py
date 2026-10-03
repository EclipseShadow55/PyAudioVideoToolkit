import queue
from abc import ABCMeta, abstractmethod
from enum import StrEnum
from typing import Protocol



class Effect[OutType](Protocol, metaclass=ABCMeta):
    @abstractmethod
    def __next__(self) -> OutType: ...

    @abstractmethod
    def run_all(self) -> list[OutType]: ...

class FixedLengthEffect(Effect, Protocol):
    def __len__(self) -> int: ...