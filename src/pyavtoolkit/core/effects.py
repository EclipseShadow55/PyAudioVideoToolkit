import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

import numpy as np
import numpy.typing as npt

from pyavtoolkit.core import Effect, FixedLengthEffect, IndexableEffect
from pyavtoolkit.core.utils import check_slice


class ScalingTypes:
    class Scaling(Iterator, ABC):
        min_freq: float | int
        max_freq: float | int
        bar_count: int
        index: int

        def __init__(self, min_freq: float | int, max_freq: float | int, bar_count: int):
            self.min_freq = min_freq
            self.max_freq = max_freq
            self.bar_count = bar_count

            self.index = -1

        @abstractmethod
        def __len__(self) -> int: ...

        @abstractmethod
        def __getitem__(self, ind: int): ...

        @abstractmethod
        def copy(self) -> 'ScalingTypes.Scaling': ...

    class LinearScaling(Scaling):
        step: float | int

        def __init__(self, min_freq: float | int, max_freq: float | int, bar_count: int):
            super().__init__(min_freq, max_freq, bar_count)

            self.step = (max_freq - min_freq) / (bar_count - 1)

        def __next__(self) -> float | int:
            if self.index > self.bar_count - 1:
                raise StopIteration()

            self.index += 1
            return self.min_freq + self.step * self.index

        def __len__(self) -> int:
            return self.bar_count

        def __getitem__(self, ind: int):
            if ind < 0 or ind >= self.bar_count:
                raise ValueError("index out of bounds")

            return self.min_freq + self.step * ind

        def copy(self):
            return self.__class__(self.min_freq, self.max_freq, self.bar_count)

    class LogScaling(Scaling):
        step: float | int

        def __init__(self, min_freq: float | int, max_freq: float | int, bar_count: int):
            super().__init__(min_freq, max_freq, bar_count)

            self.step = (max_freq / min_freq) ** (1 / bar_count)

        def __len__(self):
            return self.bar_count

        def __next__(self) -> float | int:
            if self.index > self.bar_count - 1:
                raise StopIteration()

            self.index += 1
            return self.min_freq * self.step ** self.index

        def __getitem__(self, ind: int):
            if ind < 0 or ind >= self.bar_count:
                raise ValueError("index out of bounds")

            return self.min_freq * self.step ** ind

        def copy(self):
            return self.__class__(self.min_freq, self.max_freq, self.bar_count)

class SequenceEffect[OutType](IndexableEffect[OutType]):
    arr: Sequence[OutType]
    index: int

    def __init__(self, arr: Sequence[OutType]):
        self.arr = arr
        self.index = -1

    def __len__(self) -> int:
        return len(self.arr)

    def __next__(self) -> OutType:
        if self.index >= len(self.arr) - 1:
            raise StopIteration()

        self.index += 1
        return self.arr[self.index]

    def __getitem__(self, ind: int | slice) -> OutType | npt.NDArray[OutType]:
        if isinstance(ind, int):
            if 0 > ind >= len(self.arr):
                raise ValueError("index out of bounds")
        else:
            check_slice(ind, len(self))

        return self[ind]

class Np1DArrayEffect[OutType](IndexableEffect[OutType]):
    arr: npt.NDArray[OutType]
    index: int

    def __init__(self, arr: npt.NDArray[OutType]):
        if len(arr.shape) != 1:
            raise ValueError("arr must be 1d")

        self.arr = arr
        self.index = -1

    def __len__(self) -> int:
        return len(self.arr)

    def __next__(self) -> OutType:
        if self.index >= len(self.arr) - 1:
            raise StopIteration()

        self.index += 1
        return self.arr[self.index]

    def __getitem__(self, ind: int | slice) -> OutType | npt.NDArray[OutType]:
        if isinstance(ind, int):
            if 0 > ind >= len(self.arr):
                raise ValueError("index out of bounds")
        else:
            check_slice(ind, len(self))

        return self[ind]


class ChunkProducerEffect[OutType](FixedLengthEffect[OutType]):
    arr: npt.NDArray[OutType]
    num_chunks: int

    def __init__(self, arr: npt.NDArray[OutType], num_chunks: int, hop_overlap: float | int):
        self.arr = arr
        self.num_chunks = num_chunks
        self.hop_overlap = hop_overlap

    def __len__(self):
        return self.num_chunks


class BasicDuplicatorEffect[OutType](FixedLengthEffect[OutType]):
    def __init__(self, base: FixedLengthEffect[OutType], num: int):
        self.base = base
        self.buffer = []
        self.clients = [DuplicatorClientEffect(self) for _ in range(num)]

    def __len__(self) -> int:
        return len(self.base)

    def __next__(self) -> OutType:
        try:
            self.buffer.append(next(self.base))
            return self.buffer[-1]
        except StopIteration:
            raise StopIteration()

    def get(self, ind: int):
        if ind >= len(self):
            raise ValueError("index out of bounds")

        while ind >= len(self.buffer):
            next(self)

        return self.buffer[ind]

class OptimizedDuplicatorEffect[OutType](BasicDuplicatorEffect[OutType]):
    def __init__(self, base: FixedLengthEffect[OutType], num: int):
        super().__init__(base, num)
        self.shift = 0

    def __len__(self) -> int:
        return len(self.base)

    def __next__(self) -> OutType:
        try:
            self.buffer.append(next(self.base))
            return self.buffer[-1]
        except StopIteration:
            raise StopIteration()

    def get(self, ind: int) -> OutType:
        if ind >= len(self):
            raise ValueError("index out of bounds")

        while ind - self.shift >= len(self.buffer):
            next(self)

        self._optimize_buffer()

        return self.buffer[ind - self.shift]

    def _optimize_buffer(self):
        min_ind = min([client.index for client in self.clients])
        if min_ind > self.shift:
            self.buffer.pop(0)
            self.shift += 1

class DuplicatorClientEffect[OutType](FixedLengthEffect[OutType]):
    def __init__(self, server: BasicDuplicatorEffect[OutType]):
        self.server = server
        self.index = -1

    def __len__(self) -> int:
        return len(self.server)

    def __next__(self) -> OutType:
        if self.index >= len(self) - 1:
            raise StopIteration()

        self.index += 1
        return self.server.get(self.index)


def create_optimized_duplicator[OutType](
        base: FixedLengthEffect[OutType],
        num: int) -> tuple['OptimizedDuplicatorEffect[OutType]', list['DuplicatorClientEffect[OutType]']]:
    server = OptimizedDuplicatorEffect(base, num)
    clients = server.clients

    return server, clients

def as_effect[OutType](obj: Effect[OutType] | Sequence[OutType]) -> Effect[OutType]:
    if isinstance(obj, Effect):
        return obj
    elif isinstance(obj, np.ndarray) and len(obj.shape) == 1:
        return Np1DArrayEffect(obj)
    else:
        return SequenceEffect(obj)

def power_above(exp: float | int, bound: float | int) -> float | int:
    if exp <= 0 or exp == 1:
        raise ValueError("pow must be greater than zero and not equal to 1")
    if bound <= 0:
        raise ValueError("bound must be greater than zero")

    cur = exp
    mult = bound / cur
    power = math.floor(math.log(mult, exp)) + 1

    return cur * exp ** power

def get_chunk_size(arr_len: int, chunk_num: int, hop_overlap: float | int) -> float | int:
    return arr_len / ((chunk_num - 1) * (1 - hop_overlap) + 1)