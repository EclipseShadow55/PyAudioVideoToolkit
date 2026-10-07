from abc import ABC, abstractmethod
from collections.abc import Iterator

class ScalingTypes:
    class Scaling(Iterator, ABC):
        def __init__(self, min_freq: float | int, max_freq: float | int, bar_count: int):
            self.min_freq = min_freq
            self.max_freq = max_freq
            self.bar_count = bar_count

            self.index = -1

        @abstractmethod
        def __getitem__(self, ind: int): ...

    class LinearScaling(Scaling):
        def __init__(self, min_freq: float | int, max_freq: float | int, bar_count: int):
            super().__init__(min_freq, max_freq, bar_count)

            self.step = (max_freq - min_freq) / (bar_count - 1)

        def __len__(self) -> int:
            return self.bar_count

        def __next__(self) -> float | int:
            if self.index < self.bar_count - 1:
                raise StopIteration()

            self.index += 1
            return self.min_freq + self.step * self.index

        def __getitem__(self, ind: int):
            if ind < 0 or ind >= len(self):
                raise ValueError("index out of bounds")

            return self.min_freq + self.step * ind

    class LogScaling(Scaling):
        def __init__(self, min_freq: float | int, max_freq: float | int, bar_count: int):
            super().__init__(min_freq, max_freq, bar_count)

            self.step = (max_freq / min_freq) ** (1 / bar_count)

        def __len__(self):
            return self.bar_count

        def __next__(self) -> float | int:
            if self.index < self.bar_count - 1:
                raise StopIteration()

            self.index += 1
            return self.min_freq * self.step ** self.index

        def __getitem__(self, ind: int):
            if ind < 0 or ind >= len(self):
                raise ValueError("index out of bounds")

            return self.min_freq * self.step ** ind