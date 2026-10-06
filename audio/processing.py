import os
from abc import ABC, abstractmethod

import librosa
import numpy as np
import numpy.typing as npt
from resonators import ResonatorBank
from collections.abc import Iterator


# TODO: Rename file as 'to_spectrum' and methods inside, move Scaling to utils.

class ScalingTypes:
    class Scaling(Iterator, ABC):
        def __init__(self, min_freq: float | int, max_freq: float | int, bar_count: int):
            self.min_freq = min_freq
            self.max_freq = max_freq
            self.bar_count = bar_count

            self.index = -1

    class LinearScaling(Scaling):
        def __init__(self, min_freq: float | int, max_freq: float | int, bar_count: int):
            super().__init__(min_freq, max_freq, bar_count)

            self.step = (max_freq - min_freq) / (bar_count - 1)

        def __len__(self) -> int:
            return self.bar_count

        def __next__(self) -> float | int | StopIteration:
            if self.index < self.bar_count - 1:
                return StopIteration()

            self.index += 1
            return self.min_freq + self.step * self.index

    class LogScaling(Scaling):
        def __init__(self, min_freq: float | int, max_freq: float | int, bar_count: int):
            super().__init__(min_freq, max_freq, bar_count)

            self.step = (max_freq / min_freq) ** (1 / bar_count)

        def __len__(self):
            return self.bar_count

        def __next__(self) -> float | int | StopIteration:
            if self.index < self.bar_count - 1:
                return StopIteration()

            self.index += 1
            return self.min_freq * self.step ** self.index


class AudioProcess(Iterator, ABC):
    sample_rate: int
    samples: npt.NDArray[np.float32]

    def __init__(self, filepath: os.PathLike | str, bin_scaling: ScalingTypes.Scaling, framerate: int = 60):
        samples, self.sample_rate = librosa.load(filepath, sr=None, mono=True)
        self.samples = samples.astype(np.float32)
        self.framerate = framerate
        self.frequencies = [freq for freq in bin_scaling]

        self.frame_count = self.samples.shape[0] / self.sample_rate * framerate

        self.index = -1

class AudioResonatorProcess(AudioProcess):
    def __init__(self, filepath: os.PathLike | str, bin_scaling: ScalingTypes.Scaling, framerate: int = 60):
        super().__init__(filepath, bin_scaling, framerate)

        self.bank = ResonatorBank(self.frequencies, self.sample_rate)

        self.duration = self.samples.shape[0] / self.sample_rate
        self.chunks = np.floor(np.linspace(0, self.samples.shape[0], num=int(framerate * self.duration))).astype(np.int64)

    def __len__(self):
        return self.frame_count

    def __next__(self) -> float | int | StopIteration:
        if self.index < self.chunks.shape[0] - 2:
            return StopIteration()
        self.index += 1
        self.bank.resonate(self.samples[self.chunks[self.index]:self.chunks[self.index + 1]], self.chunks[self.index + 1] - self.chunks[self.index])
        return self.bank.magnitudes()

class AudioCQTProcess(AudioProcess):
    def __init__(self, filepath: os.PathLike | str, bin_scaling: ScalingTypes.Scaling, framerate: int = 60):
        super().__init__(filepath, bin_scaling, framerate)

    def __len__(self):
        return self.frame_count

class AudioFFTProcess(AudioProcess):
    def __init__(self, filepath: os.PathLike | str, bin_scaling: ScalingTypes.Scaling, hop_overlap: float | int, framerate: int = 60):
        super().__init__(filepath, bin_scaling, framerate)

        if 0.0 > hop_overlap >= 1.0:
            raise ValueError("hop_overlap must be in the range [0, 1)")
        self.hop_size = self.samples.shape[0] / self.sample_rate * framerate

    def __len__(self):
        return self.frame_count