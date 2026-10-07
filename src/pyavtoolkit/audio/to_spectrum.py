import os
from abc import ABC

import librosa
import numpy as np
import numpy.typing as npt
from resonators import ResonatorBank
from collections.abc import Iterator

from pyavtoolkit.core import ScalingTypes


class ToSpectrumEffect(Iterator, ABC):
    sample_rate: int
    framerate: int
    frequencies: list[float| int]
    frame_count: int
    index: int
    samples: npt.NDArray[np.float32]

    def __init__(self, audio: npt.NDArray[np.floating], sample_rate: int, bin_scaling: ScalingTypes.Scaling, framerate: int = 60):
        self.samples = audio.astype(np.float32)
        self.sample_rate = sample_rate
        self.framerate = framerate
        self.frequencies = [freq for freq in bin_scaling]

        self.frame_count = self.samples.shape[0] / self.sample_rate * framerate

        self.index = -1

class ResonatorToSpectrumEffect(ToSpectrumEffect):
    bank: ResonatorBank
    duration: float | int
    chunks: npt.NDArray[np.int64]

    def __init__(self, audio: npt.NDArray[np.floating], sample_rate: int, bin_scaling: ScalingTypes.Scaling, framerate: int = 60):
        super().__init__(audio, sample_rate, bin_scaling, framerate)

        continuous_frequencies = np.ascontiguousarray(np.array(self.frequencies, dtype=np.float64))
        self.bank = ResonatorBank(continuous_frequencies, self.sample_rate)

        self.duration = self.samples.shape[0] / self.sample_rate
        self.chunks = np.floor(np.linspace(0, self.samples.shape[0], num=int(framerate * self.duration))).astype(np.int64)


    def __next__(self) -> npt.NDArray[np.float64]:
        if self.index < self.chunks.shape[0] - 2:
            raise StopIteration()
        self.index += 1
        self.bank.resonate(self.samples[self.chunks[self.index]:self.chunks[self.index + 1]], self.chunks[self.index + 1] - self.chunks[self.index])
        return self.bank.magnitudes()

    def __len__(self):
        return self.frame_count

class CQTToSpectrumEffect(ToSpectrumEffect):
    lenfft: int

    def __init__(self, audio: npt.NDArray[np.floating], sample_rate: int, bin_scaling: ScalingTypes.Scaling, framerate: int = 60):
        super().__init__(audio, sample_rate, bin_scaling, framerate)



    def __len__(self):
        return self.frame_count

class FFTToSpectrumEffect(ToSpectrumEffect):
    def __init__(self, audio: npt.NDArray[np.floating], sample_rate: int, bin_scaling: ScalingTypes.Scaling, hop_overlap: float | int, framerate: int = 60):
        super().__init__(audio, sample_rate, bin_scaling, framerate)

        if 0.0 > hop_overlap >= 1.0:
            raise ValueError("hop_overlap must be in the range [0, 1)")
        self.hop_size = self.samples.shape[0] / self.sample_rate * framerate

    def __len__(self):
        return self.frame_count


if __name__ == "__main__":
    freq_scale = ScalingTypes.LogScaling(55.0, 880.0, 48)

    sr = 48000
    ts = np.arange(0, 1, 1/sr)
    audio1 = 3 * np.sin(ts * 55, dtype=np.float64)
    audio2 = 10 * np.sin(ts * 293, dtype=np.float64)
    audio3 = 2.5 * np.sin(ts * 220, dtype=np.float64)

    rts_effect = ResonatorToSpectrumEffect(audio1 + audio2 + audio3, sr, freq_scale, 60)

    first_out = next(rts_effect)
    print(type(first_out))
    print(first_out)