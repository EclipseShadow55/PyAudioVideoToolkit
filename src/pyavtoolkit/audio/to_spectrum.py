from abc import ABC
from typing import Sequence

import numpy as np
import numpy.typing as npt
from resonators import ResonatorBank

from pyavtoolkit.core import Effect, FixedLengthEffect
from pyavtoolkit.core.effects import ScalingTypes

# Refactor all using ChunkProducerEffect to make fully lazy
class ToSpectrumEffect(FixedLengthEffect[np.floating], ABC):
    sample_rate: int
    framerate: int
    frequencies: list[float| int]
    frame_count: int
    index: int
    samples: npt.NDArray[np.float32]

    def __init__(self,
                 audio: Effect[float | int] | Sequence[float | int] | npt.NDArray[np.floating],
                 sample_rate: int,
                 bin_scaling: Effect[float | int] | Sequence[float | int],
                 framerate: int = 60):
        self.samples = np.asarray(audio)
        self.sample_rate = sample_rate
        self.framerate = framerate
        self.frequencies = [freq for freq in bin_scaling]

        self.frame_count = self.samples.shape[0] / self.sample_rate * framerate

        self.index = -1

class ResonatorToSpectrumEffect(ToSpectrumEffect):
    bank: ResonatorBank
    duration: float | int
    chunks: npt.NDArray[np.int64]

    def __init__(self,
                 audio: Effect[float | int] | Sequence[float | int] | npt.NDArray[np.floating],
                 sample_rate: int,
                 bin_scaling: Effect[float | int] | Sequence[float | int],
                 framerate: int = 60):
        super().__init__(audio, sample_rate, bin_scaling, framerate)

        self.bank = ResonatorBank(np.asarray(self.frequencies, dtype=np.float32), self.sample_rate)

        self.duration = self.samples.shape[0] / self.sample_rate
        self.chunks = np.floor(np.linspace(0, self.samples.shape[0], num=int(framerate * self.duration))).astype(np.int64)

    def __next__(self) -> npt.NDArray[np.float32]:
        if self.index > self.chunks.shape[0] - 2:
            raise StopIteration()
        self.index += 1
        self.bank.resonate(self.samples[self.chunks[self.index]:self.chunks[self.index + 1]], self.chunks[self.index + 1] - self.chunks[self.index])
        return self.bank.magnitudes()

    def __len__(self):
        return self.frame_count

class CQTToSpectrumEffect(ToSpectrumEffect):
    lenfft: int
    kernal: npt.NDArray[np.float32]
    fft: 'FFTToSpectrumEffect'

    def __init__(self,
                 audio: Effect[float | int] | Sequence[float | int],
                 sample_rate: int,
                 bin_scaling: Effect[float | int] | Sequence[float | int],
                 quality: float | int,
                 hop_overlap: float | int,
                 framerate: int = 60):
        if 0.0 > hop_overlap >= 1.0:
            raise ValueError("hop_overlap must be in the range [0, 1)")

        super().__init__(audio, sample_rate, bin_scaling, framerate)

        filter_lens = np.round(self.sample_rate * quality / np.array(self.frequencies)).astype(np.int64)
        lenfft = np.max(filter_lens)

        self.k_hats = np.zeros((filter_lens.shape[0], lenfft))

        for i, flen in enumerate(filter_lens):
            ts = np.arange(flen)
            window = np.hanning(flen)

            c_exp = (1.0 / flen) * window * np.exp(2j * np.pi * self.frequencies[i] * ts)
            c_exp_padded = np.zeros(lenfft, dtype=np.complex128)

            start = (lenfft - flen) // 2
            c_exp_padded[start:start + flen] = c_exp

            k_hat_k = np.fft.fft(c_exp_padded)
            self.k_hats[i] = k_hat_k


    def __len__(self):
        return self.frame_count

class FFTToSpectrumEffect(ToSpectrumEffect):
    def __init__(self,
                 audio: Effect[float | int] | Sequence[float | int],
                 sample_rate: int,
                 bin_scaling: Effect[float | int] | Sequence[float | int],
                 hop_overlap: float | int,
                 framerate: int = 60):
        if 0.0 > hop_overlap >= 1.0:
            raise ValueError("hop_overlap must be in the range [0, 1)")

        super().__init__(audio, sample_rate, bin_scaling, framerate)



    def __len__(self):
        return self.frame_count


if __name__ == "__main__":
    freq_scale = ScalingTypes.LogScaling(55.0, 880.0, 48)

    sr = 48000
    ts = np.arange(0, 1, 1/sr)
    audio1 = 3 * np.sin(ts * 55 * np.pi * 2, dtype=np.float32)
    audio2 = 10 * np.sin(ts * 293 * np.pi * 2, dtype=np.float32)
    audio3 = 2.5 * np.sin(ts * 220 * np.pi * 2, dtype=np.float32)

    rts_effect = ResonatorToSpectrumEffect(audio1 + audio2 + audio3, sr, freq_scale, 60)

    print(list(zip(freq_scale.copy(), next(rts_effect))))