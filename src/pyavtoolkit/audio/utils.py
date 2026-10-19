import numpy as np
import numpy.typing as npt
from collections.abc import Iterator
from pyavtoolkit.core.abc import FixedLengthEffect


class FFTEffect(Iterator):
    lenfft: int
    audio: npt.NDArray[np.float32]

    def __init__(self, lenfft: int, audio: FixedLengthEffect[float | int]):
        self.lenfft = lenfft

        self.audio = np.asarray(audio)