import numpy as np
import numpy.typing as npt
from collections.abc import Iterator

class IndividualCQT(Iterator):
    def __init__(self, freq: float | int, lenfft: int, audio: npt.NDArray[np.float32 | np.float64]):