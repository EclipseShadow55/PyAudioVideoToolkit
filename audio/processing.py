import os

import librosa
import numpy as np
from resonators import ResonatorBank
from core import Effect, FixedLengthEffect


class AudioResonatorProcess(Effect):
    def __init__(self, filepath: os.PathLike | str, bar_count: int, min_freq: float | int = 31.25, max_freq: float | int = 16000, framerate: int = 60): ...