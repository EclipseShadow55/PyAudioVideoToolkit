import os

import librosa
import numpy as np
from resonators import ResonatorBank


# TODO: Rewrite entirely as classes
# TODO: Provide multiple processing methods (resonators, FFT, CQT, etc)
# TODO: Provide multiple scaling types (linear, exponential, etc)
# TODO: Should automatically cache in a .cache file, not npz but json, with hash of input file to
"""
, multiple scaling types, should automatically cache after use, not in npz but in specific cache file,
"""

def process_audio_file(filepath: os.PathLike | str, bar_count: int, min_freq: float | int = 31.25, max_freq: float | int = 16000, framerate: int = 60):
    samples, sample_rate = librosa.load(filepath, sr=None, mono=True)

    samples = samples.astype(np.float32)

    frequencies = min_freq * np.logspace(0, np.log(max_freq / min_freq) / np.log(2), base=2, num=bar_count, dtype=np.float32)
    bank = ResonatorBank(frequencies, sample_rate)

    duration = samples.shape[0] / sample_rate
    chunks = np.floor(np.linspace(0, samples.shape[0], num=int(framerate * duration))).astype(np.int64)

    results = np.zeros((chunks.shape[0] - 1, frequencies.shape[0]), dtype=np.float64)

    for i in range(chunks.shape[0] - 1):
        bank.resonate(samples[chunks[i]:chunks[i + 1]], chunks[i + 1] - chunks[i])
        results[i] = bank.magnitudes()

    return results, frequencies, duration

def save_processed_audio(filepath: os.PathLike | str, results: np.ndarray, frequencies: np.ndarray, duration: float | int):
    return np.savez_compressed(filepath, allow_pickle=False, processed_audio=results, frequency_bins=frequencies, total_time=duration)

def load_processed_audio(filepath: os.PathLike | str):
    return np.load(filepath, allow_pickle=False)