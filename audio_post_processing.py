import numpy as np


# TODO: Rewrite all as Effect classes, lazy evaluation
# TODO: Add way to also process played audio

def normalize_heights(arr: np.ndarray):
    return arr / np.max(arr)

def flatten(arr: np.ndarray, alpha: float | int = 1/2):
    return arr ** alpha

def lpf_smoothing(arr: np.ndarray, alpha: float | int = 0.3):
    ret = np.zeros((arr.shape[0], arr.shape[1]))
    ret[0] = arr[0]

    for i in range(1, arr.shape[0]):
        ret[i] = alpha * ret[i - 1] + (1 - alpha) * arr[i]

    return ret[1:]

def ballistic_smoothing(arr: np.ndarray, attack_time: float | int = 0.005, decay_time: float | int = 0.25, fps: int = 60):
    attack_factor = 1 - np.e ** (- 1 / fps / attack_time)
    decay_factor = 1 - np.e ** (- 1 / fps / decay_time)

    ret = np.zeros((arr.shape[0], arr.shape[1]))
    ret[0] = arr[0]

    for i in range(1, arr.shape[0]):
        factors = np.where(arr[i] > ret[i - 1], attack_factor, decay_factor)
        ret[i] = factors * arr[i] + (1 - factors) * ret[i - 1]

    return ret

def interpolate_heights(arr: np.ndarray):
    ret_arr = np.zeros((arr.shape[0] * 2, arr.shape[1]))
    base_arr = np.vstack([arr[0], arr, arr[0]])

    ret_arr[0::2] = arr

    p0s = base_arr[0:arr.shape[0] - 1]
    p1s = base_arr[1:arr.shape[0]]
    p2s = base_arr[2:arr.shape[0] + 1]
    p3s = base_arr[3:arr.shape[0] + 2]

    term1s = 9/16 * (p1s + p2s)
    term2s = 1/16 * (p0s + p3s)

    new_frames = term1s - term2s
    ret_arr[1:ret_arr.shape[0] - 1:2] = new_frames

    extra_frame = 15/8 * arr[-1] - 5/4 * arr[-2] + 3/8 * arr[-3]

    ret_arr[-1] = extra_frame

    return ret_arr