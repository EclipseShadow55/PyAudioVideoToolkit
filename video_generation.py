import enum

import numpy as np


# TODO: rewrite as class, called on creation objects, lazy eval, lazy computation

def compile_frame_arrays(frame_gen: VisualizerTypes,
                         heights_arrays: np.ndarray, v_area: int, h_area: int,
                         fg_colors: np.ndarray, bg_colors: np.ndarray = np.array([0, 0, 0]),
                         **kwargs):
    if len(heights_arrays.shape) != 2:
        raise ValueError("height must be 1D array")
    if v_area <= 0:
        raise ValueError("v_area must be > 0")
    if h_area <= 0:
        raise ValueError("h_area must be > 0")
    if len(fg_colors.shape) not in [1, 2]:
        raise ValueError("fg_colors must be a 1D or 2D array")
    if fg_colors.shape[-1] not in [3, 4]:
        raise ValueError("fg_colors must be of shape (n, 3 | 4)")
    if len(bg_colors.shape) not in [1, 2]:
        raise ValueError("bg_colors must be a 1D or 2D array")
    if bg_colors.shape[-1] not in [3, 4]:
        raise ValueError("bg_colors must be of shape (n, 3 | 4)")

    if len(fg_colors.shape) == 1:
        fg_colors = fg_colors[None, :]
    if len(bg_colors.shape) == 1:
        bg_colors = bg_colors[None, :]

    frame_arrs = np.zeros((heights_arrays.shape[0], v_area, h_area, 4), dtype=np.uint8)

    for frame_ind in range(heights_arrays.shape[0]):
        frame_arrs[frame_ind] = FRAME_GEN_METHODS[frame_gen.value - 1](heights_arrays[frame_ind], v_area, h_area,
                                          fg_colors[frame_ind % fg_colors.shape[0]], bg_colors[frame_ind % bg_colors.shape[0]], **kwargs)

    return frame_arrs


def square_bars_frame_gen(heights: np.ndarray, v_area: int, h_area: int,
                          fg_color: np.ndarray, bg_color: np.ndarray = np.array([0, 0, 0]),
                          width: int = 10, bidirectional: bool = False, **kwargs):
    if len(heights.shape) != 1:
        raise ValueError("height must be 1D array")
    if len(fg_color.shape) != 1:
        raise ValueError("fg_color must be a 1D array")
    if fg_color.shape[0] not in [3, 4]:
        raise ValueError("fg_color must be of shape (3 | 4,)")
    if len(bg_color.shape) != 1:
        raise ValueError("bg_color must be a 1D array")
    if bg_color.shape[0] not in [3, 4]:
        raise ValueError("bg_color must be of shape (3 | 4,)")

    frame_arr = np.full((v_area, h_area, 4), 255, dtype=np.uint8)
    frame_arr[:, :, 0:bg_color.shape[0]] = bg_color

    chunks = np.floor(np.linspace(0, h_area, num=heights.shape[0] + 1)).astype(np.int64)

    for i in range(chunks.shape[0] - 1):
        mid_point = (chunks[i + 1] - chunks[i]) // 2 + chunks[i]
        needle_area = mid_point - (width - 1) // 2, mid_point + width // 2 + 1
        px_height = int(heights[i] * (v_area - 1) + 1)
        px_shift = v_area // 2 - px_height // 2 if bidirectional else 0

        frame_arr[v_area - px_shift - px_height:v_area - px_shift, needle_area[0]:needle_area[1], 0:fg_color.shape[
            0]] = fg_color

    return frame_arr


def rounded_rects_frame_gen(heights: np.ndarray, v_area: int, h_area: int,
                            fg_color: np.ndarray, bg_color: np.ndarray = np.array([0, 0, 0]),
                            width: int = 10, bidirectional: bool = False, corner_radius: float | int = 0.3, **kwargs):
    if len(heights.shape) != 1:
        raise ValueError("height must be 1D array")
    if len(fg_color.shape) != 1:
        raise ValueError("fg_color must be a 1D array")
    if fg_color.shape[0] not in [3, 4]:
        raise ValueError("fg_color must be of shape (3 | 4,)")
    if len(bg_color.shape) != 1:
        raise ValueError("bg_color must be a 1D array")
    if bg_color.shape[0] not in [3, 4]:
        raise ValueError("bg_color must be of shape (3 | 4,)")
    if not 0 <= corner_radius <= 1:
        raise ValueError("corner_radius must be in the range of [0, 1]")

    frame_mask = np.zeros((v_area, h_area), dtype=np.bool)

    chunks = np.floor(np.linspace(0, h_area, num=heights.shape[0] + 1)).astype(np.int64)

    for i in range(chunks.shape[0] - 1):
        mid_point = (chunks[i + 1] - chunks[i]) // 2 + chunks[i]
        needle_area = mid_point - (width - 1) // 2, mid_point + width // 2 + 1
        px_height = int(heights[i] * (v_area - 1) + 1)
        px_shift = v_area // 2 - px_height // 2 if bidirectional else 0
        px_corner_radius = int(width / 2 * corner_radius)

        x_range = np.arange(*needle_area)

        tline_y_range = np.full(width - px_corner_radius * 2, px_height + px_shift)
        trcorner_y_range = px_height + px_shift - px_corner_radius + np.round(np.sqrt(px_corner_radius ** 2 - np.arange(px_corner_radius) ** 2))
        tlcorner_y_range = trcorner_y_range[::-1]

        t_y_range = np.hstack([tlcorner_y_range, tline_y_range, trcorner_y_range])
        b_y_range = (px_height + px_shift * 2 - t_y_range) if bidirectional else np.full(width, 0)

        t_y_range = v_area - t_y_range
        b_y_range = v_area - b_y_range

        y_range = np.arange(v_area)[:, None]
        frame_mask[:, x_range] = (y_range <= b_y_range) & (y_range >= t_y_range)

        # pure_height = px_height - int(width * corner_radius) * (2 if bidirectional else 1)

        # frame_arr[v_area - px_shift - pure_height:v_area - px_shift, needle_area[0]:needle_area[1], :fg_color.shape[0]] = fg_color

    frame_arr = np.full((v_area, h_area, 4), 255, dtype=np.uint8)
    frame_arr[:, :, :bg_color.shape[0]] = bg_color
    frame_arr[frame_mask, :fg_color.shape[0]] = fg_color

    return frame_arr




FRAME_GEN_METHODS = [square_bars_frame_gen, rounded_rects_frame_gen]


class VisualizerTypes(enum.Enum):
    SQUARE_BARS = enum.auto()
    ROUNDED_BARS = enum.auto()
