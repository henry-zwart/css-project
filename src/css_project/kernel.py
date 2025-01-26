import numpy as np

NEIGHBOUR_COUNT = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int64)
NEIGHBOUR_COUNT_3D = NEIGHBOUR_COUNT[None, ...]

"""Only Moore neighbourhood supported for now."""
N_NEIGHBOURS = 8


def nutrient_diffusion_kernel(rate: float):
    kernel = np.full(
        (3, 3),
        fill_value=rate / N_NEIGHBOURS,
        dtype=np.float64,
    )
    kernel[1, 1] = 1 - rate
    return kernel


def neighbour_count_kernel(radius: int = 1) -> np.ndarray:
    if radius < 1:
        raise ValueError(f"Radius must be at least 1, found: {radius}.")

    side_length = 2 * radius + 1
    kernel = np.ones((side_length, side_length), dtype=np.int64)
    kernel[(side_length // 2), (side_length // 2)] = 0
    return kernel
