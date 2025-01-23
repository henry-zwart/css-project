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
