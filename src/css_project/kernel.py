import numpy as np

NEIGHBOUR_COUNT = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int64)
NEIGHBOUR_COUNT_3D = NEIGHBOUR_COUNT[None, ...]
center = 6
radius = 6
x_offset = 0
y_offset = 0
y, x = np.ogrid[:13, :13]
distance_from_origin = np.sqrt(
    (x - (center + x_offset)) ** 2 + (y - (center + y_offset)) ** 2
)
NEIGHBOUR_COUNT_R6 = (distance_from_origin <= 6).astype(int)

"""Only Moore neighbourhood supported for now."""
N_NEIGHBOURS = 8


def nutrient_diffusion_kernel(rate: float):
    """Prepare nutrient diffusion kernel for convolution"""
    kernel = np.full(
        (3, 3),
        fill_value=rate / N_NEIGHBOURS,
        dtype=np.float64,
    )
    kernel[1, 1] = 1 - rate
    return kernel


def neighbour_count_kernel(radius: int = 1) -> np.ndarray:
    """Prepare activator-inhibotor kernel for convolution"""
    if radius < 1:
        raise ValueError(f"Radius must be at least 1, found: {radius}.")

    side_length = 2 * radius + 1
    kernel = np.ones((side_length, side_length), dtype=np.int64)
    kernel[(side_length // 2), (side_length // 2)] = 0
    return kernel
