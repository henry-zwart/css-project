import numpy as np
from numpy.random import default_rng
from scipy import signal

KERNEL_NEIGHBOUR_COUNT = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int64)
KERNEL_NEIGHBOUR_COUNT_3D = KERNEL_NEIGHBOUR_COUNT[None, ...]


class FineGrained:
    N_STATES = 2
    N_NEIGHBOURS = 8

    grid: np.ndarray
    nutrients: np.ndarray

    def __init__(
        self,
        width: int = 128,
        nutrient_level: float = 1.0,
        nutrient_consume_rate: float = 0.1,
        nutrient_diffusion_rate: float = 0.1,
        random_seed: int | None = 42,
    ):
        if not (0.0 <= nutrient_level <= 1.0):
            raise ValueError(
                f"Global nutrient level must be in range: [0, 1], "
                f"found {nutrient_level}."
            )
        if not (0.0 <= nutrient_consume_rate <= 1.0):
            raise ValueError(
                f"Nutrient consumption rate must be in range: [0, 1], "
                f"found {nutrient_consume_rate}."
            )
        elif nutrient_consume_rate > nutrient_level:
            raise ValueError(
                f"Nutrient consumption rate must not exceed global nutrient level, "
                f"found {nutrient_consume_rate} > {nutrient_level}."
            )
        self.rng = default_rng(random_seed)
        self.grid = np.zeros((width, width), dtype=np.int64)
        self.width = width
        self.nutrients = np.zeros_like(self.grid, dtype=np.float64) + nutrient_level
        self.nutrient_level = nutrient_level
        self.nutrient_consume_rate = nutrient_consume_rate

        self.nutrient_diffusion_kernel = np.full(
            (3, 3),
            fill_value=nutrient_diffusion_rate / self.N_NEIGHBOURS,
            dtype=np.float64,
        )
        self.nutrient_diffusion_kernel[1, 1] = 1 - nutrient_diffusion_rate

    def initial_grid(self, p: float):
        """Randomly initialise grid with occupation probability p."""
        random_matrix = self.rng.random(self.grid.shape)
        self.grid = np.where(random_matrix < p, 1, 0)

    def update(self):
        """Update grid state according to transition rules.

        First occupy new cells. Then consume nutrients.
        """
        n_occupied = count_neighbours(self.grid)
        self.grid[np.where(n_occupied >= 3)] = 1
        self.nutrients[np.where(self.grid)] -= self.nutrient_consume_rate
        self.grid[np.where(self.nutrients < self.nutrient_consume_rate)] = 0
        self.diffuse_nutrients()

    def diffuse_nutrients(self):
        total_nutrients_before = self.nutrients.sum()
        self.nutrients = signal.convolve2d(
            self.nutrients, self.nutrient_diffusion_kernel, mode="same", boundary="fill"
        )
        total_nutrients_after = self.nutrients.sum()
        nutrients_lost = total_nutrients_before - total_nutrients_after
        self.nutrients += nutrients_lost / self.width**2


def count_neighbours(states: np.ndarray) -> np.ndarray:
    """Count the neighbours of each cell in a grid.

    If the input array is 2D, assumed to be the binary states
    at each row, col.

    If the input array is 3D, the first dimension is assumed
    to be a 'species' dimension, with the full shape being
    (species, row, column). Each layer is assumed to be
    a 2D array as in the 2D case.

    In the 2D case, returns the number of neighbours for each
    cell. In the 3D case, separates these counts by species,
    returning a matrix of the same shape as the input array.
    """
    match states.ndim:
        case 2:
            return signal.convolve2d(
                states,
                KERNEL_NEIGHBOUR_COUNT,
                mode="same",
                boundary="fill",
            )
        case 3:
            return signal.convolve(
                states,
                KERNEL_NEIGHBOUR_COUNT_3D,
                mode="same",
            )
        case _:
            raise ValueError("Grid shape should be either 2 or 3 dimensions.")
