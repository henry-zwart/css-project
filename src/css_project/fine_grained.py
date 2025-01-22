import numpy as np
from scipy import signal

KERNEL_NEIGHBOUR_COUNT = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int64)
KERNEL_NEIGHBOUR_COUNT_3D = KERNEL_NEIGHBOUR_COUNT[None, ...]


class FineGrained:
    N_STATES = 2

    grid: np.ndarray
    nutrients: np.ndarray

    def __init__(
        self,
        width: int = 128,
        nutrient_level: float = 1.0,
        nutrient_consume_rate: float = 0.1,
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
        self.grid = np.zeros((width, width), dtype=np.int64)
        self.nutrients = np.zeros_like(self.grid, dtype=np.float64) + nutrient_level
        self.nutrient_level = nutrient_level
        self.nutrient_consume_rate = nutrient_consume_rate
        self.width = width

    def initial_grid(self, p: float):
        """Randomly initialise grid with occupation probability p."""
        random_matrix = np.random.random(self.grid.shape)
        self.grid = np.where(random_matrix < p, 1, 0)

    def update(self):
        """Update grid state according to transition rules.

        First occupy new cells. Then consume nutrients.
        """
        n_occupied = count_neighbours(self.grid)
        self.grid[np.where(n_occupied >= 3)] = 1
        self.nutrients[np.where(self.grid)] -= self.nutrient_consume_rate
        self.grid[np.where(self.nutrients < self.nutrient_consume_rate)] = 0


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
