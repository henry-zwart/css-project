import numpy as np
from numpy.random import default_rng
from scipy import signal

from . import kernel


class FineGrained:
    """Fine grained model with nutrient diffusion
    Attributes:
        N_STATES (int): Number of states a cell can be int
        N_STATES (int): Number of neighbours
        rng (int): Random seed
        grid (2D np.array): Status of the vegetetation in the grid
        width (int): Width of the grid
        area (int): Total number of cells in the grid
        nutrients (2D np.array): Status of nutrient avaibility in the grid
        plant_matter (2D np.array): Status of nutrients present in vegetation
        compost (2D np.array): Status of decomposing matter in the grid
        nutrient_level (float): Amount of nutrient present
            at each cell in the beginning of simulation
        nutrient_consume_rate (float): Rate at which nutrients are consumed
            by vegetation
        nutrient_regenerate_rate (float): Rate at which nutrients are recovered
        nutrient_diffusion_kernel (object): Kernel to be used for simulation
        proportion_alive_list (list): Stores number of alive cells at each step
        original_grid (2D np.array): Copy of initial grid
    """

    N_STATES = 2
    N_NEIGHBOURS = 8

    grid: np.ndarray
    nutrients: np.ndarray

    def __init__(
        self,
        width: int = 128,
        nutrient_level: float = 1.0,
        nutrient_consume_rate: float = 0.1,
        nutrient_diffusion_rate: float = 0.21,
        nutrient_regenerate_rate: float = 0.8,
        random_seed: int | None = 42,
    ):
        if any(
            x < 0.0
            for x in (nutrient_level, nutrient_consume_rate, nutrient_diffusion_rate)
        ):
            raise ValueError(
                "Global nutrient level, nutrient consumption rate, and "
                "nutrient diffusion rate must all be non-negative."
            )
        elif nutrient_consume_rate > nutrient_level:
            raise ValueError(
                f"Nutrient consumption rate must not exceed global nutrient level, "
                f"found {nutrient_consume_rate} > {nutrient_level}."
            )
        self.rng = default_rng(random_seed)
        self.grid = np.zeros((width, width), dtype=np.int64)
        self.width = width
        self.area = width * width
        self.nutrients = np.zeros_like(self.grid, dtype=np.float64) + nutrient_level
        self.plant_matter = np.zeros_like(self.nutrients)
        self.compost = np.zeros_like(self.nutrients)
        self.nutrient_level = nutrient_level
        self.nutrient_consume_rate = nutrient_consume_rate
        self.nutrient_regenerate_rate = nutrient_regenerate_rate
        self.nutrient_diffusion_kernel = kernel.nutrient_diffusion_kernel(
            nutrient_diffusion_rate
        )
        self.proportion_alive_list = []

    def initial_grid(self, p: float):
        """Randomly initialise grid with occupation probability p."""
        random_matrix = self.rng.random(self.grid.shape)
        self.grid = np.where(random_matrix < p, 1, 0)
        self.original_grid = self.grid.copy()

    def reset(self):
        """Resets grid to original configuration"""
        self.grid = self.original_grid.copy()
        self.nutrients = (
            np.zeros_like(self.grid, dtype=np.float64) + self.nutrient_level
        )
        self.plant_matter = np.zeros_like(self.nutrients)
        self.compost = np.zeros_like(self.nutrients)

    def total_alive(self):
        """Counts total number of alive cells in the grid."""

        alive = self.grid.sum()

        return alive

    def is_steady_state(self):
        """Checks whether a steady state is reached using first order and second
        order difference"""

        # Selects 21 last iterations to look at a trend instead of local fluctuation
        # and the index of the middle term of second difference was
        # a whole integer
        if (len(self.proportion_alive_list)) > 21:
            der = self.proportion_alive_list[-21] - self.proportion_alive_list[-1]
            der_2 = (
                self.proportion_alive_list[-21]
                + self.proportion_alive_list[-1]
                - 2 * self.proportion_alive_list[-11]
            )
            if abs(der) < 0.001 and abs(der_2) < 0.001:
                return True
        return False

    def find_steady_state(self, iterations):
        """Run simulation until steady state is reached"""
        for _ in range(iterations):
            if self.is_steady_state():
                break
            self.update()

    def update(self):
        """Update grid state according to transition rules.

        Order of operations:
        1. Consume nutrients. If nutrient level insufficient --> die.
        2. Vegetation spreads to nearby cells.
        3. Diffuse nutrients.
        """
        is_occupied = self.grid == 1
        has_sufficient_nutrients = self.nutrients > self.nutrient_consume_rate
        reduce_nutrients = is_occupied & has_sufficient_nutrients
        exhaust_nutrients = is_occupied & ~has_sufficient_nutrients

        # Regenerate nutrients in soil from dead plants
        sufficient_compost = self.compost > self.nutrient_regenerate_rate
        self.nutrients[sufficient_compost] += self.nutrient_regenerate_rate
        self.compost[sufficient_compost] -= self.nutrient_regenerate_rate
        self.nutrients[~sufficient_compost] += self.compost[~sufficient_compost]
        self.compost[~sufficient_compost] = 0.0

        # Plants consume nutrients
        self.plant_matter[reduce_nutrients] += self.nutrient_consume_rate
        self.nutrients[reduce_nutrients] -= self.nutrient_consume_rate

        # If insufficient nutrients, exhaust supply and kill plant
        self.plant_matter[exhaust_nutrients] += self.nutrients[exhaust_nutrients]
        self.nutrients[exhaust_nutrients] = 0.0
        self.grid[exhaust_nutrients] = 0
        self.compost[exhaust_nutrients] += self.plant_matter[exhaust_nutrients]
        self.plant_matter[exhaust_nutrients] = 0.0

        # Spread vegetation to nearby cells
        n_occupied = count_neighbours(self.grid)
        self.grid[np.where(n_occupied >= 3)] = 1

        # Diffuse nutrients
        self.diffuse_nutrients()

        self.proportion_alive_list.append(self.total_alive() / self.area)

    def diffuse_nutrients(self):
        """Diffuse nutrients across the grid"""
        total_nutrients_before = self.nutrients.sum()
        self.nutrients = signal.convolve2d(
            self.nutrients, self.nutrient_diffusion_kernel, mode="same", boundary="fill"
        )
        total_nutrients_after = self.nutrients.sum()
        nutrients_lost = total_nutrients_before - total_nutrients_after
        self.nutrients += nutrients_lost / self.width**2

    def system_nutrients(self):
        "Return the total amount of nutrients in the system"
        return self.nutrients.sum() + self.plant_matter.sum() + self.compost.sum()


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
                kernel.NEIGHBOUR_COUNT,
                mode="same",
                boundary="fill",
            )
        case 3:
            return signal.convolve(
                states,
                kernel.NEIGHBOUR_COUNT_3D,
                mode="same",
            )
        case _:
            raise ValueError("Grid shape should be either 2 or 3 dimensions.")
