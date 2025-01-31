import numpy as np
from numba import njit
from scipy import signal

from . import kernel
from .model import VegetationModel


@njit
def calculate_transition_probabilities(
    grid, nearby_count, competition, N_NEIGHBOURS, DELTA_T_R
):
    """Numba-optimized computation of transition probabilities."""
    rows, cols = grid.shape
    transition_prob = np.zeros_like(grid, dtype=np.float64)

    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0:  # Unoccupied
                if competition[i, j] >= 0:
                    transition_prob[i, j] = (
                        DELTA_T_R
                        * (nearby_count[i, j] / (N_NEIGHBOURS - nearby_count[i, j]))
                        * competition[i, j]
                    )
                else:
                    transition_prob[i, j] = 0
            else:  # Occupied
                if competition[i, j] < 0:
                    transition_prob[i, j] = 1 + DELTA_T_R * competition[i, j]
                else:
                    transition_prob[i, j] = 1

    return np.clip(transition_prob, 0, 1)


class Logistic(VegetationModel):
    """Logistic model
    Attributes:
        N_NEIGHBOURS (int): Number of neighbours
        grid (2D np.array): Status of the vegetetation in the grid
        nutrients (2D np.array): Status of nutrient avaibility in the grid
        width (int): Width of the grid
        consume_rate (float): Rate at which nutrients are consumed by vegetation
        control (float): Nutrient supplement rate
        random_seed (int): Random seed at each cell in the beginning of simulation
        alive_prop (float): Proportion of alive cells at the beginning
        init_method (str): Method to initialize the grid
    """

    N_NEIGHBOURS = kernel.NEIGHBOUR_COUNT_R6.sum()

    grid: np.ndarray
    nutrients: np.ndarray

    def __init__(
        self,
        width: int = 128,
        consume_rate: float = 63.7,
        control: float = 35.0,
        random_seed: int | None = 42,
        alive_prop: float = 0.01,
        init_method: str = "random",
    ):
        super().__init__(width, alive_prop, init_method, random_seed)
        self.transition_prob = np.zeros_like(self.grid, dtype=np.float64)
        self.consume_rate = consume_rate
        self.supplement_rate = control
        self.proportion_alive_list = []

    @property
    def n_states(self) -> int:
        return 2

    def set_control(self, value):
        self.supplement_rate = value

    def calculate_competition(self, nearby_vegetation: np.ndarray) -> np.ndarray:
        """
        Calculate the competition term of the adapted logistic equation. 
        """
        return 1 - (self.consume_rate * nearby_vegetation) / (
            self.supplement_rate * self.N_NEIGHBOURS
        )

    def update(self):
        """Update grid state according to transition rules."""
        nearby_count = count_neighbours(self.grid)
        competition = self.calculate_competition(nearby_count)
        DELTA_T_R = 0.5 * 2
        self.transition_prob = calculate_transition_probabilities(
            self.grid, nearby_count, competition, self.N_NEIGHBOURS, DELTA_T_R
        )

        # Update the grid using the transition probabilities
        non_zero_prob = self.transition_prob > 0
        self.grid[non_zero_prob] = self.rng.binomial(
            1, self.transition_prob[non_zero_prob]
        )

        # Record proportion of alive cells
        self.proportion_alive_list.append(self.total_alive() / self.area)

        # Reset transition probabilities
        self.transition_prob.fill(0)


class LogisticTwoNative(VegetationModel):
    """Logistic model including invasive species
    Attributes:
        N_NEIGHBOURS (int): Number of neighbours
        grid (2D np.array): Status of the vegetetation in the grid
        nutrients (2D np.array): Status of nutrient avaibility in the grid
        width (int): Width of the grid
        consume_rate_1 (float): Rate at which nutrients are consumed by native species
        consume_rate_2 (float): Rate at which nutrients are consumed by invasive species
        control (float): Nutrient supplement rate
        species_prop (list): proportion of species at the beginning
        init_method (str): Method to initialize the grid
        random_seed (int): Random seed at each cell in the beginning of simulation
    """

    N_NEIGHBOURS = kernel.NEIGHBOUR_COUNT_R6.sum()

    grid: np.ndarray
    nutrients: np.ndarray

    def __init__(
        self,
        width: int = 128,
        consume_rate_1: float = 63.7,
        consume_rate_2: float = 63.7,
        control: float = 35.0,
        species_prop: list[float] | tuple[float, float] | np.ndarray = (0.25, 0.25),
        init_method: str = "random",
        random_seed: int | None = 42,
    ):
        super().__init__(width, list(species_prop), init_method, random_seed)
        self.transition_prob = np.zeros((*self.grid.shape, 3), dtype=np.float64)
        self.consume_rate_1 = consume_rate_1
        self.consume_rate_2 = consume_rate_2
        self.supplement_rate = control
        self.proportion_native_alive_list = []
        self.proportion_invasive_alive_list = []

    @property
    def n_states(self) -> int:
        return 3

    def set_control(self, value):
        self.supplement_rate = value

    def calculate_competition(
        self, count_species_1: np.ndarray, count_species_2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the competition term of the adapted logistic equation
        combining the consumption rates of both species. 
        """
        return 1 - (
            self.consume_rate_1 * count_species_1
            + self.consume_rate_2 * count_species_2
        ) / (self.supplement_rate * self.N_NEIGHBOURS)

    def update(self):
        """Update grid state according to transition rules.

        Order of operations:
        1. Consume nutrients. If nutrient level insufficient --> die.
        2. Vegetation spreads to nearby cells.
        3. Diffuse nutrients.
        """
        nearby_species_1 = count_neighbours(self.grid == 1)
        nearby_species_2 = count_neighbours(self.grid == 2)

        total_count = count_neighbours(self.grid != 0)
        competition = self.calculate_competition(nearby_species_1, nearby_species_2)

        unoccupied = self.grid == 0
        DELTA_T = 0.5
        R = 2

        state_prob = np.zeros((*self.grid.shape, 3))
        state_prob[unoccupied, 1] = np.where(
            competition[unoccupied] >= 0,
            DELTA_T
            * R
            * (nearby_species_1[unoccupied])
            / (self.N_NEIGHBOURS - total_count[unoccupied])
            * competition[unoccupied],
            0,
        )
        state_prob[unoccupied, 2] = np.where(
            competition[unoccupied] >= 0,
            DELTA_T
            * R
            * (nearby_species_2[unoccupied])
            / (self.N_NEIGHBOURS - total_count[unoccupied])
            * competition[unoccupied],
            0,
        )
        species_1_locs = self.grid == 1
        species_2_locs = self.grid == 2
        state_prob[species_1_locs, 1] = 1 - np.where(
            competition[species_1_locs] < 0,
            -DELTA_T * R * competition[species_1_locs],
            0,
        )
        state_prob[species_2_locs, 2] = 1 - np.where(
            competition[species_2_locs] < 0,
            -DELTA_T * R * competition[species_2_locs],
            0,
        )
        state_prob[..., 0] = 1 - state_prob[..., 1:].sum(axis=2)
        cumulative_prob = np.cumsum(state_prob, axis=2)

        assert np.isclose(cumulative_prob[..., -1], 1.0).all()

        p = self.rng.random(self.grid.shape)
        new_grid = np.argmax(p[..., None] <= cumulative_prob, axis=2)
        self.grid = new_grid

        self.proportion_alive_list.append(self.total_alive() / self.area)

        self.proportion_native_alive_list.append(self.species_alive()[0] / self.area)
        self.proportion_invasive_alive_list.append(self.species_alive()[1] / self.area)


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
    return signal.convolve(
        states,
        kernel.NEIGHBOUR_COUNT_R6,
        mode="same",
    )
