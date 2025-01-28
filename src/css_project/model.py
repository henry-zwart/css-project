"""Base class for vegetation cellular automata models."""

from abc import ABC, abstractmethod

import numpy as np


class VegetationModel(ABC):
    """2D vegetation cellular automata abstract base class.

    Attributes:
        width: The number of cells in one side of the cellular automata grid.
        area: The total number of cells in the cellular automata grid.
        proportion_alive_list: A sequence of floats describing the proportion of
            living cells at each step in the cellular automata's history.
    """

    grid: np.ndarray
    width: int
    area: int
    proportion_alive_list: list[float]

    def __init__(
        self,
        width: int,
        species_prop: float | np.ndarray | list[float],
        init_method: str,
    ):
        """Initialises the cellular automata.

        Args:
            width: The number of cells in one side of the cellular automata grid.
            species_prop: A float, or sequence of floats, describing the initial
                occurrence probability of each species. The probability of 'dead
                cells' is omitted. The order reflects the order of the integers
                representing the species'.
            init_method: The method used to initialise the grid with species.
        """
        self.width = width
        self.area = width * width
        self.proportion_alive_list = []
        self.initial_grid(species_prop, type=init_method)

    @property
    @abstractmethod
    def n_states(self) -> int:
        """The number of represented states, equal to 1 + the number of species'."""
        ...

    @abstractmethod
    def update(self):
        """Transition one timestep on the cellular automata grid.

        Update the cells via the specified transition rules, as well as any
        environmental features.
        """
        ...

    def initial_grid(self, p: float | np.ndarray | list[float], type="random"):
        """Initialise species' populations in the grid.

        After sampling from U(0,1) for each cell in the grid, the inverse-transform
        method is used to assign cells to the model's species' (including dead cells)
        in their specified proportions.

        Args:
            p: A float, or sequence of floats, describing the initial
                occurrence probability of each species. The probability of 'dead
                cells' is omitted. The order reflects the order of the integers
                representing the species'.
            type: The method used to initialise the grid (**currently unused**).

        Raises:
            ValueError: Supplied species probabilities don't permit a valid probability
            distribution (i.e., their sum exceeds 1.0).
        """
        match p:
            case float(prob):
                probs = np.array([prob])
            case _:
                probs = np.asarray(p)

        zero_prob = 1 - probs.sum()
        if zero_prob < 0.0:
            raise ValueError("Species proportions sum must not exceed 1.0.")

        probs = np.concat(([0, zero_prob], probs))
        prob_ranges = np.cumsum(probs)

        grid = np.zeros((self.width, self.width), dtype=np.int64)
        samples = np.random.random(grid.shape)
        for i in range(len(prob_ranges) - 1):
            update_cells = (prob_ranges[i] <= samples) & (samples < prob_ranges[i + 1])
            grid[update_cells] = i

        self.grid = grid

    def total_alive(self) -> int:
        """Count the total number of living cells in the grid.

        Returns:
            The number of cells with non-zero state.
        """
        alive = (self.grid > 0).sum()
        return alive

    def species_alive(self) -> list[int]:
        """Calculate the frequency of each species in the grid.

        Returns:
            For each species, the number of cells with that state. Ordered by
            species identifier.
        """
        x = [(self.grid == i).sum() for i in range(1, self.n_states)]
        return x

    def proportion_alive(self) -> float:
        """Count the proportion of living cells in the grid.

        Returns:
            The proportion of cells with non-zero state.
        """
        return self.total_alive() / self.area

    def is_steady_state(self, threshold: float = 0.001) -> bool:
        """Determine whether the grid is in a steady state compared to its history.

        Estimates the derivative of the proportion of living cells with respect
        to the prior 21 updates.

        Args:
            threshold: Threshold under which the first derivative is considered stable.

        Returns:
            True if the current state is determined to be at equilibrium. Otherwise
            False.
        """
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
            if abs(der) < threshold and abs(der_2) < threshold:
                return True
        return False

    def find_steady_state(self, iterations: int):
        """Run the cellular automata until a steady state is reached.

        Args:
            iterations: Maximum number of iterations before termination.
        """
        for _ in range(iterations):
            if self.is_steady_state():
                break
            self.update()

    def run(self, iterations: int = 1000):
        """Run the cellular automata for a specified number of iterations
        Args:
            iterations: number of iterations to be executed

        """
        for _ in range(iterations):
            self.update()
