import numpy as np
from scipy import signal

from css_project.kernel import neighbour_count_kernel

from .model import VegetationModel

# np.random.seed(2)


class Vegetation(VegetationModel):
    """Vegetation cellular automata model for native species.

    Vegetation growth is governed by a neighborhood feedback rule. Positive
    and negative feedbacks scale linearly with the quantity of vegetation in
    separate radii. The former reflects the positive contribution of nearby
    vegetation to soil quality. The latter reflects the competition for local
    resources (nutrients).

    Attributes:
        positive_factor: Weight coefficient applied to positive feedback
        negative_factor: Weight coefficient applied to negative feedback
        close_kernel: Convolution kernel used to compute the number of nearby
            neighbors for positive feedback.
        far_kernel: Convolution kernel used to compute the number of nearby
            neighbors for negative feedback.
    """

    positive_factor: int
    negative_factor: int
    close_kernel: np.ndarray
    far_kernel: np.ndarray

    def __init__(
        self,
        width: int = 128,
        small_radius: int = 1,
        large_radius: int = 4,
        control: int = 7,
        negative_factor: int = 1,
        init_method="random",
        alive_prop: float = 0.5,
    ):
        """Initialise the Vegetation cellular automata model.

        Args:
            width: The number of cells in one side of the cellular automata grid.
            small_radius: The square radius within which to count neighbors for
                positive feedback.
            large_radius: The square radius within which to count neighbors for
                negative feedback.
            positive_factor: Weight coefficient applied to positive feedback.
            negative_factor: Weight coefficient applied to negative feedback.
            init_method: Method used to initialise population on the grid.
            alive_prop: Initial proportion of occupied cells.
        """
        super().__init__(width, alive_prop, init_method)
        self.positive_factor = control
        self.negative_factor = negative_factor
        self.close_kernel = neighbour_count_kernel(small_radius)
        self.far_kernel = neighbour_count_kernel(large_radius)

    @property
    def n_states(self) -> int:
        """Number of states in the model."""
        return 2

    def compute_feedback(self, n_close: np.ndarray, n_far: np.ndarray) -> np.ndarray:
        """Calculate feedback as a linear combination of neighbour frequencies.

        The feedback is rounded toward zero, and clipped to be in the range [-1, 1].
        Values after rounding and clipping are:
        - -1, if feedback <= -1
        - 0, if -1 < feedback < 1
        - 1, if 1 <= feedback

        Args:
            n_close: For each cell, the number of nearby neighbors.
            n_far: For each cell, the number of distance neighbors.

        Returns:
            The feedback value for each cell in the grid.
        """
        raw_feedback = self.positive_factor * n_close - self.negative_factor * n_far
        return np.clip(np.fix(raw_feedback), -1, 1).astype(int)

    def update(self):
        """Perform a single transition on the grid.

        Calculate the feedback for each cell in the grid.
        Feedback:
        - **-1:** Cell dies (or stays dead)
        - **0**: Cell retains its state
        - **1**: Cell becomes alive (or stays alive)
        """
        close_neighbours = count_neighbours(self.grid, self.close_kernel)
        far_neighbours = count_neighbours(self.grid, self.far_kernel)
        feedback = self.compute_feedback(close_neighbours, far_neighbours)
        self.grid[feedback < 0] = 0
        self.grid[feedback > 0] = 1
        self.proportion_alive_list.append(self.total_alive() / self.area)


class InvasiveVegetation(VegetationModel):
    def __init__(
        self,
        width: int = 128,
        small_radius: int = 1,
        large_radius: int = 4,
        control: int = 8,
        neg_factor_nat: int = 1,
        neg_factor_inv: int = 1,
        init_method="random",
        species_prop: list[float] | tuple[float, float] | np.ndarray = (0.25, 0.25),
    ):
        super().__init__(width, list(species_prop), init_method)
        self.small_radius = small_radius
        self.large_radius = large_radius
        self.pos_factor = control
        self.neg_factor_nat = neg_factor_nat
        self.neg_factor_inv = neg_factor_inv
        self.close_kernel = neighbour_count_kernel(small_radius)
        self.far_kernel = neighbour_count_kernel(large_radius)
        self.proportion_native_alive_list = []
        self.proportion_invasive_alive_list = []

    @property
    def n_states(self) -> int:
        return 3

    def introduce_invasive(self, p_inv=0.1, type="empty"):
        if type == "random":
            random_matrix = np.random.random(self.grid.shape)
            self.grid[np.where(random_matrix <= p_inv)] = 2
        elif type == "empty":
            empty_grid = self.grid == 0
            random_cells = (empty_grid).sum()
            random_nrs = np.random.random(random_cells)
            self.grid[empty_grid] = np.where(random_nrs < p_inv, 2, 0)
        else:
            raise ValueError("No valid type for invasive introduction")

    def compute_feedback(
        self,
        positive_factor,
        n_close: np.ndarray,
        n_far_nat: np.ndarray,
        n_far_inv: np.ndarray,
    ) -> np.ndarray:
        """Calculate feedback as a linear combination of neighbour frequencies.

        The feedback is rounded toward zero, and clipped to be in the range [-1, 1].
        Values after rounding and clipping are:
        - -1, if feedback <= -1
        - 0, if -1 < feedback < 1
        - 1, if 1 <= feedback

        Args:
            n_close: For each cell, the number of nearby neighbors.
            n_far: For each cell, the number of distance neighbors.

        Returns:
            The feedback value for each cell in the grid.
        """
        raw_feedback = positive_factor * n_close - (
            (n_far_nat * self.neg_factor_nat) + (n_far_inv * self.neg_factor_inv)
        )

        return np.clip(np.fix(raw_feedback), -1, 1).astype(int)

    def update(self):
        close_neighbours_nat = count_neighbours(
            self.grid == 1, self.close_kernel
        )  # correct
        close_neighbours_inv = count_neighbours(
            self.grid == 2, self.close_kernel
        )  # correct

        # since some states are 2, will it be disproportionally doubled now?
        far_neighbours_nat = count_neighbours(
            self.grid == 1, self.far_kernel
        )  # correct
        far_neighbours_inv = count_neighbours(
            self.grid == 2, self.far_kernel
        )  # correct

        feedback_nat = self.compute_feedback(
            self.pos_factor,
            close_neighbours_nat,
            far_neighbours_nat,
            far_neighbours_inv,
        )

        feedback_inv = self.compute_feedback(
            self.pos_factor,
            close_neighbours_inv,
            far_neighbours_nat,
            far_neighbours_inv,
        )

        # Cell states
        dead_cell = self.grid == 0
        native_cell = self.grid == 1
        invasive_cell = self.grid == 2

        # Feedback statements
        feedback_nat_eq_inv = feedback_nat == feedback_inv
        feedback_nat_gt_0 = feedback_nat > 0
        feedback_inv_gt_0 = feedback_inv > 0
        feedback_nat_st_0 = feedback_nat < 0
        feedback_inv_st_0 = feedback_inv < 0
        feedback_nat_gt_inv = feedback_nat > feedback_inv
        feedback_inv_gt_nat = feedback_inv > feedback_nat

        # Find cells where we need to choose between 1 and 2
        cells_eq = (dead_cell & feedback_nat_eq_inv & feedback_nat_gt_0).sum()
        random_nrs = np.random.random(cells_eq)
        self.grid[dead_cell & feedback_nat_eq_inv & feedback_nat_gt_0] = np.where(
            random_nrs < 0.5, 1, 2
        )

        # Other updates for Dead cells
        self.grid[dead_cell & feedback_nat_gt_0 & feedback_nat_gt_inv] = 1
        self.grid[dead_cell & feedback_inv_gt_0 & feedback_inv_gt_nat] = 2

        # Updates for native cells
        self.grid[native_cell & feedback_nat_st_0] = 0

        # Updates for invasive cells
        self.grid[invasive_cell & feedback_inv_st_0] = 0

        self.proportion_native_alive_list.append(self.species_alive()[0] / self.area)
        self.proportion_invasive_alive_list.append(self.species_alive()[1] / self.area)


def count_neighbours(states: np.ndarray, kern: np.ndarray) -> np.ndarray:
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
    return signal.convolve(states, kern, mode="same")
