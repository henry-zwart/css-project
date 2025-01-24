import numpy as np

from .model import VegetationModel

np.random.seed(2)


class Vegetation(VegetationModel):
    def __init__(
        self,
        width: int = 128,
        small_radius: int = 1,
        large_radius: int = 4,
        positive_factor: int = 7,
        negative_factor: int = 1,
        init_method="random",
        alive_prop: float = 0.5,
    ):
        super().__init__(width, alive_prop, init_method)
        self.small_radius = small_radius
        self.large_radius = large_radius
        self.positive_factor = positive_factor
        self.negative_factor = negative_factor

    @property
    def n_states(self) -> int:
        return 2

    def update(self):
        temp_grid = np.empty_like(self.grid)

        for y in range(self.width):
            for x in range(self.width):
                # Find local neighbors and calculate sum
                close = self.find_neighbors(y, x, self.small_radius)
                close_sum = self.find_states(close)[0]

                # Find non-local neighbors and calculate sum
                far = self.find_neighbors(y, x, self.large_radius)
                far_sum = self.find_states(far)[0]

                # Calculate positive and negative feedback
                feedback = (
                    self.positive_factor * close_sum - self.negative_factor * far_sum
                )

                # Add either -1, 0 or 1 to current state
                temp_grid[y, x] = self.grid[y, x] + max(-1, min(1, int(feedback)))

                # Ensure minimum/maximum possible values
                if temp_grid[y, x] < 0:
                    temp_grid[y, x] = 0
                if temp_grid[y, x] > 1:
                    temp_grid[y, x] = 1

        self.grid = temp_grid
        self.proportion_alive_list.append(self.total_alive() / self.area)


class InvasiveVegetation(VegetationModel):
    def __init__(
        self,
        width: int = 128,
        small_radius: int = 1,
        large_radius: int = 4,
        pos_factor_nat: int = 8,
        neg_factor_nat: int = 1,
        pos_factor_inv: int = 8,
        neg_factor_inv: int = 1,
        init_method="random",
        species_prop: list[float] | tuple[float, float] | np.ndarray = (0.25, 0.25),
    ):
        super().__init__(width, list(species_prop), init_method)
        self.small_radius = small_radius
        self.large_radius = large_radius
        self.pos_factor_nat = pos_factor_nat
        self.neg_factor_nat = neg_factor_nat
        self.pos_factor_inv = pos_factor_inv
        self.neg_factor_inv = neg_factor_inv

    @property
    def n_states(self) -> int:
        return 3

    def introduce_invasive(self, p_inv=0.1, type="random"):
        if type == "random":
            random_matrix = np.random.random(self.grid.shape)
            self.grid[np.where(random_matrix <= p_inv)] = 2

    def update(self):
        temp_grid = np.empty_like(self.grid)

        for y in range(self.width):
            for x in range(self.width):
                # Find local neighbors and calculate sum
                close = self.find_neighbors(y, x, self.small_radius)
                close_nat, close_inv = self.find_states(close)

                # Find non-local neighbors and calculate sum
                far = self.find_neighbors(y, x, self.large_radius)
                far_nat, far_inv = self.find_states(far)

                # Calculate negative feedback
                neg_feedback = self.neg_factor_nat * far_nat

                # Calculate positive and negative feedback
                feedback_nat = self.pos_factor_nat * close_nat - (
                    neg_feedback + (self.neg_factor_inv * far_inv)
                )

                feedback_inv = self.pos_factor_inv * close_inv - (
                    neg_feedback + (self.neg_factor_inv * far_inv)
                )

                # Empty cell
                if self.grid[y, x] == 0:
                    if feedback_nat == feedback_inv and feedback_nat > 0:
                        temp_grid[y, x] = np.random.choice((1, 2))
                    elif feedback_nat > 0 and feedback_nat > feedback_inv:
                        temp_grid[y, x] = 1
                    elif feedback_inv > 0 and feedback_inv > feedback_nat:
                        temp_grid[y, x] = 2
                    else:
                        temp_grid[y, x] = 0

                # Native cell
                elif self.grid[y, x] == 1:
                    # Cell dies
                    if feedback_nat < 0:
                        temp_grid[y, x] = 0
                    else:
                        temp_grid[y, x] = 1

                # Invasive cell
                elif self.grid[y, x] == 2:
                    # Cell dies
                    if feedback_inv < 0:
                        temp_grid[y, x] = 0
                    else:
                        temp_grid[y, x] = 2

        self.grid = temp_grid

    # def total_alive(self):
    #    """Counts total number of alive cells in the grid."""
    #    alive_nat = 0
    #    alive_inv = 0

    #    alive_nat += (self.grid == 1).sum().item()
    #    alive_inv += (self.grid == 2).sum().item()

    #    return alive_nat, alive_inv
