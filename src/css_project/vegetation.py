import numpy as np

np.random.seed(1)


class Vegetation:
    N_STATES = 2

    grid: np.ndarray

    def __init__(
        self,
        width: int = 128,
        small_radius: int = 1,
        large_radius: int = 4,
        positive_factor: int = 7,
        negative_factor: int = 1,
    ):
        self.grid = np.zeros((width, width), dtype=int)
        self.width = width
        self.small_radius = small_radius
        self.large_radius = large_radius
        self.positive_factor = positive_factor
        self.negative_factor = negative_factor
        self.area = width * width
        self.proportion_alive_list = []

    def initial_grid(self, p):
        random_matrix = np.random.random(self.grid.shape)
        self.grid = np.where(random_matrix < p, 1, 0)

    def find_neighbors(self, x, y, radius):
        indexes = [
            [x + dx, y + dy]
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            if (dx != 0 or dy != 0)
            and 0 <= x + dx < self.width
            and 0 <= y + dy < self.width
        ]
        return indexes

    def find_states(self, neighbors):
        alive = 0

        # Sums up all 'alive' neighbors
        for row, column in neighbors:
            alive += self.grid[row, column]

        return alive

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
                print(f"difference: {der}, second order difference: {der}")
                return True
        return False

    def find_steady_state(self, iterations):
        for iter in range(iterations):
            if self.is_steady_state():
                print(f"Iteration: {iter}")
                break
            self.update()

    def update(self):
        temp_grid = np.empty_like(self.grid)

        for y in range(self.width):
            for x in range(self.width):
                # Find local neighbors and calculate sum
                close = self.find_neighbors(y, x, self.small_radius)
                close_sum = self.find_states(close)

                # Find non-local neighbors and calculate sum
                far = self.find_neighbors(y, x, self.large_radius)
                far_sum = self.find_states(far)

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


class InvasiveVegetation:
    N_STATES = 3

    grid: np.ndarray

    def __init__(
        self,
        width: int = 128,
        small_radius: int = 1,
        large_radius: int = 4,
        pos_factor_nat: int = 8,
        neg_factor_nat: int = 1,
        pos_factor_inv: int = 8,
        neg_factor_inv: int = 1,
    ):
        self.grid = np.zeros((width, width), dtype=int)
        self.width = width
        self.small_radius = small_radius
        self.large_radius = large_radius
        self.pos_factor_nat = pos_factor_nat
        self.neg_factor_nat = neg_factor_nat
        self.pos_factor_inv = pos_factor_inv
        self.neg_factor_inv = neg_factor_inv

    def initial_grid(self, p_nat=0.25, p_inv=0.25):
        """p_nat and p_inv should be percentages for
        which something occurs. Example:
        p_nat = 0.3 so 30% of the initial grid is native.
        The total of p_nat and p_inv cannot larger than 1 (100%).
        """
        if (p_nat + p_inv) > 1:
            raise ValueError("Total of p_nat and p_inv cannot be larger than 1")

        # Set the percentile region for p_inv
        p_inv += p_nat

        # Assume p_nat is the lowest
        random_matrix = np.random.random(self.grid.shape)
        self.grid[np.where(random_matrix <= p_nat)] = 1
        self.grid[np.where((random_matrix > p_nat) & (random_matrix <= p_inv))] = 2
        self.grid[np.where(random_matrix > p_inv)] = 0

    def find_neighbors(self, x, y, radius):
        """Positive = close
        Negative = far"""
        indexes = []

        left = -1 * radius
        right = radius + 1

        for delta_y in range(left, right):
            if y + delta_y < 0 or y + delta_y > self.width - 1:
                continue
            for delta_x in range(left, right):
                if x + delta_x < 0 or x + delta_x > self.width - 1:
                    continue
                if delta_x == 0 and delta_y == 0:
                    continue
                indexes.append([x + delta_x, y + delta_y])

        return indexes

    def find_states(self, neighbors):
        """return number of 1 and 2's"""
        native = 0
        invasive = 0

        # Sums up all 'alive' neighbors
        for row, column in neighbors:
            if self.grid[row, column] == 1:
                native += 1
            elif self.grid[row, column] == 2:
                invasive += 1

        return native, invasive

    def update(self):
        temp_grid = np.empty_like(self.grid)

        for y in range(self.width):
            for x in range(self.width):
                # Find local neighbors and calculate sum
                close = self.find_neighbors(y, x, self.small_radius)
                close_nat, close_inv = self.find_states(close)
                # close_sum = self.find_states(close)

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

    def total_alive(self):
        """Counts total number of alive cells in the grid."""
        alive_nat = 0
        alive_inv = 0

        alive_nat += (self.grid == 1).sum().item()
        alive_inv += (self.grid == 2).sum().item()

        return alive_nat, alive_inv
