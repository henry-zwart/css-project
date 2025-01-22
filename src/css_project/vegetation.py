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
        positive_factor: int = 8,
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

    def find_close_neighbors(self, x, y):
        indexes = []
        left = -1 * self.small_radius
        right = self.small_radius + 1
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

    def find_far_neighbors(self, x, y):
        indexes = []
        left = -1 * self.large_radius
        right = self.large_radius + 1
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
        alive = 0

        # Sums up all 'alive' neighbors
        for row, column in neighbors:
            # print(self.grid[x, y])
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
                close = self.find_close_neighbors(y, x)
                close_sum = self.find_states(close)

                # Find non-local neighbors and calculate sum
                far = self.find_far_neighbors(y, x)
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

    def total_alive(self):
        """Counts total number of alive cells in the grid."""
        alive = 0

        for y in range(self.width):
            for x in range(self.width):
                if self.grid[y, x] == 1:
                    alive += 1

        return alive


class InvasiveVegetation:
    N_STATES = 3

    grid: np.ndarray

    def __init__(
        self,
        width: int = 128,
        small_radius: int = 1,
        large_radius: int = 4,
        positive_factor: int = 8,
        negative_factor: int = 1,
    ):
        self.grid = np.zeros((width, width), dtype=int)
        self.width = width
        self.small_radius = small_radius
        self.large_radius = large_radius
        self.positive_factor_native = positive_factor
        self.positive_factor_invasive = positive_factor
        self.negative_factor = negative_factor

    def initial_grid(self, p_nat=0.5, p_inv=0.75):
        """P_none, p_inv and p_nat should be percentile regions for
        which something occurs:
        0.0 - 0.5: 0
        0.5 - 0.75: 1
        0.75 - 1.0: 2
        """

        # Assume p_nat is the lowest
        random_matrix = np.random.random(self.grid.shape)
        self.grid[np.where(random_matrix < p_nat)] = 0
        self.grid[np.where((random_matrix >= p_nat) & (random_matrix < p_inv))] = 1
        self.grid[np.where(random_matrix >= p_inv)] = 2
        print(self.grid)

    def find_close_neighbors(self, x, y):
        indexes = []
        left = -1 * self.small_radius
        right = self.small_radius + 1
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

    def find_far_neighbors(self, x, y):
        indexes = []
        left = -1 * self.large_radius
        right = self.large_radius + 1
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
        alive = 0

        # Sums up all 'alive' neighbors
        for row, column in neighbors:
            # print(self.grid[x, y])
            alive += self.grid[row, column]

        return alive

    def update(self):
        temp_grid = np.empty_like(self.grid)

        for y in range(self.width):
            for x in range(self.width):
                # Find local neighbors and calculate sum
                close = self.find_close_neighbors(y, x)
                close_sum = self.find_states(close)

                # Find non-local neighbors and calculate sum
                far = self.find_far_neighbors(y, x)
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

    def total_alive(self):
        """Counts total number of alive cells in the grid."""
        alive = 0

        for y in range(self.width):
            for x in range(self.width):
                if self.grid[y, x] == 1:
                    alive += 1

        return alive
