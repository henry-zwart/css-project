import numpy as np


class Vegetation:
    grid: np.ndarray

    def __init__(self, width: int = 128):
        self.grid = np.zeros((width, width), dtype=int)
        self.width = width

    def initial_grid(self, p):
        random_matrix = np.random.random(self.grid.shape)
        self.grid = np.where(random_matrix < p, 1, 0)

    def find_close_neighbors(self, x, y):
        indexes = []
        for delta_y in range(-1, 2):
            if y + delta_y < 0 or y + delta_y > self.width - 1:
                continue
            for delta_x in range(-1, 2):
                if x + delta_x < 0 or x + delta_x > self.width - 1:
                    continue
                if delta_x == 0 and delta_y == 0:
                    continue
                indexes.append([x + delta_x, y + delta_y])
        return indexes
    
    def find_far_neighbors(self, x, y):
        indexes = []
        for delta_y in range(-5, 6):
            if y + delta_y < 0 or y + delta_y > self.width - 1:
                continue
            for delta_x in range(-5, 6):
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
                feedback = 2 * close_sum - 3 * far_sum

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
