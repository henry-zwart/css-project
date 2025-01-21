import numpy as np


class GameOfLife:
    grid: np.ndarray

    def __init__(self, width: int = 128):
        self.grid = np.zeros((width, width), dtype=int)
        self.width = width

    def initial_grid(self, p):
        random_matrix = np.random.random(self.grid.shape)
        self.grid = np.where(random_matrix < p, 1, 0)

    def find_neighbors(self, x, y):
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
                test = self.find_neighbors(y, x)
                alive = self.find_states(test)

                if self.grid[y, x] == 0:
                    # Cell becomes alive
                    if alive == 3:
                        temp_grid[y, x] = 1
                    # Unchanged
                    else:
                        temp_grid[y, x] = 0

                elif self.grid[y, x] == 1:
                    # Cell Death (exposure and crowding)
                    if alive < 2 or alive > 3:
                        temp_grid[y, x] = 0
                    # Unchanged
                    else:
                        temp_grid[y, x] = 1

        self.grid = temp_grid
