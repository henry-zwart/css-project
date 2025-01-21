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
        for delta_x in range(-1, 2):
            if x + delta_x < 0 or x + delta_x > self.width - 1:
                continue
            for delta_y in range(-1, 2):
                if y + delta_y < 0 or y + delta_y > self.width - 1:
                    continue
                if delta_x == 0 and delta_y == 0:
                    continue
                indexes.append([x + delta_x, y + delta_y])
        return indexes

    def update(self):
        test = self.find_neighbors(9, 0)
        print(test)
