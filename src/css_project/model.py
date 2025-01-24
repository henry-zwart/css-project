from abc import ABC, abstractmethod

import numpy as np


class VegetationModel(ABC):
    grid: np.ndarray
    proportion_alive_list: list[float]

    def __init__(
        self,
        width: int,
        species_prop: float | np.ndarray | list[float],
        init_method: str,
    ):
        self.width = width
        self.area = width * width
        self.proportion_alive_list = []
        self.initial_grid(species_prop, type=init_method)

    @property
    @abstractmethod
    def n_states(self) -> int: ...

    @abstractmethod
    def update(self): ...

    def initial_grid(self, p: float | np.ndarray | list[float], type="random"):
        match p:
            case float(prob):
                probs = np.array([prob])
            case _:
                probs = np.asarray(p)

        zero_prob = 1 - probs.sum()
        if zero_prob < 0.0:
            raise ValueError("Species proportions sum must not exceed 1.0.")

        probs = np.concat(([0, zero_prob], probs))
        print(probs)
        prob_ranges = np.cumsum(probs)

        grid = np.zeros((self.width, self.width), dtype=np.int64)
        samples = np.random.random(grid.shape)
        for i in range(len(prob_ranges) - 1):
            update_cells = (prob_ranges[i] <= samples) & (samples < prob_ranges[i + 1])
            grid[update_cells] = i

        self.grid = grid

    def find_states(self, neighbors) -> list[int]:
        """Count neighbours of each species."""
        counts = [0 for _ in range(self.n_states)]
        for row, column in neighbors:
            for s in range(1, self.n_states):
                counts[s] += self.grid[row, column] == s

        return counts[1:]

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

    def total_alive(self) -> int:
        """Counts total number of alive cells in the grid."""
        alive = (self.grid > 0).sum()
        return alive

    def species_alive(self) -> list[int]:
        x = [(self.grid == i).sum() for i in range(1, self.n_states)]
        return x

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

    def find_steady_state(self, iterations: int):
        for iter in range(iterations):
            if self.is_steady_state():
                print(f"Iteration: {iter}")
                break
            self.update()
