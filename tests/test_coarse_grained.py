import numpy as np

from css_project.kernel import neighbour_count_kernel
from css_project.vegetation import Vegetation, count_neighbours


def test_count_neighbors_small():
    # model = Vegetation(width=5, large_radius=3)
    grid = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
        ]
    )

    expected_count = np.array(
        [
            [1, 2, 0, 1, 1],
            [1, 2, 0, 1, 0],
            [1, 1, 0, 2, 2],
            [0, 1, 1, 2, 0],
            [0, 1, 0, 2, 1],
        ]
    )

    kern = neighbour_count_kernel(radius=1)
    actual_count = count_neighbours(grid, kern)

    assert (expected_count == actual_count).all()


def test_count_neighbors_large():
    # model = Vegetation(width=5, large_radius=3)
    grid = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
        ]
    )

    expected_count = np.array(
        [
            [1, 4, 4, 4, 2],
            [2, 5, 5, 5, 2],
            [3, 5, 5, 5, 3],
            [3, 5, 5, 5, 2],
            [2, 4, 3, 4, 3],
        ]
    )

    kern = neighbour_count_kernel(radius=3)
    actual_count = count_neighbours(grid, kern)

    assert (expected_count == actual_count).all()


def test_feedback():
    model = Vegetation(width=5, large_radius=3)
    nearby_count = np.array(
        [
            [1, 2, 0, 1, 1],
            [1, 2, 0, 1, 0],
            [1, 1, 0, 2, 2],
            [0, 1, 1, 2, 0],
            [0, 1, 0, 2, 1],
        ]
    )
    far_count = np.array(
        [
            [1, 4, 4, 4, 2],
            [2, 5, 5, 5, 2],
            [3, 5, 5, 5, 3],
            [3, 5, 5, 5, 2],
            [2, 4, 3, 4, 3],
        ]
    )
    expected_feedback = np.array(
        [
            [1, 1, -1, 1, 1],
            [1, 1, -1, 1, -1],
            [1, 1, -1, 1, 1],
            [-1, 1, 1, 1, -1],
            [-1, 1, -1, 1, 1],
        ]
    )

    actual_feedback = model.compute_feedback(nearby_count, far_count)

    assert (expected_feedback == actual_feedback).sum()


def test_update():
    model = Vegetation(width=5, large_radius=3)
    model.grid = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
        ]
    )
    expected_new_grid = np.array(
        [
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 1],
        ]
    )

    model.update()

    assert (model.grid == expected_new_grid).all()
