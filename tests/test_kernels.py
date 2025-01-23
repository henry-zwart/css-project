import numpy as np

from css_project.fine_grained import count_neighbours


def test_2d_neighbour_count():
    """Kernel counts the number of neighbours of a cell."""
    state = np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )

    expected_counts = np.array(
        [
            [2, 2, 1],
            [3, 4, 2],
            [2, 1, 1],
        ]
    )

    actual_counts = count_neighbours(state)
    assert (actual_counts == expected_counts).all()


def test_3d_neighbour_count():
    """Kernel counts the number of neighbours of a cell."""
    state = np.array(
        [
            [
                [1, 1, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            [
                [1, 0, 1],
                [1, 0, 1],
                [0, 1, 0],
            ],
        ]
    )

    expected_counts = np.array(
        [
            [
                [2, 2, 1],
                [3, 4, 2],
                [2, 1, 1],
            ],
            [
                [1, 4, 1],
                [2, 5, 2],
                [2, 2, 2],
            ],
        ]
    )

    actual_counts = count_neighbours(state)
    assert (actual_counts == expected_counts).all()
