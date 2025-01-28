import numpy as np

from css_project.complexity import (
    count_clusters,
    maximum_cluster_size,
    ratio_cluster_size,
)


def test_max_cluster_size_no_clusters():
    grid = np.zeros((4, 4), dtype=np.int64)
    expected_val = 0
    actual_val = maximum_cluster_size(grid)
    assert np.isclose(expected_val, actual_val)


def test_max_cluster_size_single_cluster():
    # 4x4 grid with ones in first 2 positions of top row
    grid = np.zeros((4, 4), dtype=np.int64)
    grid[0, :2] = 1
    expected_val = 2 / 16
    actual_val = maximum_cluster_size(grid)
    assert np.isclose(expected_val, actual_val)

    # 4x4 grid with ones in top left corner (3)
    grid = np.zeros((4, 4), dtype=np.int64)
    grid[0, :2] = 1
    grid[1, 0] = 1
    expected_val = 3 / 16
    actual_val = maximum_cluster_size(grid)
    assert np.isclose(expected_val, actual_val)

    # 4x4 grid with diagonal cluster from top left
    grid = np.zeros((4, 4), dtype=np.int64)
    grid[np.diag_indices(4)] = 1
    expected_val = 4 / 16
    actual_val = maximum_cluster_size(grid)
    assert np.isclose(expected_val, actual_val)

    # 4x4 grid with all ones
    grid = np.ones((4, 4), dtype=np.int64)
    expected_val = 16 / 16
    actual_val = maximum_cluster_size(grid)
    assert np.isclose(expected_val, actual_val)


def test_max_cluster_size_multi_cluster():
    # 4x4 grid with ones in:
    #   - top left corner (3)
    #   - bottom right corner (2)
    grid = np.zeros((4, 4), dtype=np.int64)
    grid[0, :2] = 1
    grid[1, 0] = 1
    grid[3, -2:] = 1
    expected_val = 3 / 16
    actual_val = maximum_cluster_size(grid)
    assert np.isclose(expected_val, actual_val)

    # 4x4 grid with ones in:
    #   - top left corner (3)
    #   - bottom right corner (3)
    grid = np.zeros((4, 4), dtype=np.int64)
    grid[0, :2] = 1
    grid[1, 0] = 1
    grid[3, -2:] = 1
    grid[2, -1] = 1
    expected_val = 3 / 16
    actual_val = maximum_cluster_size(grid)
    assert np.isclose(expected_val, actual_val)


def test_cluster_count():
    # No clusters
    grid = np.zeros((4, 4), dtype=np.int64)
    expected_val = 0
    actual_val = count_clusters(grid)
    assert expected_val == actual_val

    # One cluster (top left)
    grid = np.zeros((4, 4), dtype=np.int64)
    grid[0, :2] = 1
    grid[1, 0] = 1
    expected_val = 1
    actual_val = count_clusters(grid)
    assert expected_val == actual_val

    # Two clusters (top left, bottom right). Same size.
    grid = np.zeros((4, 4), dtype=np.int64)
    grid[0, :2] = 1
    grid[1, 0] = 1
    grid[3, -2:] = 1
    grid[2, -1] = 1
    expected_val = 2
    actual_val = count_clusters(grid)
    assert expected_val == actual_val

    # One cluster, diagonal
    grid = np.zeros((4, 4), dtype=np.int64)
    grid[np.diag_indices(4)] = 1
    expected_val = 1
    actual_val = count_clusters(grid)
    assert expected_val == actual_val


def test_cluster_size_ratio_no_clusters():
    grid = np.zeros((4, 4), dtype=np.int64)
    expected_val = np.inf
    actual_val = ratio_cluster_size(grid)
    assert expected_val == actual_val


def test_cluster_size_ratio_one_cluster():
    grid = np.zeros((4, 4), dtype=np.int64)
    grid[:2, :2] = 1
    expected_val = 1
    actual_val = ratio_cluster_size(grid)
    assert np.isclose(expected_val, actual_val)


def test_cluster_size_ratio_multi_cluster():
    # 2x2 cluster in top left. 1x1 in bottom right.
    grid = np.zeros((4, 4), dtype=np.int64)
    grid[:2, :2] = 1
    grid[-1, -1] = 1
    expected_val = 4
    actual_val = ratio_cluster_size(grid)
    assert np.isclose(expected_val, actual_val)

    # Three clusters of size: 16, 10, 4.
    grid = np.zeros((100, 100), dtype=np.int64)
    grid[:4, :4] = 1
    grid[20:22, :5] = 1
    grid[0, -4:] = 1
    expected_val = 4
    actual_val = ratio_cluster_size(grid)
    assert np.isclose(expected_val, actual_val)


def test_cluster_count_mixed_states():
    grid = np.zeros((4, 4), dtype=np.int64)
    grid[0:2, :2] = 1
    grid[2:4, :2] = 2
    expected_val = 1
    actual_val = count_clusters(grid)
    assert np.isclose(expected_val, actual_val)


def test_cluster_size_mixed_states():
    grid = np.zeros((4, 4), dtype=np.int64)
    grid[0:2, :2] = 1
    grid[2:4, :2] = 2
    expected_val = 0.25
    actual_val = maximum_cluster_size(grid)
    assert np.isclose(expected_val, actual_val)
