import bz2
import gzip
import lzma
import pickle
import zlib
from enum import StrEnum

import numpy as np
from scipy import ndimage

CLUSTER_COUNT_STRUCTURE = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int)


class Compression(StrEnum):
    ZLIB = "zlib"
    GZIP = "gzip"
    BZ2 = "bz2"
    LZMA = "lzma"

    def compress(self, string: bytes):
        match self:
            case Compression.ZLIB:
                return zlib.compress(string)
            case Compression.GZIP:
                return gzip.compress(string)
            case Compression.BZ2:
                return bz2.compress(string)
            case Compression.LZMA:
                return lzma.compress(string)


def compressed_size(arr: np.ndarray, compression: Compression) -> float:
    """Compress a numpy array, and return the compression factor.

    This gives an estimate of the Kolmogorov complexity of the array.
    Supported compression methods:
        - zlib
        - gzip
        - bz2
        - lzma
    """
    serialised_arr = pickle.dumps(arr)
    return len(compression.compress(serialised_arr)) / len(serialised_arr)


def _get_cluster_sizes(arr: np.ndarray) -> np.ndarray:
    """Calculate the cluster sizes in a 2D grid.

    Clustering assumes a Moore's neighbourhood, so includes diagonal neighbours.
    Two neighbouring cells are considered part of a cluster if their values are
    non-zero.

    Args:
        arr: A 2D numpy array of integers representing the states of the grid
            cells.

    Returns:
        Numpy array containing the sizes of identified clusters.
    """
    cluster_matrix, _ = ndimage.label(arr, structure=CLUSTER_COUNT_STRUCTURE)
    return ndimage.sum(arr, cluster_matrix, index=np.arange(cluster_matrix.max() + 1))


def count_clusters(arr: np.ndarray) -> int:
    """Get the number of clusters in a cellular automata grid.

    Assumes a Moore's neighbourhood, so diagonal neighbours are included.

    Args:
        arr: A 2D numpy array of integers representing the states of the grid
            cells.

    Returns:
        Integer number of clusters identified.
    """
    _, n_clusters = ndimage.label(arr, structure=CLUSTER_COUNT_STRUCTURE)
    return n_clusters


def ratio_cluster_size(arr: np.ndarray) -> float:
    """Ratio of largest to smallest cluster size in a cellular automata grid.

    Assumes a Moore's neighbourhood, so diagonal neighbours are included.

    Args:
        arr: A 2D numpy array of integers representing the states of the grid
            cells.

    Returns:
        The ratio of maximum to minimum cluster sizes (maximum / minimum).
        `np.inf` returned if no clusters are found.

    Raises:
        ValueError: If supplied array is not 2D.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, received array of shape: {arr.shape}")

    cluster_sizes = _get_cluster_sizes(arr)[1:]
    try:
        return cluster_sizes.max() / cluster_sizes.min()
    except ValueError:
        return np.inf
    except:
        raise


def maximum_cluster_size(arr: np.ndarray) -> float:
    """Proportional size of the maximum cluster in a cellular automata grid.

    Assumes a Moore's neighbourhood, so diagonal neighbours are included.

    Args:
        arr: A 2D numpy array of integers representing the states of the grid
            cells.

    Returns:
        The maximum cluster size as a proportion of the number of grid cells.

    Raises:
        ValueError: If supplied array is not 2D.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, received array of shape: {arr.shape}")
    cluster_sizes = _get_cluster_sizes(arr)
    return cluster_sizes.max() / (arr.shape[0] * arr.shape[1])
