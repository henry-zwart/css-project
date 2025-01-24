import bz2
import gzip
import lzma
import pickle
import zlib
from enum import StrEnum

import numpy as np


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
