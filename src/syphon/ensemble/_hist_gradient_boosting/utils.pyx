"""
Utility module.
"""
from cython.parallel import prange
from .common cimport FLOAT64_DTYPE_C

def sum_parallel(FLOAT64_DTYPE_C [:] arr):
    """Find the summation of values in an array with parallelization.

    Parameters
    ----------
    arr: np.ndarray
        array of floats to sum

    Returns
    -------
    result: np.float64
        Scalar sum of input array
    """
    cdef:
        FLOAT64_DTYPE_C result = 0.0
        int i = 0

    for i in prange(arr.shape[0], schedule="static", nogil=True):
        result += arr[i]

    return result