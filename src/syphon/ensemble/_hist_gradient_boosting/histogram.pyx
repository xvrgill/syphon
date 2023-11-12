"""
Module that contains logic for building histograms from binned feature values.
"""
import cython

from .common import HISTOGRAM_DTYPE

from .common cimport hist_struct
from .common cimport UINT8_DTYPE_C
from .common cimport FLOAT64_DTYPE_C

@cython.final
cdef class HistogramBuilder:
    """Class used to build histograms.

    Each feature has its own histogram where the x-axis represents a bin and
    the y-axis represents the sum of gradients within a given bin.
    """

    cdef public:
        const UINT8_DTYPE_C [::1, :] X_binned
        unsigned int n_features
        unsigned int n_bins
        FLOAT64_DTYPE_C [::1] gradients
        FLOAT64_DTYPE_C [::1] ordered_gradients

    def __init__(self,
                const UINT8_DTYPE_C [::1, :] X_binned,
                unsigned int n_bins,
                FLOAT64_DTYPE_C [::1] gradients):
        self.X_binned = X_binned
        self.n_features = X_binned.shape[1]
        self.n_bins = n_bins
        self.gradients = gradients
        self.ordered_gradients = gradients.copy()

    def compute_histograms_brute_force(
            HistogramBuilder self,
            const unsigned int [::1] sample_indices,
            # const unsigned int [:] allowed_features=None
    ):
        """Compute histograms of the node by using all the data."""
        ...

    @cython.exceptval(check=False)
    cdef void _compute_histogram_brute_force_single_feature(
            HistogramBuilder self,
            const int feature_idx,
            const unsigned int [::1] sample_indices,
            hist_struct [:, ::1] histograms
    ) nogil:
        ...

    def compute_histograms_subtraction(
            HistogramBuilder self,
            hist_struct [:, ::1] parent_histograms,
            hist_struct [:, ::1] sibling_histograms,
            const unsigned int [:] allowed_features=None
    ):
        ...

    @staticmethod
    @cython.exceptval(check=False)
    cpdef void _subtract_histograms(
            const int feature_idx,
            unsigned int n_bins,
            hist_struct [:, ::1] hist_a,
            hist_struct [:, ::1] hist_b,
            hist_struct [:, ::1] out
    ) nogil:
        ...