"""
Module that contains logic for building histograms from binned feature values.
"""
import cython
# noinspection PyUnresolvedReferences
from cython.parallel import prange
import numpy as np

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
        FLOAT64_DTYPE_C [::1] hessians
        FLOAT64_DTYPE_C [::1] ordered_gradients
        FLOAT64_DTYPE_C [::1] ordered_hessians
        unsigned char hessians_are_constant
        int n_threads

    def __init__(self,
                const UINT8_DTYPE_C [::1, :] X_binned,
                unsigned int n_bins,
                FLOAT64_DTYPE_C [::1] gradients,
                FLOAT64_DTYPE_C [::1] hessians,
                unsigned char hessians_are_constant,
                int n_threads):
        self.X_binned = X_binned
        self.n_features = X_binned.shape[1]
        self.n_bins = n_bins
        self.gradients = gradients
        self.hessians = hessians
        # Copy gradients/hessians so that they aren't mutated
        self.ordered_gradients = gradients.copy()
        self.ordered_hessians = hessians.copy()
        self.hessians_are_constant = hessians_are_constant
        self.n_threads = n_threads

    def compute_histograms_brute_force(
            HistogramBuilder self,
            const unsigned int [::1] sample_indices,
            # const unsigned int [:] allowed_features=None
    ):
        """Compute histograms of the node by using all the data."""
        cdef:
            int n_samples
            int feature_idx
            # int f_idx
            int i
            # Can avoid python interactions by creating local views of vars
            unsigned char hessians_are_constant = self.hessians_are_constant
            int n_allowed_features = self.n_features
            FLOAT64_DTYPE_C [::1] ordered_gradients = self.ordered_gradients
            FLOAT64_DTYPE_C [::1] gradients = self.gradients
            FLOAT64_DTYPE_C [::1] ordered_hessians = self.ordered_hessians
            FLOAT64_DTYPE_C [::1] hessians = self.hessians
            # Histograms will be initialized to zero later within a prange
            hist_struct[:, ::1] histograms = np.empty(
                shape=(self.n_features, self.n_bins),
                dtype=HISTOGRAM_DTYPE
            )
            int n_threads = self.n_threads

        with nogil:
            n_samples = sample_indices.shape[0]

            # Populate ordered gradients/hessians. Arranges items to be in the
            # same order as sample indices and moves all items to the beginning
            # of the gradient/hessian array. Need to make sure that only the
            # first n elements of each of these is used moving forward. Not
            # doing so would mean that items for all insances would be used.
            # Note that this is the desired behavior for root nodes.

            # If this is not a root node...
            if sample_indices.shape[0] != gradients.shape[0]:
                # and hessians are constant...
                if hessians_are_constant:
                    # re-order gradients only
                    for i in prange(n_samples, schedule="static",
                                    num_threads=n_threads):
                        ordered_gradients[i] = self.gradients[sample_indices[i]]
                else:
                    # re-order gradients and hessians
                    for i in prange(n_samples, schedule="static",
                                    num_threads=n_threads):
                        ordered_gradients[i] = self.gradients[sample_indices[i]]
                        ordered_hessians[i] = self.gradients[sample_indices[i]]

            # Compute histogram for each feature
            for feature_idx in prange(n_allowed_features, schedule="static",
                                      num_threads=n_threads):
                self._compute_histogram_brute_force_single_feature(
                    feature_idx, sample_indices, histograms
                )

        return histograms

    cdef void _compute_histogram_brute_force_single_feature(
            HistogramBuilder self,
            const int feature_idx,
            const unsigned int [::1] sample_indices,
            hist_struct [:, ::1] histograms
    ) noexcept nogil:
        """Calculate histogram for a single feature.
        
        Parameters
        ----------
        feature_idx
        sample_indices
        histograms
        """

        cdef:
            unsigned int n_samples = sample_indices.shape[0]
            const UINT8_DTYPE_C [::1] X_binned = self.X_binned[:, feature_idx]
            # True if samples in node is the same as total samples
            unsigned int root_node = X_binned.shape[0] == n_samples
            # Gradients in the same order as sample indices in `ordered_gradients`
            # so we need to select a subset from ordered gradients.This vector is
            # created by moving sample gradients to beginning of the original
            # gradients array.
            FLOAT64_DTYPE_C [::1] ordered_gradients = self.ordered_gradients[:n_samples]
            FLOAT64_DTYPE_C[::1] ordered_hessians = self.ordered_hessians[:n_samples]
            unsigned char hessians_are_constant = self.hessians_are_constant
            unsigned int bin_idx = 0

        # For each bin...
        for bin_idx in range(self.n_bins):
            # Initialize gradients, hessians, and count
            histograms[feature_idx, bin_idx].sum_gradients = 0
            histograms[feature_idx, bin_idx].sum_hessians = 0
            histograms[feature_idx, bin_idx].count = 0

        # If this node is the root node...
        if root_node:
            # and hessians are constant...
            if hessians_are_constant:
                # build root histograms with no hessians
                _build_histogram_root_no_hessian(
                    feature_idx,
                    X_binned,
                    ordered_gradients,
                    histograms
                )
            else:
                # build root histograms with hessians
                _build_histogram_root(
                    feature_idx,
                    X_binned,
                    ordered_gradients,
                    ordered_hessians,
                    histograms
                )
        # If this node is not the root node...
        else:
            # and hessians are constant...
            if hessians_are_constant:
                # build histograms normally but without hessians
                _build_histogram_no_hessian(
                    feature_idx,
                    sample_indices,
                    X_binned,
                    ordered_gradients,
                    histograms
                )
            else:
                # build histograms normally
                _build_histogram(
                    feature_idx,
                    sample_indices,
                    X_binned,
                    ordered_gradients,
                    ordered_hessians,
                    histograms
                )



    def compute_histograms_subtraction(
            HistogramBuilder self,
            hist_struct [:, ::1] parent_histograms,
            hist_struct [:, ::1] sibling_histograms,
            const unsigned int [:] allowed_features=None
    ):
        """Compute the histograms of the node using the subtraction trick.

        hist(parent) = hist(left_child) + hist(right_child)

        Parameters
        ----------
        parent_histograms
            Histograms of a parent node
        sibling_histograms
            Histograms of a sibling node (i.e. one of the parent's children)
        allowed_features
            Features that are eligible for split based on constraints

        Returns
        -------
        histograms : ndarray of HISTOGRAM_DTYPE, shape(n_features, n_bins)
            The computed histograms of the current node.
        """
        cdef:
            int feature_idx
            int f_idx
            int n_allowed_features = self.n_features
            hist_struct [:, ::1] histograms = np.empty(
                shape=(self.n_features, self.n_bins),
                dtype=HISTOGRAM_DTYPE
            )
            bint has_interaction_cst = allowed_features is not None
            int n_threads = self.n_threads

        if has_interaction_cst:
            n_allowed_features = allowed_features.shape[0]

        # Compute histogram of each feature
        for f_idx in prange(n_allowed_features, schedule='static', nogil=True,
                            num_threads=n_threads):
            if has_interaction_cst:
                feature_idx = allowed_features[f_idx]
            else:
                feature_idx = f_idx

            _subtract_histograms(
                feature_idx,
                self.n_bins,
                parent_histograms,
                sibling_histograms,
                histograms,
            )
        return histograms


cpdef void _subtract_histograms(
        const int feature_idx,
        unsigned int n_bins,
        hist_struct [:, ::1] hist_a,
        hist_struct [:, ::1] hist_b,
        hist_struct [:, ::1] out) noexcept nogil:
    """Compute the difference between two histograms.

    Used to subtract a child histogram from its parent. The result is the
    histogram of the other child.

    Parameters
    ----------
    feature_idx
    n_bins
    hist_a
    hist_b
    out
    """
    cdef unsigned int i = 0
    # For each bin...
    for i in range(n_bins):
        # Subtract sum of gradients
        out[feature_idx, i].sum_gradients = (
            hist_a[feature_idx, i].sum_gradients -
            hist_b[feature_idx, i].sum_gradients
        )
        # Subtract sum of hessians
        out[feature_idx, i].sum_hessians = (
            hist_a[feature_idx, i].sum_hessians -
            hist_b[feature_idx, i].sum_hessians
        )
        # Subtract count
        out[feature_idx, i].count = (
            hist_a[feature_idx, i].count -
            hist_b[feature_idx, i].count
        )

cpdef void _build_histogram_root(
        const int feature_idx,
        const UINT8_DTYPE_C [::1] binned_feature,
        const FLOAT64_DTYPE_C [::1] all_gradients,
        const FLOAT64_DTYPE_C [::1] all_hessians,
        hist_struct [:, ::1] out) noexcept nogil:
    """Build histogram for a feature at root node.

    Use all instances to calculate this.

    Parameters
    ----------
    feature_idx
    binned_feature
    all_gradients
    all_hessians
    out
    """
    cdef:
        int i = 0
        unsigned int n_samples = binned_feature.shape[0]
        unsigned int unrolled_upper = (n_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

    # Use unrolled loop to tell compiler to compute in semi-parallel
    for i in range(0, unrolled_upper, 4):
        # Retrieve the bin that a sample falls into
        bin_0 = binned_feature[i]
        bin_1 = binned_feature[i + 1]
        bin_2 = binned_feature[i + 2]
        bin_3 = binned_feature[i + 3]

        # Calculate sum of gradients for that bin
        # Ordered gradients is vector of gradients in sample indices
        # This does not contain all gradients
        out[feature_idx, bin_0].sum_gradients += all_gradients[i]
        out[feature_idx, bin_1].sum_gradients += all_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += all_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += all_gradients[i + 3]

        # Calculate sum of hessians for that bin
        out[feature_idx, bin_0].sum_hessians += all_hessians[i]
        out[feature_idx, bin_1].sum_hessians += all_hessians[i + 1]
        out[feature_idx, bin_2].sum_hessians += all_hessians[i + 2]
        out[feature_idx, bin_3].sum_hessians += all_hessians[i + 3]

        # Update count of instances for that bin
        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    # Compute the same metrics for the remaining samples
    # Some won't be included in the above loop due to floor division
    for i in range(unrolled_upper, n_samples):
        bin_idx = binned_feature[i]
        out[feature_idx, bin_idx].sum_gradients += all_gradients[i]
        out[feature_idx, bin_idx].sum_hessians += all_hessians[i]
        out[feature_idx, bin_idx].count += 1

cpdef void _build_histogram_root_no_hessian(
        const int feature_idx,
        const UINT8_DTYPE_C [::1] binned_feature,
        const FLOAT64_DTYPE_C [::1] all_gradients,
        hist_struct [:, ::1] out) noexcept nogil:
    """Build histogram from a feature without hessians.

    Parameters
    ----------
    feature_idx
    binned_feature
    all_gradients
    out
    """
    cdef:
        unsigned int i = 0
        unsigned int n_samples = binned_feature.shape[0]
        unsigned int unrolled_upper = (n_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[i]
        bin_1 = binned_feature[i + 1]
        bin_2 = binned_feature[i + 2]
        bin_3 = binned_feature[i + 3]

        out[feature_idx, bin_0].sum_gradients += all_gradients[i]
        out[feature_idx, bin_1].sum_gradients += all_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += all_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += all_gradients[i + 3]

        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    for i in range(unrolled_upper, n_samples):
        bin_idx = binned_feature[i]
        out[feature_idx, bin_idx].sum_gradients += all_gradients[i]
        out[feature_idx, bin_idx].count += 1

cpdef void _build_histogram(
        const int feature_idx,
        const unsigned int [::1] sample_indices,
        const UINT8_DTYPE_C [::1] binned_feature,
        const FLOAT64_DTYPE_C [::1] ordered_gradients,
        const FLOAT64_DTYPE_C [::1] ordered_hessians,
        hist_struct [:, ::1] out) noexcept nogil:
    """Create a histogram for a feature"""
    cdef:
        int i = 0
        unsigned int n_node_samples = sample_indices.shape[0]
        unsigned int unrolled_upper = (n_node_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

    # Use unrolled loop to tell compiler to compute in semi-parallel
    for i in range(0, unrolled_upper, 4):
        # Retrieve the bin that a sample falls into
        bin_0 = binned_feature[sample_indices[i]]
        bin_1 = binned_feature[sample_indices[i]]
        bin_2 = binned_feature[sample_indices[i]]
        bin_3 = binned_feature[sample_indices[i]]

        # Calculate sum of gradients for that bin
        # Ordered gradients is vector of gradients in sample indices
        # This does not contain all gradients
        out[feature_idx, bin_0].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_1].sum_gradients += ordered_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += ordered_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += ordered_gradients[i + 3]

        # Calculate sum of hessians for that bin
        out[feature_idx, bin_0].sum_hessians += ordered_hessians[i]
        out[feature_idx, bin_1].sum_hessians += ordered_hessians[i + 1]
        out[feature_idx, bin_2].sum_hessians += ordered_hessians[i + 2]
        out[feature_idx, bin_3].sum_hessians += ordered_hessians[i + 3]

        # Update count of instances for that bin
        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    # Compute the same metrics for the remaining samples
    # Some won't be included in the above loop due to floor division
    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[i]
        out[feature_idx, bin_idx].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_idx].sum_hessians += ordered_hessians[i]
        out[feature_idx, bin_idx].count += 1

cpdef void _build_histogram_no_hessian(
        const int feature_idx,
        const unsigned int [::1] sample_indices,
        const UINT8_DTYPE_C [::1] binned_feature,
        const FLOAT64_DTYPE_C [::1] ordered_gradients,
        hist_struct [:, ::1] out) noexcept nogil:
    """Build histogram for feature without using hessians.

    Parameters
    ----------
    feature_idx
    sample_indices
    binned_feature
    all_gradients
    all_hessians
    out

    Returns
    -------

    """
    cdef:
        int i = 0
        unsigned int n_node_samples = sample_indices.shape[0]
        unsigned int unrolled_upper = (n_node_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

        # Use unrolled loop to tell compiler to compute in semi-parallel
    for i in range(0, unrolled_upper, 4):
        # Retrieve the bin that a sample falls into
        bin_0 = binned_feature[sample_indices[i]]
        bin_1 = binned_feature[sample_indices[i]]
        bin_2 = binned_feature[sample_indices[i]]
        bin_3 = binned_feature[sample_indices[i]]

        # Calculate sum of gradients for that bin
        # Ordered gradients is vector of gradients in sample indices
        # This does not contain all gradients
        out[feature_idx, bin_0].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_1].sum_gradients += ordered_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += ordered_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += ordered_gradients[i + 3]

        # Update count of instances for that bin
        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

        # Compute the same metrics for the remaining samples
        # Some won't be included in the above loop due to floor division
    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[i]
        out[feature_idx, bin_idx].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_idx].count += 1
