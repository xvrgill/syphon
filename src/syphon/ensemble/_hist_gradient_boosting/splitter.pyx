"""
Module containing splitting logic
"""
cimport cython
from cython.parallel import prange
import numpy as np
from libc.math cimport INFINITY
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

from .common cimport FLOAT64_DTYPE_C
from .common cimport UINT8_DTYPE_C

from .histogram import hist_struct

cdef struct split_info_struct:
    # C implementation of `SplitInfo` python class.
    # This can be used in nogil sections w/ arrays
    FLOAT64_DTYPE_C gain
    int feature_idx
    size_t bin_idx
    FLOAT64_DTYPE_C sum_gradient_left
    FLOAT64_DTYPE_C sum_gradient_right
    FLOAT64_DTYPE_C sum_hessian_left
    FLOAT64_DTYPE_C sum_hessian_right
    unsigned int n_samples_left
    unsigned int n_samples_right
    FLOAT64_DTYPE_C value_left
    FLOAT64_DTYPE_C value_right


class SplitInfo:
    """Data class to store information on a given split"""


    def __int__(self,
                gain,
                feature_idx,
                bin_idx,
                sum_gradient_left,
                sum_gradient_right,
                sum_hessian_left,
                sum_hessian_right,
                n_samples_left,
                n_samples_right,
                value_left,
                value_right):
        self.gain = gain
        self.feature_idx = feature_idx
        self.bin_idx = bin_idx
        self.sum_gradient_left = sum_gradient_left
        self.sum_gradient_right = sum_gradient_right
        self.sum_hessian_left = sum_hessian_left
        self.sum_hessian_right = sum_hessian_right
        self.n_samples_left = n_samples_left
        self.n_samples_right = n_samples_right
        self.value_left = value_left
        self.value_right = value_right

@cython.final
cdef class Splitter:
    """Utility used to find the best possible split at each node"""
    cdef public:
        const UINT8_DTYPE_C [::1, :] X_binned
        unsigned int n_features
        const unsigned int [::1] n_bins
        unsigned int min_samples_leaf
        FLOAT64_DTYPE_C min_gain_to_split
        FLOAT64_DTYPE_C l2_regularization

        unsigned int [::1] partition
        unsigned int [::1] left_indices_buffer
        unsigned int [::1] right_indices_buffer

    def __init__(self,
                 const UINT8_DTYPE_C [::1, :] X_binned,
                 const unsigned int [::1] n_bins,
                 unsigned int min_samples_leaf=20,
                 FLOAT64_DTYPE_C min_gain_to_split=0.0,
                 FLOAT64_DTYPE_C l2_regularization=0.0):
        self.X_binned = X_binned
        self.n_features = X_binned.shape[1]
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_to_split = min_gain_to_split
        self.l2_regularization = l2_regularization

        # Create array to serve as partition - contains all instances at root
        self.partition = np.arange(X_binned.shape[0], dtype=np.uint32)
        # Create buffers used to split nodes in parallel
        self.left_indices_buffer = np.empty_like(self.partition)
        self.right_indices_buffer = np.empty_like(self.partition)


    def split_indices(Splitter self,
                      # Remove type - class in python but struct in C
                      split_info,
                      unsigned int [::1] sample_indices):
        cdef:
            int n_samples = sample_indices.shape[0]
            UINT8_DTYPE_C bin_idx = split_info.bin_idx
            int feature_idx = split_info.feature_idx
            const UINT8_DTYPE_C [::1] X_binned = self.X_binned[:, feature_idx]
            unsigned int [::1] left_indices_buffer = self.left_indices_buffer
            unsigned int [::1] right_indices_buffer = self.right_indices_buffer
            # TODO: Use the class attribute for n_threads instead of hardcode
            int n_threads = 4

            # We can choose a number of threads to use here
            # Initialize the buffers...

            # Initialize vew to keep track of the size of the buffer in each thread
            int [:] sizes = np.full(n_threads, n_samples // n_threads, dtype=np.int32)
            # Initialize view to keep track of offset
            # Distance from the start of the parent view
            int [:] offset_in_buffers = np.zeros(n_threads, dtype=np.int32)
            # Number of items in the left and right buffers
            # Sum of left and right at a given index is the total number of elements allowed
            int [:] left_counts = np.zeros(n_threads, dtype=np.int32)
            int [:] right_counts = np.zeros(n_threads, dtype=np.int32)
            int left_count
            int right_count
            int start
            int stop
            int i
            int thread_idx
            int sample_idx
            # Start index of the right child in original partition
            # Don't need to track left. Left is start idx of partition.
            int right_child_position
            # Should a given index go to the left
            unsigned char turn_left
            # Offsets of each buffer
            # Distance of each buffer from the start of the base position
            # Tells us where to actually add values from thread to respective buffer index
            int [:] left_offset = np.zeros(n_threads, dtype=np.int32)
            int [:] right_offset = np.zeros(n_threads, dtype=np.int32)

        with nogil:

            # Fix buffer sizes when n_samples can't be neatly divided by threads
            # First n threads will have new sizes to account for remainder
            for thread_idx in range(n_samples % n_threads):
                sizes[thread_idx] += 1

            # Set buffer offsets
            # Use start position and size of previous buffer
            # Left and right buffer offsets are never different
            for thread_idx in range(1, n_threads):
                offset_in_buffers[thread_idx] = \
                    offset_in_buffers[thread_idx - 1] + sizes[thread_idx - 1]

            # Map indices from sample_indices to their respective buffers
            # Perform this in parallel
            for thread_idx in prange(n_threads, schedule="static",
                                     chunksize=1, num_threads=n_threads):
                # Initialize counts in left and right buffers
                left_count = 0
                right_count = 0

                # Start at first offset index
                # Will always be zero at index 0, but not afterward
                start = offset_in_buffers[thread_idx]
                # Stop at the end of the buffer
                # This is where the buffer starts plus the number of (valid) elements in that buffer
                end = start + sizes[thread_idx]

                # Loop through each element in the current buffer
                for i in range(start, stop):
                    # Get the index of the training sample
                    sample_idx = sample_indices[i]

                    # Decide if the current sample should go to left buffer
                    # If not, it will go to right
                    turn_left = ...

                    if turn_left:
                        # Add training sample index to the left buffer
                        left_indices_buffer[start + left_count] = sample_idx
                        # Advance count on left
                        # Tells us where to put the next element in this section
                        left_count = left_count + 1
                    else:
                        # Add training sample index to the right buffer
                        right_indices_buffer[start + right_count] = sample_idx
                        # Advance count on right
                        right_count = right_count + 1

                # Update left and right counts for current thread
                left_counts[thread_idx] = left_count
                right_counts[thread_idx] = right_count

                # Set right child position to be after all the left indices
                right_child_position = 0
                for thread_idx in range(n_threads):
                    right_child_position += left_counts[thread_idx]

                # Update offsets for left and right
                # This determines where each buffer will start to write
                right_offset[0] = right_child_position
                for thread_idx in range(1, n_threads):
                    left_offset[thread_idx] = \
                        left_offset[thread_idx - 1] + left_counts[thread_idx - 1]
                    right_offset[thread_idx] = \
                        right_offset[thread_idx - 1] + right_counts[thread_idx - 1]

                # Apply changes to the left and right buffers to sample indices
                # This occurs in original array also. Sample indices is a view
                for thread_idx in prange(n_threads, schedule="static",
                                         chunksize=1, num_threads=n_threads):
                    # Copy memory directly with mem copy
                    memcpy(
                        # Destination
                        &sample_indices[left_offset[thread_idx]],
                        # Source
                        &left_indices_buffer[offset_in_buffers[thread_idx]],
                        # Number of bytes to copy
                        sizeof(unsigned int) * left_counts[thread_idx]
                    )

                    # Need to handle right differently
                    # Right most node could have an indexing issue
                    # Simpy check to ensure that there are elements in the right buffer
                    # See note below
                    if right_counts[thread_idx] > 0:
                        # Credit: Sklearn & LightGBM
                        # If we're splitting the rightmost node of the tree, i.e. the
                        # rightmost node in the partition array, and if n_threads >= 2, one
                        # might have right_counts[-1] = 0 and right_offset[-1] = len(sample_indices)
                        # leading to evaluating
                        #
                        #    &sample_indices[right_offset[-1]] = &samples_indices[n_samples_at_node]
                        #                                      = &partition[n_samples_in_tree]
                        #
                        # which is an out-of-bounds read access that can cause a segmentation fault.
                        # When boundscheck=True, removing this check produces this exception:
                        #
                        #    IndexError: Out of bounds on buffer access
                        memcpy(
                            &sample_indices[right_offset[thread_idx]],
                            &right_indices_buffer[offset_in_buffers[thread_idx]],
                            sizeof(unsigned int) * right_counts[thread_idx]
                        )

                # Return info for left and right children
                return (
                    # Indices of samples send to left child
                    sample_indices[:right_child_position],
                    # Indices of samples sent to right child,
                    sample_indices[right_child_position:],
                    # Where right children start in the sample indices arr
                    right_child_position
                )





    def find_best_split(Splitter self,
                        unsigned int n_samples,
                        hist_struct [:, ::1] histograms,
                        const FLOAT64_DTYPE_C sum_gradients,
                        const FLOAT64_DTYPE_C sum_hessians,
                        const FLOAT64_DTYPE_C value,
                        const FLOAT64_DTYPE_C lower_bound = -INFINITY,
                        const FLOAT64_DTYPE_C upper_bound = INFINITY):
        """Find the best split point for a given feature using binned data.

        Returns the best split out of all the used features.

        Parameters
        ----------
        n_samples : int
            Number of samples within the node.
        histograms : ndarray of HISTOGRAM_DTYPE of shape (n_features, max_bins)
            Histograms of the binned features at the node.
        sum_gradients : float
            Sum of gradients for each sample
        value: float
            The predicted value of the current node [ F(x) ]
        lower_bound : float
            Lower bound for the value of children nodes. Used to enforce
            monotonic constraints
        upper_bound : float
            Upper bound for the value of children nodes. Used to enforce
            monotonic constraints

        Returns
        -------
        best_split_info : SplitInfo
            Information on the best possible split within the node.
        """
        cdef:
            # Don't need this if we're using all features
            # int feature_idx
            int split_info_idx # <-- Index of split info object within an array
            int best_split_info_idx # <-- Index of split info object in array that is the best choice
            int n_allowed_features
            split_info_struct split_info
            split_info_struct * split_infos # <-- pointer to array of split info objects

        # This can be changed later to only include certain features in split
        # Will always use all features as is
        n_allowed_features = self.n_features

        # Perform operations without gil
        with nogil:

            # Allocate memory for array containing split info objects
            split_infos = <split_info_struct *> malloc(
                sizeof(split_info_struct) * n_allowed_features
            )

            # Create parallel loop to test splits on each feature
            for split_info_idx in prange(
                    n_allowed_features, schedule="static", nogil=True
            ):
                # Set feature index for current split info object
                # Tells us which feature is being split
                split_infos[split_info_idx].feature_idx = split_info_idx

                # Find best bin to split on
                # Gain is -1 by default. No better split was found
                # This node will be turned into a leaf if gain is -1
                split_infos[split_info_idx].gain = -1

                # TODO: Handle missing values here...
                self._find_best_bin_to_split(
                    split_info_idx,
                    histograms,
                    n_samples,
                    sum_gradients,
                    sum_hessians,
                    value,
                    lower_bound,
                    upper_bound,
                    &split_infos[split_info_idx]
                )

                # Finally, choose which feature to split on
                # Previous method found best split point for each feature
                # This method chooses one of those split points / features as final
                best_split_info_idx = self._find_best_feature_to_split_helper(
                    split_infos, n_allowed_features
                )

                # Fetch split info object corresponding to best split
                split_info = split_infos[best_split_info_idx]

            # Create object to store our final best split information
            # This copies C struct data over to python object
            out = SplitInfo(
                split_info.gain,
                split_info.feature_idx,
                split_info.bin_idx,
                split_info.sum_gradient_left,
                split_info.sum_gradient_right,
                split_info.sum_hessian_left,
                split_info.sum_hessian_right,
                split_info.n_samples_left,
                split_info.n_samples_right,
                split_info.value_left,
                split_info.value_right
            )

            # Free memory allocated to split info array
            free(split_infos)

            # Return Python object with best split information
            return out

    @cython.exceptval(check=False)
    cdef int _find_best_feature_to_split_helper(Splitter self,
                                                split_info_struct * split_infos,
                                                int n_allowed_features) nogil:
        """Find the best split by returning index of split info object.
        
        Index is the position of the split info object that represents the 
        best split point from within the array of split info objects.

        Returns
        -------
        split_info_idx: int
            Index of best split info object
        """
        cdef:
            int split_info_idx
            int best_split_info_idx = 0

        # Range starts at one because of comparison
        # Index of zero is hard coded, don't need in loop
        # Best split idx is updated within loop
        for split_info_idx in range(1, n_allowed_features):
            if split_infos[split_info_idx].gain > split_infos[best_split_info_idx].gain:
                # Update best split
                best_split_info_idx = split_info_idx

        return best_split_info_idx

    @cython.exceptval(check=False)
    cdef void _find_best_bin_to_split(Splitter self,
                                     unsigned int feature_idx,
                                     const hist_struct [:, ::1] histograms,
                                     unsigned int n_samples,
                                     FLOAT64_DTYPE_C sum_gradients,
                                     FLOAT64_DTYPE_C sum_hessians,
                                     FLOAT64_DTYPE_C value,
                                     FLOAT64_DTYPE_C lower_bound,
                                     FLOAT64_DTYPE_C upper_bound,
                                     split_info_struct * split_info):
        """Find the best bin to split on for a given feature.
        
        Parameters
        ----------
        feature_idx
        histograms
        n_samples
        sum_gradients
        value
        split_info

        Returns
        -------

        """
        cdef:
            unsigned int bin_idx
            unsigned int n_samples_left
            unsigned int n_samples_right
            unsigned int n_samples_total = n_samples
            unsigned int end = self.n_bins[feature_idx] - 1 # <-- Unsure if we need this unless there are missing vals
            FLOAT64_DTYPE_C sum_gradients_left
            FLOAT64_DTYPE_C sum_gradients_right
            FLOAT64_DTYPE_C sum_hessians_left
            FLOAT64_DTYPE_C sum_hessians_right
            FLOAT64_DTYPE_C loss_current_node
            FLOAT64_DTYPE_C gain
            unsigned char found_better_split = False

            # Best split info variables
            FLOAT64_DTYPE_C best_sum_hessian_left
            FLOAT64_DTYPE_C best_sum_gradient_left
            unsigned int best_bin_idx
            unsigned int best_n_samples_left
            FLOAT64_DTYPE_C best_gain = -1

        sum_gradients_left, sum_gradients_right = 0.0, 0.0
        n_samples_left = 0

        # Compute loss at current node from its value
        # This uses half ordinary least squares for basic version
        loss_current_node = _loss_from_value(value, sum_gradients)

        # Loop through bins to find best split point
        for bin_idx in range(end):
            # Save the number of samples in left child
            n_samples_left += histograms[feature_idx, bin_idx].count
            # Save the number of samples in right child
            n_samples_right += n_samples_total - n_samples_left

            # Hessians are always constant for now, but can change later
            # Constant hessian has value of 1 so we can just use the count here
            sum_hessians_left += histograms[feature_idx, bin_idx].count
            # Right hessians are just the difference
            # Sum hessian node - sum hessian left
            sum_hessians_right += sum_hessians - sum_hessians_left

            # Compute gradients of left and right children
            sum_gradients_left += histograms[feature_idx, bin_idx].sum_gradients
            sum_gradients_right += sum_gradients - sum_gradients_right

            # Check hyperparameter conditions are met

            # Min samples leaf
            if n_samples_left < self.min_samples_leaf:
                # We can't split this any further... move onto next bin
                # Tries to split again until this condition is false
                continue
            if n_samples_right < self.min_samples_leaf:
                # The split is invalid and there is no possibility of better split
                break

            # Min hessian to split
            # Don't need this with current loss
            ...

            # Compute gain
            gain = _split_gain(
                sum_gradients_left, sum_hessians_left,
                sum_gradients_right, sum_hessians_right,
                loss_current_node, lower_bound, upper_bound,
                self.l2_regularization
            )

            # Update best split based on gain
            if gain > best_gain and gain > self.min_gain_to_split:
                found_better_split = True
                best_gain = gain
                best_bin_idx = bin_idx
                best_sum_gradient_left = sum_gradients_left
                best_sum_hessian_left = sum_hessians_left
                best_n_samples_left = n_samples_left

        # Perform operations if a better split is found...
        if found_better_split:
            split_info.gain = best_gain
            split_info.bin_idx = best_bin_idx
            split_info.sum_gradient_left = best_sum_gradient_left
            split_info.sum_gradient_right = sum_gradients - best_sum_gradient_left
            split_info.sum_hessian_left = best_sum_hessian_left
            split_info.sum_hessian_right = sum_hessians - sum_hessians_left
            split_info.n_samples_left = best_n_samples_left
            split_info.n_samples_right = n_samples - best_n_samples_left

            # Recompute best values
            split_info.value_left = compute_node_value(
                split_info.sum_gradient_left,
                split_info.sum_hessian_left,
                lower_bound,
                upper_bound,
                self.l2_regularization
            )
            split_info.value_right = compute_node_value(
                split_info.sum_gradient_right,
                split_info.sum_hessian_right,
                lower_bound,
                upper_bound,
                self.l2_regularization
            )

@cython.exceptval(check=False)
cpdef inline FLOAT64_DTYPE_C compute_node_value(FLOAT64_DTYPE_C sum_gradient,
                                                FLOAT64_DTYPE_C sum_hessian,
                                                FLOAT64_DTYPE_C lower_bound,
                                                FLOAT64_DTYPE_C upper_bound,
                                                FLOAT64_DTYPE_C l2_regularization) nogil:
    """Compute the value of a node.
    
    A node's value is essentially a weight term.
    
    Notes
    -----
    
    The equation for computing a node's value is as follows:
    
    .. math::
    
        v =  - \sum \, \mathrm{G} / \left( \mathrm{H} + \lambda + 1e-15 \\right)
        
    Where :math:`\mathrm{G}` is the gradient and :math:`\mathrm{H}` is the hessian.
    
    Parameters
    ----------
    sum_gradient
    sum_hessian
    lower_bound
    upper_bound
    l2_regularization

    Returns
    -------

    """
    cdef:
        FLOAT64_DTYPE_C value

    # Not using regularization yet. Defaults to zero
    value = -sum_gradient / (sum_hessian + l2_regularization + 1e-15)
    # value = -sum_gradient / (sum_hessian + 1e-15)

    # Cap the value based on bounds to respect monotonic constraints
    # Don't need this yet
    # if value < lower_bound:
    #     value = lower_bound
    # elif value > upper_bound:
    #     value = upper_bound

    return value

@cython.exceptval(check=False)
cdef inline FLOAT64_DTYPE_C _loss_from_value(FLOAT64_DTYPE_C value,
                                             FLOAT64_DTYPE_C sum_gradient) nogil:
    """Calculate the loss of a node from its bounded value.
    
    Parameters
    ----------
    value
    sum_gradient

    Returns
    -------
    loss : float
        Loss of current node
    """
    return sum_gradient * value

@cython.exceptval(check=False)
cdef inline FLOAT64_DTYPE_C _split_gain(FLOAT64_DTYPE_C sum_gradient_left,
                                        FLOAT64_DTYPE_C sum_hessian_left,
                                        FLOAT64_DTYPE_C sum_gradient_right,
                                        FLOAT64_DTYPE_C sum_hessian_right,
                                        FLOAT64_DTYPE_C loss_current_node,
                                        FLOAT64_DTYPE_C lower_bound,
                                        FLOAT64_DTYPE_C upper_bound,
                                        FLOAT64_DTYPE_C l2_regularization) nogil:
    """Compute reduction in loss after split.
    
    Compare the reduction in loss after a split is performed to the loss if 
    the node were to simply become a leaf.
    
    Parameters
    ----------
    sum_gradient_left
    sum_hessian_left
    sum_gradient_right
    sum_hessian_right
    loss_current_node
    lower_bound
    upper_bound
    l2_regularization

    Returns
    -------
    gain : float
        Information gain value
    """
    cdef:
        FLOAT64_DTYPE_C gain
        FLOAT64_DTYPE_C value_left
        FLOAT64_DTYPE_C value_right

    # Compute values of potential left and right children
    value_left = compute_node_value(
        sum_gradient_left,
        sum_hessian_left,
        lower_bound,
        upper_bound,
        l2_regularization
    )
    value_right = compute_node_value(
        sum_gradient_right,
        sum_hessian_right,
        lower_bound,
        upper_bound,
        l2_regularization
    )

    # Check monotonic constraints here
    # Don't need this yet
    ...

    # Compute information gain
    # Start with base gain as the loss of the parent node
    # Subtract losses of child nodes to get final potential gain
    # If potential gain is greater than existing gain, it's a better split
    # Gain starts at -1
    gain = loss_current_node
    gain -= _loss_from_value(value_left, sum_gradient_left)
    gain -= _loss_from_value(value_right, sum_gradient_right)

    return gain


