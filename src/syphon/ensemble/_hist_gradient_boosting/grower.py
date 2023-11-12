"""
Module that contains logic used to build trees.

This is not the public api! See ``gradient_boosting.py`` for that.
"""
from heapq import heappop, heappush
from timeit import default_timer as time
from typing import Union, Self

from .utils import sum_parallel

from .histogram import HistogramBuilder
from .splitter import SplitInfo, Splitter


class TreeNode:
    """Structure that represents a node in a decision tree"""

    split_info: Union[SplitInfo, None] = None
    left_child = None
    right_child = None
    histograms = None

    partition_start = 0
    partition_stop = 0

    def __init__(self, depth, sample_indices, sum_gradients, sum_hessians, value=None):
        self.depth = depth
        self.sample_indices = sample_indices
        self.n_samples = sample_indices.shape[0]
        self.sum_gradients = sum_gradients
        self.sum_hessians = sum_hessians
        self.value = value
        self.is_leaf = False
        self.set_bounds_of_children(float("-inf"), float("+inf"))

    def set_bounds_of_children(self, lower, upper):
        """Set bounds of child nodes to respect monotonic constrains."""
        self.children_lower_bound = lower
        self.children_upper_bound = upper

    def __lt__(self, other: Self):
        """Comparison for priority queue implementation of a heap."""
        return self.split_info.gain > other.split_info.gain


class TreeGrower:
    """Class responsible for growing the tree"""

    def __init__(self,
                 X_binned,
                 gradients,
                 max_leaf_nodes=None,
                 max_depth=None,
                 min_samples_leaf=20,
                 min_gain_to_split=0.0,
                 n_bins=256,
                 shrinkage=1.0,
                 n_threads=None):
        # Set basic class attributes
        self.X_binned = X_binned
        self.n_features = X_binned.shape[0]
        self.gradients = gradients
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_to_split = min_gain_to_split
        self.n_bins = n_bins
        self.shrinkage = shrinkage
        self.n_threads = n_threads

        # Set tree related attributes
        self.splittable_nodes = []
        self.finalized_leaves = []
        self.time_spent_finding_splits = 0.0
        self.time_spent_computing_histograms = 0.0
        self.time_spent_splitting_nodes = 0.0

        # Initialize root node of the estimator
        self._initialize_root(gradients)

        # Initialize the number of nodes... one because root is initialized
        self.n_nodes = 1

        # Initialize histogram builder to construct histograms
        self.histogram_builder = HistogramBuilder(
            X_binned,
            n_bins,
            gradients
        )

        # Initialize a new splitter for each tree
        # Gets initial values and will be updated at each split
        self.splitter = Splitter(
            X_binned,
            n_bins,
            min_samples_leaf,
            min_gain_to_split
        )

    def _initialize_root(self, gradients):
        """Initialize root node of tree."""
        # Compute number of samples in training data
        n_samples = self.X_binned.shape[0]

        # Set current depth to zero
        depth = 0

        # Compute the sum of gradients at the root node
        sum_gradients = sum_parallel(gradients)

        # Hessians will always be constant for now
        sum_hessians = n_samples

        # Create root node
        self.root = TreeNode(
            depth=depth,
            sample_indices=self.splitter.partition,
            sum_gradients=sum_gradients,
            sum_hessians=sum_hessians,
            value=0
        )

        # Initialize start and stop indices of partition
        self.root.partition_start = 0
        self.root.partition_stop = n_samples

        # If number of samples is less than min samples leaf * 2...
        # Don't compute split statistics. Just finalize the root as a leef
        if self.root.n_samples < self.min_samples_leaf * 2:
            ...

        # Compute histograms and time it
        start_time = time()
        self.root.histograms = self.histogram_builder.compute_histograms_brute_force(
            self.root.sample_indices
        )
        stop_time = time()
        self.time_spent_computing_histograms += stop_time - start_time

        # Find best split point and time it
        start_time = time()
        self._find_best_split(self.root)
        stop_time = time()
        self.time_spent_finding_splits += stop_time - start_time

    def grow(self):
        """Recursively grow tree while there are splittable nodes."""
        ...

    def _find_best_split(self, node):
        """Find the best split point for a given node.

        Generates `SplitInfo` object for the passed node. Pushes the node to
        the heap if it is valid.
        """
        node.split_info = self.splitter.find_best_split(
            n_samples=node.n_samples,
            histograms=node.histograms,
            sum_gradients=node.sum_gradients,
            sum_hessians=node.sum_hessians,
            value=node.value,
            # lower_bound=node.children_lower_bound,
            # upper_bound=node.children_upper_bound,
        )

        if node.split_info.gain <= 0:
            # Invalid split. Make current node a leaf.
            self._finalize_leaf(node)
        else:
            heappush(self.splittable_nodes, node)

    def split(self):
        """Perform the split on a given node

        Returns
        -------
        left : TreeNode
            Left child of the current node after split
        right : TreeNode
            right child of the current node after split
        """
        # Pop a node from the heap
        # Node with the highest priority (largest gain) will be retrieved
        node: TreeNode = heappop(self.splittable_nodes)

        # Perform the split by separating indices into left/right children
        # Record time to apply the found best split
        start = time()
        (
            sample_indices_left,
            sample_indices_right,
            right_child_position
        ) = self.splitter.split_indices(node.split_info, node.sample_indices)
        stop = time()
        self.time_spent_splitting_nodes += stop - start

        # These vars are used to calculate whether to split or finalize a leaf

        # Update depth - depth of node plus a level
        depth = node.depth + 1
        # Calculate total number of leaves - finalized and not
        # If a node can be split, it could be a leaf but won't be in the finalized arr
        n_leaf_nodes = len(self.finalized_leaves) + len(self.splittable_nodes)
        # Add two more since we are now splitting the current node
        # Node is no longer in splittable nodes. Has been popped
        n_leaf_nodes += 2

        # Create left and right children node objects
        left_child_node = TreeNode(
            depth,
            sample_indices_left,
            node.split_info.sum_gradient_left,
            node.split_info.sum_hessian_left,
            value=node.split_info.value_right
        )
        right_child_node = TreeNode(
            depth,
            sample_indices_right,
            node.split_info.sum_gradient_right,
            node.split_info.sum_hessian_right,
            value=node.split_info.value_right
        )

        # Set the left and right children to current node
        node.left_child =right_child_node
        node.right_child = right_child_node

        # Set indices indicators on child nodes
        # To be used to identify which instances are in each node
        left_child_node.partition_start = node.partition_start
        left_child_node.partition_stop = node.partition_start + right_child_position
        right_child_node.partition_start = left_child_node.partition_stop
        right_child_node.partition_stop = node.partition_stop

        # TODO: Deal with interactions and/or missing vals here if needed
        ...

        # Update the total node count
        # This update reflects the two new child nodes
        self.n_nodes += 2

        # Check hyperparameters
        # Turn child nodes into leaves if they don't respect params
        # Better than doing it at the current node
        # Don't need to start a new split if we know it shouldn't be split further
        if self.max_leaf_nodes is not None and n_leaf_nodes == self.max_leaf_nodes:
            # Finalize left child node as leaf
            self._finalize_leaf(left_child_node)
            # Finalize right child node as leaf
            self._finalize_leaf(right_child_node)
            # Finalize all remaining nodes as leaves. Can't split anything else
            self._finalize_splittable_nodes()
            # Return left and right children
            # Not sure how doing this helps us, but maybe check public api?
            return left_child_node, right_child_node

        if self.max_depth is not None and depth == self.max_depth:
            self._finalize_leaf(left_child_node)
            self._finalize_leaf(right_child_node)
            self._finalize_splittable_nodes()
            return left_child_node, right_child_node

        # If children have less than 2 * min samples, don't split it
        # This works because splitting a child results in two nodes
        # To meet this requirement, the best case scenario is that samples
        # are distributed equally...
        # If that is true, the parent must have at least 2n samples
        # If either child is less than min_samples_leaf, the split won't be allowed
        # Note: A parent in this context is actually the child of current node.
        #       Therefore, the "child" is actually the child of the child of
        #       the current node.
        # Don't want to return these because they may not be leaves
        # Will check if they're leaves later
        if left_child_node.n_samples < self.min_samples_leaf * 2:
            # Make left child a leaf node
            # At least one child of left child must be less than threshold
            # Can't split if a child won't have the required number of samples
            self._finalize_leaf(left_child_node)
        if right_child_node.n_samples < self.min_samples_leaf * 2:
            # The same goes for the right child
            self._finalize_leaf(right_child_node)

        # TODO: Handle monotonic constraints here w/ bounds if needed
        ...

        # Compute histograms of children
        # Also find the best split for each
        # Only perform this when necessary
        should_split_left = left_child_node.is_leaf
        should_split_right = right_child_node.is_leaf
        if should_split_left or should_split_right:
            # Will always compute histograms of both
            # Optimization method makes this inexpensive to perform
            # Works by subtracting histogram for the larger child
            n_samples_left = left_child_node.sample_indices.shape[0]
            n_samples_right = right_child_node.sample_indices.shape[0]
            # Find the smaller child to perform more expensive op on it
            if n_samples_left < n_samples_right:
                smallest_child = left_child_node
                largest_child = right_child_node
            else:
                smallest_child = right_child_node
                largest_child = right_child_node

        # Create histograms
        # Brute force for smallest child
        # Subtraction for largest child
        # Also time the histogram computation
        start = time()
        # noinspection PyUnboundLocalVariable
        smallest_child.histograms = self.histogram_builder.compute_histograms_brute_force(
            smallest_child.sample_indices
        )
        # TODO: Add subtraction method once it's created
        # noinspection PyUnboundLocalVariable
        largest_child.histograms = ...


    def _finalize_leaf(self, node):
        """Turn node into a leaf node.

        Parameters
        ----------
        node : TreeNode
            Node to be turned into a leaf
        """
        # Set is leaf property to true to indicate that it's a leaf
        node.is_leaf = True
        # Add leaf node to list of leaves
        self.finalized_leaves.append(node)

    def _finalize_splittable_nodes(self):
        """Turn all splittable nodes into leaves.

        Convert all possible splittable nodes
        """
        # While there are still splittable nodes
        while len(self.splittable_nodes) > 0:
            # Pop them from the heap
            node = heappop(self.splittable_nodes)
            # Turn them into finalized leaves
            self._finalize_leaf(node)
