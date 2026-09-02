"""
Numba compiled kernels.

Two conventions apply throughout this module:

1. Node indexed arrays follow the tskit quintuply linked tree encoding, in
   which the arrays have ``num_nodes + 1`` entries and the last entry
   corresponds to the virtual root.

2. Numba compiles with bounds checking disabled by default, so an out of
   bounds access here fails silently rather than raising ``IndexError``.
   Keeping the compiled functions in one module lets us unit test them
   through their ``py_func`` attribute in ``tests/test_jit.py``, which runs
   the untranslated Python and therefore gets numpy's bounds checking and
   coverage measurement for free.
"""

import numba
import numpy as np
import tskit


@numba.njit
def _compute_nodes_genetic_value(
    left_child_array,
    right_sib_array,
    causal_nodes,
    has_mutation,
    effect_size,
):
    """
    Compute the genetic value of each node by assigning ``effect_size`` to
    every node in ``causal_nodes`` and to each of their descendants that is
    not separated from them by a node carrying a mutation.

    The ``left_child_array``, ``right_sib_array`` and ``has_mutation`` inputs
    are indexed by node ID and have ``num_nodes + 1`` entries. The
    ``causal_nodes`` are distinct node IDs, and may include the virtual root,
    which happens when the ancestral state is the causal allele. The returned
    array has one entry per node and does not include the virtual root.
    """
    num_nodes = len(left_child_array) - 1
    # One extra entry, so that the virtual root can be a causal node.
    genetic_value = np.zeros(num_nodes + 1)
    # Each node is pushed at most once, since causal nodes other than the
    # virtual root all carry a mutation and so are never pushed as a child.
    stack = np.empty(num_nodes + 1, dtype=np.int32)
    stack_top = len(causal_nodes)
    stack[:stack_top] = causal_nodes
    while stack_top > 0:
        stack_top -= 1
        parent_node_id = stack[stack_top]
        genetic_value[parent_node_id] = effect_size
        child_node_id = left_child_array[parent_node_id]
        while child_node_id != tskit.NULL:
            if not has_mutation[child_node_id]:
                stack[stack_top] = child_node_id
                stack_top += 1
            child_node_id = right_sib_array[child_node_id]
    return genetic_value[:num_nodes]


@numba.njit
def _accumulate_individual_values(
    nodes_genetic_value, nodes_individual, num_individuals
):
    """
    Accumulate the individual genetic values by summing their node
    contributions.
    """
    individuals_genetic_value = np.zeros(num_individuals)
    for u in range(len(nodes_individual)):
        ind = nodes_individual[u]
        if ind != tskit.NULL:
            individuals_genetic_value[ind] += nodes_genetic_value[u]
    return individuals_genetic_value
