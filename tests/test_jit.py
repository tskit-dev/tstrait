"""
Unit tests for the numba compiled kernels in ``tstrait.jit``.

The kernels are exercised through ``unjit``, which returns the untranslated
Python function. Numba compiles with bounds checking disabled, so an out of
bounds access in the compiled code fails silently; running the Python gives us
numpy's bounds checking and lets coverage measure the kernels. The examples
here are small enough to check by hand.
"""

import numpy as np
import pytest
import tskit

from tstrait import jit


def unjit(func):
    """
    Return the pure Python implementation of a numba jitted function.
    """
    return getattr(func, "py_func", func)


compute_nodes_genetic_value = unjit(jit._compute_nodes_genetic_value)
accumulate_individual_values = unjit(jit._accumulate_individual_values)

# A balanced binary tree on 4 leaves, as returned by
# tskit.Tree.generate_balanced(4). Node 7 is the virtual root, whose only
# child is the root of the tree.
#
#         6
#       /   \
#      4     5
#     / \   / \
#    0   1 2   3
#
NUM_NODES = 7
VIRTUAL_ROOT = 7
LEFT_CHILD = np.array([-1, -1, -1, -1, 0, 2, 4, 6], dtype=np.int32)
RIGHT_SIB = np.array([1, -1, 3, -1, 5, -1, -1, -1], dtype=np.int32)


def mutation_array(nodes):
    """
    Return the ``has_mutation`` array in which the specified nodes are marked.
    """
    has_mutation = np.zeros(NUM_NODES + 1, dtype=bool)
    has_mutation[nodes] = True
    return has_mutation


def compute(causal_nodes, mutation_nodes=(), effect_size=1):
    return compute_nodes_genetic_value(
        left_child_array=LEFT_CHILD,
        right_sib_array=RIGHT_SIB,
        causal_nodes=np.array(causal_nodes, dtype=np.int32),
        has_mutation=mutation_array(list(mutation_nodes)),
        effect_size=effect_size,
    )


def test_tree_arrays_match_tskit():
    """
    Guard against the hand written arrays above drifting from tskit.
    """
    tree = tskit.Tree.generate_balanced(4)
    assert tree.tree_sequence.num_nodes == NUM_NODES
    assert tree.virtual_root == VIRTUAL_ROOT
    np.testing.assert_array_equal(tree.left_child_array, LEFT_CHILD)
    np.testing.assert_array_equal(tree.right_sib_array, RIGHT_SIB)


class TestComputeNodesGeneticValue:
    def test_no_causal_nodes(self):
        np.testing.assert_array_equal(compute([]), np.zeros(NUM_NODES))

    def test_leaf(self):
        # Node 0 has no children, so the sibling walk is never entered.
        np.testing.assert_array_equal(compute([0]), [1, 0, 0, 0, 0, 0, 0])

    def test_internal_node(self):
        np.testing.assert_array_equal(compute([4]), [1, 1, 0, 0, 1, 0, 0])

    def test_root(self):
        np.testing.assert_array_equal(compute([6]), np.ones(NUM_NODES))

    def test_virtual_root(self):
        # The ancestral state is the causal allele, so the virtual root is
        # causal. Its value is computed but is not part of the output.
        value = compute([VIRTUAL_ROOT])
        assert len(value) == NUM_NODES
        np.testing.assert_array_equal(value, np.ones(NUM_NODES))

    def test_mutation_blocks_subtree(self):
        # Node 4 carries a mutation, so it and its descendants keep the value
        # of whatever allele that mutation introduced, ie. zero here.
        np.testing.assert_array_equal(compute([6], [4]), [0, 0, 1, 1, 0, 1, 1])

    def test_multiple_causal_nodes(self):
        np.testing.assert_array_equal(compute([1, 5], [1, 5]), [0, 1, 1, 1, 0, 1, 0])

    def test_effect_size(self):
        np.testing.assert_array_equal(
            compute([4], effect_size=-2.5), [-2.5, -2.5, 0, 0, -2.5, 0, 0]
        )

    @pytest.mark.parametrize("causal_nodes", [[], [0], [4], [6], [VIRTUAL_ROOT], [1, 5]])
    def test_output_excludes_virtual_root(self, causal_nodes):
        assert len(compute(causal_nodes)) == NUM_NODES


class TestAccumulateIndividualValues:
    def test_diploid(self):
        # The node to individual map of tests.data.binary_tree, in which node
        # 6 belongs to no individual.
        nodes_individual = np.array([1, 1, 2, 2, 0, 0, tskit.NULL], dtype=np.int32)
        nodes_genetic_value = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
        value = accumulate_individual_values(nodes_genetic_value, nodes_individual, 3)
        np.testing.assert_array_equal(value, [11, 3, 7])

    def test_repeated_nodes(self):
        # Triploids: three nodes contribute to each individual.
        nodes_individual = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        nodes_genetic_value = np.array([1, 2, 4, 8, 16, 32], dtype=float)
        value = accumulate_individual_values(nodes_genetic_value, nodes_individual, 2)
        np.testing.assert_array_equal(value, [21, 42])

    def test_no_individuals(self):
        nodes_individual = np.full(NUM_NODES, tskit.NULL, dtype=np.int32)
        value = accumulate_individual_values(np.zeros(NUM_NODES), nodes_individual, 0)
        np.testing.assert_array_equal(value, [])


def test_jitted_matches_python():
    """
    The tests above all bypass the compiler, so check that the kernels still
    compile and agree with the Python they were written as.
    """
    causal_nodes = np.array([VIRTUAL_ROOT], dtype=np.int32)
    has_mutation = mutation_array([4])
    nodes_individual = np.array([1, 1, 2, 2, 0, 0, tskit.NULL], dtype=np.int32)

    args = (LEFT_CHILD, RIGHT_SIB, causal_nodes, has_mutation, 0.5)
    nodes_genetic_value = jit._compute_nodes_genetic_value(*args)
    np.testing.assert_array_equal(
        nodes_genetic_value, compute_nodes_genetic_value(*args)
    )
    np.testing.assert_array_equal(nodes_genetic_value, [0, 0, 0.5, 0.5, 0, 0.5, 0.5])

    individual_value = jit._accumulate_individual_values(
        nodes_genetic_value, nodes_individual, 3
    )
    np.testing.assert_array_equal(
        individual_value,
        accumulate_individual_values(nodes_genetic_value, nodes_individual, 3),
    )
    np.testing.assert_array_equal(individual_value, [0.5, 0, 1])
