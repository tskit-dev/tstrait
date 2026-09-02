"""
Unit tests for the numba compiled kernels in ``tstrait.jit``.

Each test runs twice, against the compiled kernel and against the Python it was
written as, through the ``jit`` and ``nojit`` parameters of the fixtures below.
Numba compiles with bounds checking disabled, so an out of bounds access in the
compiled code fails silently; running the Python gives us numpy's bounds
checking and lets coverage measure the kernels.

The examples are small trees from the tskit generators, drawn in the class
docstrings so that the expected values can be checked against the topology.
"""

import numpy as np
import pytest
import tskit

from tstrait import jit

from .data import (
    binary_tree,
    diff_ind_tree,
    triploid_tree,
)  # noreorder


def kernel(func, param):
    """
    Return either the compiled kernel or the Python it was written as.
    """
    return func if param == "jit" else func.py_func


@pytest.fixture(params=["jit", "nojit"])
def node_genetic_value(request):
    """
    Return a function computing the node genetic values for a tskit tree.

    The ``causal_nodes`` are the nodes at which the causal allele appears,
    which includes the virtual root when the ancestral state is causal, and the
    ``mutation_nodes`` are the nodes at which a mutation occurs.
    """
    func = kernel(jit._compute_nodes_genetic_value, request.param)

    def f(tree, causal_nodes, mutation_nodes=(), effect_size=1):
        has_mutation = np.zeros(len(tree.left_child_array), dtype=bool)
        has_mutation[list(mutation_nodes)] = True
        return func(
            left_child_array=tree.left_child_array,
            right_sib_array=tree.right_sib_array,
            causal_nodes=np.array(causal_nodes, dtype=np.int32),
            has_mutation=has_mutation,
            effect_size=effect_size,
        )

    return f


@pytest.fixture(params=["jit", "nojit"])
def individual_genetic_value(request):
    """
    Return a function accumulating node genetic values over the individuals of
    a tree sequence.
    """
    func = kernel(jit._accumulate_individual_values, request.param)

    def f(ts, nodes_genetic_value):
        return func(
            np.asarray(nodes_genetic_value, dtype=float),
            ts.nodes_individual,
            ts.num_individuals,
        )

    return f


def isolated_samples_tree(n):
    """
    Return a tree of n isolated sample nodes, which are therefore all roots.
    """
    tables = tskit.TableCollection(sequence_length=1)
    for _ in range(n):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    return tables.tree_sequence().first()


def multiroot_tree():
    """
    Return generate_balanced(4) with the edges into the root removed, so that
    nodes 4 and 5 are both roots and node 6 is isolated.
    """
    tables = tskit.Tree.generate_balanced(4).tree_sequence.dump_tables()
    tables.edges.replace_with(tables.edges[tables.edges.parent != 6])
    return tables.tree_sequence().first()


def empty_tree():
    """
    Return the tree of a tree sequence that has no nodes.
    """
    return tskit.TableCollection(sequence_length=1).tree_sequence().first()


class TestBalancedBinaryTree:
    """
    tskit.Tree.generate_balanced(4), in which node 7 is the virtual root::

           6
         +-+-+
         4   5
        +++ +++
        0 1 2 3
    """

    @pytest.fixture
    def tree(self):
        return tskit.Tree.generate_balanced(4)

    def test_no_causal_nodes(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, []), [0, 0, 0, 0, 0, 0, 0]
        )

    def test_leaf(self, tree, node_genetic_value):
        # Node 0 has no children, so the sibling walk is never entered.
        np.testing.assert_array_equal(
            node_genetic_value(tree, [0]), [1, 0, 0, 0, 0, 0, 0]
        )

    def test_internal_node(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [4]), [1, 1, 0, 0, 1, 0, 0]
        )

    def test_root(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [6]), [1, 1, 1, 1, 1, 1, 1]
        )

    def test_virtual_root(self, tree, node_genetic_value):
        # The ancestral state is the causal allele, so the virtual root is
        # causal. Its own value is computed but is not part of the output.
        value = node_genetic_value(tree, [7])
        assert len(value) == 7
        np.testing.assert_array_equal(value, [1, 1, 1, 1, 1, 1, 1])

    def test_mutation_on_internal_node(self, tree, node_genetic_value):
        # Node 4 and its children carry whatever allele the mutation on node 4
        # introduced, so the traversal does not descend into them.
        np.testing.assert_array_equal(
            node_genetic_value(tree, [6], [4]), [0, 0, 1, 1, 0, 1, 1]
        )

    def test_mutation_on_leaf(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [6], [0]), [0, 1, 1, 1, 1, 1, 1]
        )

    def test_two_causal_nodes(self, tree, node_genetic_value):
        # Both mutations are back to the causal allele, so both nodes are
        # causal and each starts its own traversal.
        np.testing.assert_array_equal(
            node_genetic_value(tree, [1, 5], [1, 5]), [0, 1, 1, 1, 0, 1, 0]
        )

    def test_effect_size(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [4], effect_size=-2.5),
            [-2.5, -2.5, 0, 0, -2.5, 0, 0],
        )


class TestStarTree:
    """
    tskit.Tree.generate_star(5), in which node 6 is the virtual root. The
    polytomy at the root makes the sibling walk iterate five times::

            5
        +-+-+-+-+
        0 1 2 3 4
    """

    @pytest.fixture
    def tree(self):
        return tskit.Tree.generate_star(5)

    def test_root(self, tree, node_genetic_value):
        np.testing.assert_array_equal(node_genetic_value(tree, [5]), [1, 1, 1, 1, 1, 1])

    def test_virtual_root(self, tree, node_genetic_value):
        np.testing.assert_array_equal(node_genetic_value(tree, [6]), [1, 1, 1, 1, 1, 1])

    def test_mutation_on_root(self, tree, node_genetic_value):
        # The root is the virtual root's only child, so a mutation there stops
        # the traversal immediately.
        np.testing.assert_array_equal(
            node_genetic_value(tree, [6], [5]), [0, 0, 0, 0, 0, 0]
        )

    def test_mutations_along_sibling_chain(self, tree, node_genetic_value):
        # Mutations at both ends of the chain of children and in the middle.
        np.testing.assert_array_equal(
            node_genetic_value(tree, [5], [0, 2, 4]), [0, 1, 0, 1, 0, 1]
        )


class TestNonBinaryTree:
    """
    tskit.Tree.generate_balanced(6, arity=3), in which node 10 is the virtual
    root. The polytomy is below the root::

             9
         +---+---+
         6   7   8
        +++ +++ +++
        0 1 2 3 4 5
    """

    @pytest.fixture
    def tree(self):
        return tskit.Tree.generate_balanced(6, arity=3)

    def test_virtual_root(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [10]), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )

    def test_middle_subtree(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [7]), [0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
        )

    def test_mutation_on_middle_subtree(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [10], [7]), [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
        )

    def test_mutations_on_outer_subtrees(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [10], [6, 8]), [0, 0, 1, 1, 0, 0, 0, 1, 0, 1]
        )


class TestCombTree:
    """
    tskit.Tree.generate_comb(5), in which node 9 is the virtual root. A ladder
    is the deepest traversal for a given number of leaves::

          8
        +-+-+
        |   7
        | +-+-+
        | |   6
        | | +-++
        | | |  5
        | | | +++
        0 1 2 3 4
    """

    @pytest.fixture
    def tree(self):
        return tskit.Tree.generate_comb(5)

    def test_root(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [8]), [1, 1, 1, 1, 1, 1, 1, 1, 1]
        )

    def test_virtual_root(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [9]), [1, 1, 1, 1, 1, 1, 1, 1, 1]
        )

    def test_second_rung(self, tree, node_genetic_value):
        # Everything below node 7, which is all of the tree except leaf 0 and
        # the root.
        np.testing.assert_array_equal(
            node_genetic_value(tree, [7]), [0, 1, 1, 1, 1, 1, 1, 1, 0]
        )

    def test_mutation_near_root(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [8], [7]), [1, 0, 0, 0, 0, 0, 0, 0, 1]
        )

    def test_mutation_near_leaves(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [8], [5]), [1, 1, 1, 0, 0, 0, 1, 1, 1]
        )


class TestDegenerateTrees:
    """
    Trees that do not have a single root above a set of internal nodes.
    """

    def test_single_node(self, node_genetic_value):
        # generate_balanced(1) is the single node 0, with virtual root 1.
        tree = tskit.Tree.generate_balanced(1)
        np.testing.assert_array_equal(node_genetic_value(tree, [1]), [1])
        np.testing.assert_array_equal(node_genetic_value(tree, [0]), [1])

    def test_no_nodes(self, node_genetic_value):
        # The virtual root is node 0 and there is nothing to return.
        tree = empty_tree()
        assert tree.virtual_root == 0
        np.testing.assert_array_equal(node_genetic_value(tree, [0]), [])

    def test_isolated_samples(self, node_genetic_value):
        # Nodes 0, 1 and 2 are all roots, so the virtual root has three
        # children:
        #
        #     0 1 2
        #
        tree = isolated_samples_tree(3)
        np.testing.assert_array_equal(node_genetic_value(tree, [3]), [1, 1, 1])

    def test_isolated_samples_with_mutation(self, node_genetic_value):
        tree = isolated_samples_tree(3)
        np.testing.assert_array_equal(node_genetic_value(tree, [3], [1]), [1, 0, 1])

    def test_multiple_roots(self, node_genetic_value):
        # Nodes 4 and 5 are both roots and node 6 is isolated, so it is not
        # reached from the virtual root:
        #
        #      4   5
        #     +++ +++
        #     0 1 2 3
        #
        tree = multiroot_tree()
        assert tree.roots == [4, 5]
        np.testing.assert_array_equal(
            node_genetic_value(tree, [7]), [1, 1, 1, 1, 1, 1, 0]
        )

    def test_multiple_roots_one_blocked(self, node_genetic_value):
        tree = multiroot_tree()
        np.testing.assert_array_equal(
            node_genetic_value(tree, [7], [4]), [0, 0, 1, 1, 0, 1, 0]
        )


class TestAccumulateIndividualValues:
    """
    Sum the node genetic values over the nodes of each individual, using the
    tree sequences in tests/data.py. The node values are powers of two, so the
    individual values identify the nodes that contributed to them.
    """

    def test_binary_tree(self, individual_genetic_value):
        # Individual 0 is nodes 4 and 5, individual 1 is nodes 0 and 1, and
        # individual 2 is nodes 2 and 3. Node 6 has no individual.
        ts = binary_tree()
        np.testing.assert_array_equal(ts.nodes_individual, [1, 1, 2, 2, 0, 0, -1])
        np.testing.assert_array_equal(
            individual_genetic_value(ts, [1, 2, 4, 8, 16, 32, 64]), [48, 3, 12]
        )

    def test_diff_ind_tree(self, individual_genetic_value):
        # The same tree, with the leaves paired up the other way around.
        ts = diff_ind_tree()
        np.testing.assert_array_equal(ts.nodes_individual, [1, 2, 1, 2, 0, 0, -1])
        np.testing.assert_array_equal(
            individual_genetic_value(ts, [1, 2, 4, 8, 16, 32, 64]), [48, 5, 10]
        )

    def test_triploid_tree(self, individual_genetic_value):
        # Two triploids: nodes 0, 2 and 4, and nodes 1, 3 and 5.
        ts = triploid_tree()
        np.testing.assert_array_equal(ts.nodes_individual, [0, 1, 0, 1, 0, 1, -1, -1])
        np.testing.assert_array_equal(
            individual_genetic_value(ts, [1, 2, 4, 8, 16, 32, 64, 128]), [21, 42]
        )

    def test_no_individuals(self, individual_genetic_value):
        ts = tskit.Tree.generate_balanced(4).tree_sequence
        assert ts.num_individuals == 0
        np.testing.assert_array_equal(
            individual_genetic_value(ts, [1, 2, 4, 8, 16, 32, 64]), []
        )


class TestNodeAndIndividualValues:
    """
    The two kernels composed, as they are used in tstrait.genetic_value, so
    that the individual values can be traced back to the tree topology. The
    tree of tests.data.binary_tree is::

           6
         +-+-+
         4   5
        +++ +++
        0 1 2 3

    with individual 0 being nodes 4 and 5, individual 1 nodes 0 and 1, and
    individual 2 nodes 2 and 3.
    """

    def test_internal_node(self, node_genetic_value, individual_genetic_value):
        ts = binary_tree()
        value = node_genetic_value(ts.first(), [4])
        np.testing.assert_array_equal(value, [1, 1, 0, 0, 1, 0, 0])
        # Individual 0 has one copy through node 4, individual 1 has two
        # copies through nodes 0 and 1, and individual 2 has none.
        np.testing.assert_array_equal(individual_genetic_value(ts, value), [1, 2, 0])

    def test_virtual_root(self, node_genetic_value, individual_genetic_value):
        # The causal allele is the ancestral state, so every node carries it
        # and every diploid has two copies.
        ts = binary_tree()
        value = node_genetic_value(ts.first(), [7])
        np.testing.assert_array_equal(value, [1, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(individual_genetic_value(ts, value), [2, 2, 2])

    def test_triploid(self, node_genetic_value, individual_genetic_value):
        # tests.data.triploid_tree, in which node 6 is the parent of nodes 3,
        # 4 and 5 and node 7 is the root:
        #
        #     7
        #     +-+-+---+
        #     | | |   6
        #     | | | +-+-+
        #     0 1 2 3 4 5
        #
        ts = triploid_tree()
        value = node_genetic_value(ts.first(), [6])
        np.testing.assert_array_equal(value, [0, 0, 0, 1, 1, 1, 1, 0])
        # Individual 0 is nodes 0, 2 and 4, and individual 1 is nodes 1, 3
        # and 5.
        np.testing.assert_array_equal(individual_genetic_value(ts, value), [1, 2])
