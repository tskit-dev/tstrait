"""
Unit tests for the numba compiled kernels in ``tstrait.jit``.

Each test runs twice, against the compiled kernel and against the Python it was
written as, through the ``jit`` and ``nojit`` parameters of the fixtures below.
Numba compiles with bounds checking disabled, so an out of bounds access in the
compiled code fails silently; running the Python gives us numpy's bounds
checking and lets coverage measure the kernels.

The examples are small trees from the tskit generators, drawn in the class
docstrings so that the expected values can be checked against the topology.
Each is turned into a tree sequence with a single causal site, since the
descent works from the mutations of the ARG rather than from a tree.
"""

import numpy as np
import pandas as pd
import pytest
import tskit

from tstrait import jit
from tstrait.genetic_value import _GeneticValue

from .data import (
    binary_tree,
    diff_ind_tree,
    triploid_tree,
)  # noreorder

ANCESTRAL_STATE = "A"


def kernel(func, param):
    """
    Return either the compiled kernel or the Python it was written as.
    """
    return func if param == "jit" else func.py_func


def one_site(tree, mutations):
    """
    Return the tree's tree sequence carrying a single site at position 0, with
    the given ``(node, derived_state)`` mutations. Mutations must be listed
    from the oldest node down, so that a mutation follows its parent.
    """
    tables = tree.tree_sequence.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()
    site = tables.sites.add_row(position=0, ancestral_state=ANCESTRAL_STATE)
    for node, derived_state in mutations:
        tables.mutations.add_row(site=site, node=node, derived_state=derived_state)
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


@pytest.fixture(params=["jit", "nojit"])
def node_genetic_value(request):
    """
    Return a function computing the node genetic values of a tree carrying one
    causal site.

    The ``mutations`` are ``(node, derived_state)`` pairs against an ancestral
    state of "A", and a node's value is ``effect_size`` when the allele it
    inherits is ``causal_allele``.
    """
    func = kernel(jit._descend_arg, request.param)

    def f(
        tree,
        mutations=(),
        causal_allele=ANCESTRAL_STATE,
        effect_size=1,
        level="node",
    ):
        ts = tree if isinstance(tree, tskit.TreeSequence) else one_site(tree, mutations)
        trait_df = pd.DataFrame(
            {
                "site_id": [0],
                "effect_size": [effect_size],
                "trait_id": [0],
                "causal_allele": [causal_allele],
            }
        )
        genetic = _GeneticValue(ts, trait_df)
        return func(**genetic._descent_arguments(level, 0))

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


def isolated_samples_ts(n, mutations=()):
    """
    Return a tree sequence of n isolated sample nodes, which are all roots,
    carrying one causal site.
    """
    tables = tskit.TableCollection(sequence_length=1)
    for _ in range(n):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    site = tables.sites.add_row(position=0, ancestral_state=ANCESTRAL_STATE)
    for node, derived_state in mutations:
        tables.mutations.add_row(site=site, node=node, derived_state=derived_state)
    return tables.tree_sequence()


def multiroot_tree():
    """
    Return generate_balanced(4) with the edges into the root removed, so that
    nodes 4 and 5 are both roots and node 6 is isolated.
    """
    tables = tskit.Tree.generate_balanced(4).tree_sequence.dump_tables()
    tables.edges.replace_with(tables.edges[tables.edges.parent != 6])
    return tables.tree_sequence().first()


def empty_ts():
    """
    Return a tree sequence that has no nodes, but does have a causal site.
    """
    tables = tskit.TableCollection(sequence_length=1)
    tables.sites.add_row(position=0, ancestral_state=ANCESTRAL_STATE)
    return tables.tree_sequence()


class TestSearchSorted:
    """
    The binary search the descent uses to find the mutations of an edge that
    lie in a range of causal sites.
    """

    @pytest.fixture(params=["jit", "nojit"])
    def search_sorted(self, request):
        return kernel(jit._search_sorted, request.param)

    @pytest.mark.parametrize(
        ("key", "expected"), [(-1, 0), (0, 0), (1, 1), (3, 1), (4, 1), (5, 3), (9, 4), (10, 5)]
    )
    def test_search(self, search_sorted, key, expected):
        values = np.array([0, 4, 4, 7, 9], dtype=np.int32)
        assert search_sorted(values, 0, len(values), key) == expected

    def test_restricted_range(self, search_sorted):
        # Only the values in [1, 4) are searched, so a key below the range
        # comes back as the start of it.
        values = np.array([0, 4, 4, 7, 9], dtype=np.int32)
        assert search_sorted(values, 1, 4, -1) == 1
        assert search_sorted(values, 1, 4, 8) == 4

    def test_empty_range(self, search_sorted):
        values = np.array([0, 4, 4, 7, 9], dtype=np.int32)
        assert search_sorted(values, 2, 2, 4) == 2


class TestGroupWeight:
    """
    The total weight of the entries of one group that fall in a range of
    causal sites. Group 0 holds sites 1 and 3, group 1 is empty and group 2
    holds sites 0, 2 and 2.
    """

    @pytest.fixture(params=["jit", "nojit"])
    def group_weight(self, request):
        func = kernel(jit._group_weight, request.param)
        offset = np.array([0, 2, 2, 5], dtype=np.int32)
        site = np.array([1, 3, 0, 2, 2], dtype=np.int32)
        weight_sum = np.array([0.0, 1.0, 3.0, 7.0, 15.0, 31.0])

        def f(group, start, stop):
            return func(offset, site, weight_sum, group, start, stop)

        return f

    def test_whole_group(self, group_weight):
        assert group_weight(0, 0, 4) == 3.0
        assert group_weight(2, 0, 4) == 28.0

    def test_empty_group(self, group_weight):
        assert group_weight(1, 0, 4) == 0.0

    def test_partial_range(self, group_weight):
        assert group_weight(0, 0, 2) == 1.0
        assert group_weight(0, 2, 4) == 2.0
        # Both entries at site 2 are inside the range or outside it together.
        assert group_weight(2, 2, 3) == 24.0
        assert group_weight(2, 0, 1) == 4.0

    def test_range_outside_group(self, group_weight):
        assert group_weight(0, 4, 8) == 0.0


class TestBalancedBinaryTree:
    """
    tskit.Tree.generate_balanced(4)::

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
        # The causal allele occurs nowhere in the tree.
        np.testing.assert_array_equal(
            node_genetic_value(tree, [], "T"), [0, 0, 0, 0, 0, 0, 0]
        )

    def test_leaf(self, tree, node_genetic_value):
        # Node 0 has no children, so the descent stops as soon as it is
        # reached.
        np.testing.assert_array_equal(
            node_genetic_value(tree, [(0, "T")], "T"), [1, 0, 0, 0, 0, 0, 0]
        )

    def test_internal_node(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [(4, "T")], "T"), [1, 1, 0, 0, 1, 0, 0]
        )

    def test_root(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [(6, "T")], "T"), [1, 1, 1, 1, 1, 1, 1]
        )

    def test_ancestral_state_is_causal(self, tree, node_genetic_value):
        # Every node carries the ancestral state, so the value is seeded at
        # the root and never changes.
        np.testing.assert_array_equal(node_genetic_value(tree), [1, 1, 1, 1, 1, 1, 1])

    def test_mutation_on_internal_node(self, tree, node_genetic_value):
        # Node 4 and its children carry the allele the mutation on node 4
        # introduced, which is not the causal one.
        np.testing.assert_array_equal(
            node_genetic_value(tree, [(6, "T"), (4, "G")], "T"), [0, 0, 1, 1, 0, 1, 1]
        )

    def test_mutation_on_leaf(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [(6, "T"), (0, "G")], "T"), [0, 1, 1, 1, 1, 1, 1]
        )

    def test_back_mutations(self, tree, node_genetic_value):
        # The mutations on nodes 1 and 5 are back to the causal allele, which
        # the state changes telescope to without any special handling.
        np.testing.assert_array_equal(
            node_genetic_value(tree, [(1, "T"), (5, "T")], "T"),
            [0, 1, 1, 1, 0, 1, 0],
        )

    def test_effect_size(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [(4, "T")], "T", effect_size=-2.5),
            [-2.5, -2.5, 0, 0, -2.5, 0, 0],
        )

    def test_edge_level(self, tree, node_genetic_value):
        # The value arriving at a node is credited to the edge above it, and
        # the root has no edge. generate_balanced(4) has edges 0..3 into the
        # leaves and edges 4 and 5 into nodes 4 and 5.
        ts = one_site(tree, [(6, "T")])
        value = node_genetic_value(ts, causal_allele="T", level="edge")
        expected = np.zeros(ts.num_edges)
        for u in range(6):
            expected[ts.first().edge(u)] = 1
        np.testing.assert_array_equal(value, expected)


class TestStarTree:
    """
    tskit.Tree.generate_star(5). The polytomy at the root makes the descent
    iterate over five child edges::

            5
        +-+-+-+-+
        0 1 2 3 4
    """

    @pytest.fixture
    def tree(self):
        return tskit.Tree.generate_star(5)

    def test_root(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [(5, "T")], "T"), [1, 1, 1, 1, 1, 1]
        )

    def test_ancestral_state_is_causal(self, tree, node_genetic_value):
        np.testing.assert_array_equal(node_genetic_value(tree), [1, 1, 1, 1, 1, 1])

    def test_mutation_on_root(self, tree, node_genetic_value):
        # The root is the only child of the virtual root, so a mutation there
        # takes the causal allele away from the whole tree.
        np.testing.assert_array_equal(
            node_genetic_value(tree, [(5, "G")]), [0, 0, 0, 0, 0, 0]
        )

    def test_mutations_across_the_polytomy(self, tree, node_genetic_value):
        # Mutations at both ends of the run of children and in the middle.
        np.testing.assert_array_equal(
            node_genetic_value(tree, [(5, "T"), (0, "G"), (2, "G"), (4, "G")], "T"),
            [0, 1, 0, 1, 0, 1],
        )


class TestNonBinaryTree:
    """
    tskit.Tree.generate_balanced(6, arity=3), with the polytomy below the
    root::

             9
         +---+---+
         6   7   8
        +++ +++ +++
        0 1 2 3 4 5
    """

    @pytest.fixture
    def tree(self):
        return tskit.Tree.generate_balanced(6, arity=3)

    def test_ancestral_state_is_causal(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )

    def test_middle_subtree(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [(7, "T")], "T"),
            [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
        )

    def test_mutation_on_middle_subtree(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [(7, "G")]),
            [1, 1, 0, 0, 1, 1, 1, 0, 1, 1],
        )

    def test_mutations_on_outer_subtrees(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [(6, "G"), (8, "G")]),
            [0, 0, 1, 1, 0, 0, 0, 1, 0, 1],
        )


class TestCombTree:
    """
    tskit.Tree.generate_comb(5). A ladder is the deepest descent for a given
    number of leaves::

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
            node_genetic_value(tree, [(8, "T")], "T"), [1, 1, 1, 1, 1, 1, 1, 1, 1]
        )

    def test_ancestral_state_is_causal(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree), [1, 1, 1, 1, 1, 1, 1, 1, 1]
        )

    def test_second_rung(self, tree, node_genetic_value):
        # Everything below node 7, which is all of the tree except leaf 0 and
        # the root.
        np.testing.assert_array_equal(
            node_genetic_value(tree, [(7, "T")], "T"), [0, 1, 1, 1, 1, 1, 1, 1, 0]
        )

    def test_mutation_near_root(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [(7, "G")]), [1, 0, 0, 0, 0, 0, 0, 0, 1]
        )

    def test_mutation_near_leaves(self, tree, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(tree, [(5, "G")]), [1, 1, 1, 0, 0, 0, 1, 1, 1]
        )


class TestDegenerateTrees:
    """
    Tree sequences that do not have a single root above a set of internal
    nodes.
    """

    def test_single_node(self, node_genetic_value):
        # generate_balanced(1) is the single node 0, which is its own root.
        tree = tskit.Tree.generate_balanced(1)
        np.testing.assert_array_equal(node_genetic_value(tree), [1])
        np.testing.assert_array_equal(node_genetic_value(tree, [(0, "T")], "T"), [1])

    def test_no_nodes(self, node_genetic_value):
        np.testing.assert_array_equal(node_genetic_value(empty_ts()), [])

    def test_isolated_samples(self, node_genetic_value):
        # Nodes 0, 1 and 2 are all roots:
        #
        #     0 1 2
        #
        np.testing.assert_array_equal(
            node_genetic_value(isolated_samples_ts(3)), [1, 1, 1]
        )

    def test_isolated_samples_with_mutation(self, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(isolated_samples_ts(3, [(1, "G")])), [1, 0, 1]
        )

    def test_multiple_roots(self, node_genetic_value):
        # Nodes 4 and 5 are both roots and node 6 is isolated, so it is never
        # reached:
        #
        #      4   5
        #     +++ +++
        #     0 1 2 3
        #
        tree = multiroot_tree()
        assert tree.roots == [4, 5]
        np.testing.assert_array_equal(node_genetic_value(tree), [1, 1, 1, 1, 1, 1, 0])

    def test_multiple_roots_one_blocked(self, node_genetic_value):
        np.testing.assert_array_equal(
            node_genetic_value(multiroot_tree(), [(4, "G")]), [0, 0, 1, 1, 0, 1, 0]
        )


class TestAccumulateIndividualValues:
    """
    Sum the node genetic values over the nodes of each individual, using the
    tree sequences of tests.data.
    """

    def test_binary_tree(self, individual_genetic_value):
        # Individual 0 is nodes 4 and 5, individual 1 nodes 0 and 1, and
        # individual 2 nodes 2 and 3.
        ts = binary_tree()
        np.testing.assert_array_equal(
            individual_genetic_value(ts, [1, 2, 4, 8, 16, 32, 64]), [48, 3, 12]
        )

    def test_diff_ind_tree(self, individual_genetic_value):
        ts = diff_ind_tree()
        np.testing.assert_array_equal(
            individual_genetic_value(ts, [1, 2, 4, 8, 16, 32, 64]), [48, 5, 10]
        )

    def test_triploid_tree(self, individual_genetic_value):
        # Individual 0 is nodes 0, 2 and 4, and individual 1 is nodes 1, 3
        # and 5.
        ts = triploid_tree()
        np.testing.assert_array_equal(
            individual_genetic_value(ts, [1, 2, 4, 8, 16, 32, 64, 128]), [21, 42]
        )

    def test_no_individuals(self, individual_genetic_value):
        ts = tskit.Tree.generate_balanced(4).tree_sequence
        assert ts.num_individuals == 0
        np.testing.assert_array_equal(
            individual_genetic_value(ts, np.ones(ts.num_nodes)), []
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
        value = node_genetic_value(one_site(ts.first(), [(4, "T")]), causal_allele="T")
        np.testing.assert_array_equal(value, [1, 1, 0, 0, 1, 0, 0])
        # Individual 0 has one copy through node 4, individual 1 has two
        # copies through nodes 0 and 1, and individual 2 has none.
        np.testing.assert_array_equal(individual_genetic_value(ts, value), [1, 2, 0])

    def test_ancestral_state_is_causal(
        self, node_genetic_value, individual_genetic_value
    ):
        # The causal allele is the ancestral state, so every node carries it
        # and every diploid has two copies.
        ts = binary_tree()
        value = node_genetic_value(one_site(ts.first(), []))
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
        value = node_genetic_value(one_site(ts.first(), [(6, "T")]), causal_allele="T")
        np.testing.assert_array_equal(value, [0, 0, 0, 1, 1, 1, 1, 0])
        # Individual 0 is nodes 0, 2 and 4, and individual 1 is nodes 1, 3
        # and 5.
        np.testing.assert_array_equal(individual_genetic_value(ts, value), [1, 2])
