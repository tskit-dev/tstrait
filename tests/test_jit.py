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
kernel works from the trees of a tree sequence rather than from a tree.
"""

import numba
import numpy as np
import pandas as pd
import pytest
import tskit
import tskit.jit.numba as tskit_numba

from tstrait import jit
from tstrait.genetic_value import _GeneticValue

from .data import (
    all_trees_ts,
    binary_tree,
    binary_tree_seq,
    diff_ind_tree,
    non_binary_tree,
    simple_tree_seq,
    triploid_tree,
)  # noreorder

ANCESTRAL_STATE = "A"


# The kernels that _descend_trees and the test driver below call. A numba
# function calls whatever the name is bound to in the module when it runs, so
# swapping these for the Python they were written as makes a caller that is
# itself untranslated interpreted the whole way down rather than only in its
# own loop.
TREE_KERNELS = [
    "tree_state",
    "_remove_edge",
    "_insert_edge",
    "_apply_edge_diffs",
    "_tree_roots",
]


def kernel(func, param, monkeypatch):
    """
    Return either the compiled kernel or the Python it was written as, in the
    latter case untranslating the kernels it calls along with it.
    """
    if param == "jit":
        return func
    for name in TREE_KERNELS:
        monkeypatch.setattr(jit, name, getattr(jit, name).py_func)
    return func.py_func


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
def node_genetic_value(request, monkeypatch):
    """
    Return a function computing the node genetic values of a tree carrying one
    causal site.

    The ``mutations`` are ``(node, derived_state)`` pairs against an ancestral
    state of "A", and a node's value is ``effect_size`` when the allele it
    inherits is ``causal_allele``.
    """
    func = kernel(jit._descend_trees, request.param, monkeypatch)

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
        # The kernel accumulates into a row per trait and returns the number
        # of nodes it visited, so the values come back through the array.
        output = np.zeros((genetic.num_trait, genetic._output_size(level)))
        func(**genetic._descend_arguments(level, output))
        return output[0]

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


class TestIndividualLevel:
    """
    The individual level, which is the node level with a different output
    mapping rather than a pass of its own, over the tree sequences of
    tests.data. The tree of tests.data.binary_tree is::

           6
         +-+-+
         4   5
        +++ +++
        0 1 2 3

    A mutation on node 4 is inherited by nodes 0 and 1, which lands on
    different individuals in each of these tree sequences and so pins the
    mapping rather than only the total.
    """

    def test_binary_tree(self, node_genetic_value):
        # Individual 0 is nodes 4 and 5, individual 1 nodes 0 and 1, and
        # individual 2 nodes 2 and 3, so individual 1 has two copies.
        ts = binary_tree()
        site = one_site(ts.first(), [(4, "T")])
        np.testing.assert_array_equal(
            node_genetic_value(site, causal_allele="T"), [1, 1, 0, 0, 1, 0, 0]
        )
        np.testing.assert_array_equal(
            node_genetic_value(site, causal_allele="T", level="individual"), [1, 2, 0]
        )

    def test_diff_ind_tree(self, node_genetic_value):
        # The same tree and mutation, but individual 1 is nodes 0 and 2 and
        # individual 2 is nodes 1 and 3, so the two copies are split.
        ts = diff_ind_tree()
        value = node_genetic_value(
            one_site(ts.first(), [(4, "T")]), causal_allele="T", level="individual"
        )
        np.testing.assert_array_equal(value, [1, 1, 1])

    def test_triploid_tree(self, node_genetic_value):
        # tests.data.triploid_tree, in which node 6 is the parent of nodes 3,
        # 4 and 5 and node 7 is the root:
        #
        #     7
        #     +-+-+---+
        #     | | |   6
        #     | | | +-+-+
        #     0 1 2 3 4 5
        #
        # Individual 0 is nodes 0, 2 and 4, and individual 1 is nodes 1, 3
        # and 5.
        ts = triploid_tree()
        site = one_site(ts.first(), [(6, "T")])
        np.testing.assert_array_equal(
            node_genetic_value(site, causal_allele="T"), [0, 0, 0, 1, 1, 1, 1, 0]
        )
        np.testing.assert_array_equal(
            node_genetic_value(site, causal_allele="T", level="individual"), [1, 2]
        )

    def test_ancestral_state_is_causal(self, node_genetic_value):
        # The causal allele is the ancestral state, so every node carries it
        # and every diploid has two copies.
        ts = binary_tree()
        value = node_genetic_value(one_site(ts.first(), []), level="individual")
        np.testing.assert_array_equal(value, [2, 2, 2])

    def test_no_individuals(self, node_genetic_value):
        # Nodes belong to no individual, so every contribution is discarded.
        tree = tskit.Tree.generate_balanced(4)
        assert tree.tree_sequence.num_individuals == 0
        value = node_genetic_value(
            one_site(tree, [(4, "T")]), causal_allele="T", level="individual"
        )
        np.testing.assert_array_equal(value, [])

    def test_matches_node_level(self, node_genetic_value):
        # The individual values are the node values summed over each
        # individual's nodes, however the nodes are assigned.
        ts = diff_ind_tree()
        site = one_site(ts.first(), [(4, "T")])
        nodes = node_genetic_value(site, causal_allele="T")
        expected = np.bincount(
            ts.nodes_individual[ts.nodes_individual >= 0],
            weights=nodes[ts.nodes_individual >= 0],
            minlength=ts.num_individuals,
        )
        np.testing.assert_array_equal(
            node_genetic_value(site, causal_allele="T", level="individual"), expected
        )


@numba.njit
def _walk_trees(
    numba_ts,
    edges_parent,
    edges_child,
    samples,
    parent,
    left_child,
    right_sib,
    node_edge,
    roots,
    num_roots,
):
    """
    Build every tree in turn, recording the state of each one.

    This is the loop that drives the tree state kernels, kept here rather than
    imported because the production driver does its per site work inside it and
    has nowhere to record a tree. It is three lines; everything it calls is the
    code under test.
    """
    tree = jit.tree_state(numba_ts.num_nodes)
    marked = np.full(numba_ts.num_nodes, -1, dtype=np.int64)
    scratch = np.empty(numba_ts.num_nodes, dtype=np.int32)
    tree_index = numba_ts.tree_index()
    i = 0
    while tree_index.next():
        jit._apply_edge_diffs(tree_index, edges_parent, edges_child, tree)
        parent[i, :] = tree.parent
        left_child[i, :] = tree.left_child
        right_sib[i, :] = tree.right_sib
        node_edge[i, :] = tree.node_edge
        n = jit._tree_roots(tree, samples, marked, i, scratch)
        num_roots[i] = n
        roots[i, :n] = scratch[:n]
        i += 1
    return i


@pytest.fixture(params=["jit", "nojit"])
def walk_trees(request, monkeypatch):
    """
    Return a function giving the per tree state arrays that _walk_trees records.
    """
    driver = kernel(_walk_trees, request.param, monkeypatch)

    def f(ts):
        return _walk(driver, ts)

    return f


def _walk(driver, ts):
    numba_ts = tskit_numba.jitwrap(ts)
    shape = (max(ts.num_trees, 1), ts.num_nodes)
    got = {
        name: np.zeros(shape, dtype=np.int32)
        for name in ("parent", "left_child", "right_sib", "node_edge", "roots")
    }
    num_roots = np.zeros(shape[0], dtype=np.int32)
    count = driver(
        numba_ts,
        ts.edges_parent,
        ts.edges_child,
        ts.samples().astype(np.int32),
        got["parent"],
        got["left_child"],
        got["right_sib"],
        got["node_edge"],
        got["roots"],
        num_roots,
    )
    assert count == ts.num_trees
    return got, num_roots


def unbalanced_ts(n):
    """
    Return a comb tree sequence, the worst case for anything that walks from a
    node to its root.
    """
    return tskit.Tree.generate_comb(n).tree_sequence


class TestTreeState:
    """
    The incrementally maintained tree must match what tskit builds, tree for
    tree, over topologies that exercise multiple roots, isolated samples and
    nodes that are in no tree at all.
    """

    @pytest.mark.parametrize(
        "ts",
        [
            binary_tree(),
            diff_ind_tree(),
            non_binary_tree(),
            triploid_tree(),
            binary_tree_seq(),
            simple_tree_seq(),
            all_trees_ts(2),
            all_trees_ts(3),
            all_trees_ts(4),
            all_trees_ts(5),
            unbalanced_ts(10),
            unbalanced_ts(50),
            tskit.Tree.generate_balanced(8).tree_sequence,
            multiroot_tree().tree_sequence,
            isolated_samples_ts(4),
            empty_ts(),
        ],
    )
    def test_matches_tskit(self, ts, walk_trees):
        got, num_roots = walk_trees(ts)
        n = ts.num_nodes
        for i, tree in enumerate(ts.trees()):
            # tskit's arrays carry the virtual root in a final entry, which the
            # kernel has no use for and does not keep.
            np.testing.assert_array_equal(got["parent"][i, :n], tree.parent_array[:n])
            np.testing.assert_array_equal(
                got["left_child"][i, :n], tree.left_child_array[:n]
            )
            np.testing.assert_array_equal(got["node_edge"][i, :n], tree.edge_array[:n])
            # tskit threads the roots together as children of the virtual root,
            # so they are siblings of each other there and have no sibling
            # here. The descent never walks the siblings of a root, since it
            # starts at the roots that _tree_roots gives it, so the two agree
            # everywhere the descent looks.
            child = tree.parent_array[:n] != tskit.NULL
            np.testing.assert_array_equal(
                got["right_sib"][i, :n][child], tree.right_sib_array[:n][child]
            )
            assert np.all(got["right_sib"][i, :n][~child] == tskit.NULL)
            assert sorted(got["roots"][i, : num_roots[i]]) == sorted(tree.roots)
