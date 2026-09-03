"""
Numba compiled kernels.

Two conventions apply throughout this module:

1. Genetic values are accumulated by building each tree in turn and descending
   from the mutations of the causal sites that tree carries, so that the cost
   is the number of nodes carrying a causal allele rather than the size of the
   tree sequence.

2. Numba compiles with bounds checking disabled by default, so an out of
   bounds access here fails silently rather than raising ``IndexError``.
   Keeping the compiled functions in one module lets us unit test them
   through their ``py_func`` attribute in ``tests/test_jit.py``, which runs
   the untranslated Python and therefore gets numpy's bounds checking and
   coverage measurement for free. ``_descend_trees`` calls the tree building
   kernels rather than their ``py_func``, so running it that way interprets
   only its own loop; those kernels are covered instead by ``TestTreeState``,
   which compares them against ``tskit.Tree`` tree for tree.

An earlier implementation pushed the effect of each causal mutation down the
ARG in a single sweep of the nodes from the past to the present, and never
built a tree at all. It was measured against this one over the whole benchmark
grid, on both tree sequences, at each level, for one and three traits, and for
causal sites drawn both uniformly and from the rare ones. It was slower
everywhere the cost mattered: 1.33 to 1.70 times on uniformly drawn causal
sites, and up to 2.27 times on rare ones with three traits, since it swept once
per trait where this makes a single pass for all of them. It was ahead only for
a few rare causal sites, which give the tree building here too little to
amortise it against, and then by a millisecond or two on a call taking a few.
It is in the git history if it is ever wanted back.
"""

from collections import namedtuple

import numba
import numpy as np
import tskit

# The state of one tree, in the tskit quintuply linked encoding. A tree is
# built by applying the edge differences between it and the tree before it, so
# a full pass over the trees costs one insertion and one removal per edge.
# There is no virtual root here: the descent starts at the mutations of a
# causal site, and the roots are only needed when the ancestral state is the
# causal allele, which _tree_roots finds on demand.
_TreeState = namedtuple(
    "_TreeState",
    ["parent", "left_child", "right_child", "left_sib", "right_sib", "node_edge"],
)


@numba.njit
def tree_state(num_nodes):
    """
    Return the arrays holding a tree, in the state of a tree with no edges.
    """
    return _TreeState(
        parent=np.full(num_nodes, tskit.NULL, dtype=np.int32),
        left_child=np.full(num_nodes, tskit.NULL, dtype=np.int32),
        right_child=np.full(num_nodes, tskit.NULL, dtype=np.int32),
        left_sib=np.full(num_nodes, tskit.NULL, dtype=np.int32),
        right_sib=np.full(num_nodes, tskit.NULL, dtype=np.int32),
        node_edge=np.full(num_nodes, tskit.NULL, dtype=np.int32),
    )


@numba.njit
def _remove_edge(tree, parent_node, child_node):
    """
    Detach ``child_node`` from ``parent_node``, unlinking it from its siblings.
    """
    left_sib = tree.left_sib[child_node]
    right_sib = tree.right_sib[child_node]
    if left_sib == tskit.NULL:
        tree.left_child[parent_node] = right_sib
    else:
        tree.right_sib[left_sib] = right_sib
    if right_sib == tskit.NULL:
        tree.right_child[parent_node] = left_sib
    else:
        tree.left_sib[right_sib] = left_sib
    tree.parent[child_node] = tskit.NULL
    tree.left_sib[child_node] = tskit.NULL
    tree.right_sib[child_node] = tskit.NULL
    tree.node_edge[child_node] = tskit.NULL


@numba.njit
def _insert_edge(tree, edge, parent_node, child_node):
    """
    Attach ``child_node`` to ``parent_node`` as its rightmost child, through
    ``edge``.
    """
    right_child = tree.right_child[parent_node]
    if right_child == tskit.NULL:
        tree.left_child[parent_node] = child_node
        tree.left_sib[child_node] = tskit.NULL
    else:
        tree.right_sib[right_child] = child_node
        tree.left_sib[child_node] = right_child
    tree.right_child[parent_node] = child_node
    tree.right_sib[child_node] = tskit.NULL
    tree.parent[child_node] = parent_node
    tree.node_edge[child_node] = edge


@numba.njit
def _apply_edge_diffs(tree_index, edges_parent, edges_child, tree):
    """
    Advance ``tree`` to the tree that ``tree_index`` has moved to, by applying
    the edges leaving and then the edges entering.
    """
    for j in range(tree_index.out_range.start, tree_index.out_range.stop):
        edge = tree_index.out_range.order[j]
        _remove_edge(tree, edges_parent[edge], edges_child[edge])
    for j in range(tree_index.in_range.start, tree_index.in_range.stop):
        edge = tree_index.in_range.order[j]
        _insert_edge(tree, edge, edges_parent[edge], edges_child[edge])


@numba.njit
def _tree_roots(tree, samples, marked, mark, roots):
    """
    Fill ``roots`` with the roots of the current tree and return how many there
    are.

    A root is a node with no parent that has a sample at or below it, which is
    what tskit calls a root, so walking up from each sample and taking the top
    of the path finds all of them and nothing else. ``marked`` records the
    nodes already walked through under the value ``mark``, so that the paths of
    all the samples together cost one visit per node rather than one per
    sample.
    """
    num_roots = 0
    for i in range(len(samples)):
        u = samples[i]
        while marked[u] != mark:
            marked[u] = mark
            parent = tree.parent[u]
            if parent == tskit.NULL:
                roots[num_roots] = u
                num_roots += 1
                break
            u = parent
    return num_roots


@numba.njit
def _descend_trees(
    numba_ts,
    edges_parent,
    edges_child,
    row_site,
    row_trait,
    row_effect,
    row_ancestral,
    pair_offset,
    pair_node,
    pair_carries,
    samples,
    node_output,
    edge_level,
    output,
):
    """
    Accumulate the genetic value of the selected causal sites in one pass over
    the trees, descending from each causal mutation to the nodes that inherit
    the causal allele from it.

    The rows are sorted by site and each site belongs to one tree, so a single
    pointer walks the rows in step with the trees. A row's mutations are marked
    with the row's own index rather than into a cleared array, so a descent
    costs the nodes it reaches and nothing per node of the tree sequence.

    ``node_output`` gives the index in ``output`` that a node's contribution is
    added to, with a negative index discarding it; at the edge level the index
    is instead the edge above the node in the current tree, which is where the
    contribution arrived from. All of the traits share the pass, since a row
    knows the trait it belongs to.

    ``output`` is accumulated into, and the count of nodes visited is returned.
    That count is what the run time is proportional to, so the benchmark
    divides by it rather than keeping a copy of this loop that only counts.
    """
    num_nodes = numba_ts.num_nodes
    tree = tree_state(num_nodes)
    # Marked with the row rather than cleared, so that nothing here costs a
    # pass over the nodes.
    stamp = np.full(num_nodes, -1, dtype=np.int64)
    marked = np.full(num_nodes, -1, dtype=np.int64)
    carries = np.zeros(num_nodes, dtype=np.bool_)
    roots = np.empty(num_nodes, dtype=np.int32)
    # A node is reached by at most one seed of a row, because the descent from
    # a seed stops at the mutations of the row, so the tree bounds the stack.
    stack = np.empty(num_nodes, dtype=np.int32)

    visits = 0
    row = 0
    num_rows = len(row_site)
    tree_index = numba_ts.tree_index()
    while tree_index.next():
        _apply_edge_diffs(tree_index, edges_parent, edges_child, tree)
        site_stop = tree_index.site_range[1]
        while row < num_rows and row_site[row] < site_stop:
            start = pair_offset[row]
            stop = pair_offset[row + 1]
            # Every mutation at the site blocks the allele above it, whatever
            # it changes the state to. A node carrying more than one of them
            # takes the last, which is the youngest since tskit orders a
            # mutation after its parent.
            for k in range(start, stop):
                stamp[pair_node[k]] = row
                carries[pair_node[k]] = pair_carries[k]

            top = 0
            for k in range(start, stop):
                node = pair_node[k]
                if carries[node]:
                    # Cleared so that a node carrying several mutations at this
                    # site is seeded once.
                    carries[node] = False
                    stack[top] = node
                    top += 1
            if row_ancestral[row]:
                # The causal allele is the ancestral state, so it reaches every
                # node the roots reach. A root carrying a mutation is not one
                # of them: the mutation replaced the ancestral state there, and
                # it has already been seeded above if it carries the allele.
                num_roots = _tree_roots(tree, samples, marked, row, roots)
                for i in range(num_roots):
                    if stamp[roots[i]] != row:
                        stack[top] = roots[i]
                        top += 1

            trait = row_trait[row]
            weight = row_effect[row]
            while top > 0:
                top -= 1
                node = stack[top]
                visits += 1
                slot = tree.node_edge[node] if edge_level else node_output[node]
                if slot >= 0:
                    output[trait, slot] += weight
                child = tree.left_child[node]
                while child != tskit.NULL:
                    if stamp[child] != row:
                        stack[top] = child
                        top += 1
                    child = tree.right_sib[child]
            row += 1
    return visits
