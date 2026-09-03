"""
Numba compiled kernels.

Two conventions apply throughout this module:

1. Genetic values are accumulated by pushing the effect of each causal
   mutation down the ARG, rather than by working through the trees one at a
   time. Positions are expressed as indexes into the sorted array of causal
   sites, so that an edge is matched to a causal site with an integer
   comparison and a genomic coordinate never appears.

2. Numba compiles with bounds checking disabled by default, so an out of
   bounds access here fails silently rather than raising ``IndexError``.
   Keeping the compiled functions in one module lets us unit test them
   through their ``py_func`` attribute in ``tests/test_jit.py``, which runs
   the untranslated Python and therefore gets numpy's bounds checking and
   coverage measurement for free.
"""

from collections import namedtuple

import numba
import numpy as np
import tskit
from numba.core import types
from numba.typed import List

# What a node holds while it waits to be swept: the indexes of the seeds whose
# effect has reached it. The causal site and effect size of a seed are looked
# up in arrays that are fixed for the whole sweep, so an item is one integer.
_SEED_LIST = types.ListType(types.int32)


@numba.njit
def _push_down_arg(
    child_index,
    edges_child,
    edges_output,
    edges_site_start,
    edges_site_stop,
    nodes_by_time,
    seed_node,
    seed_site,
    seed_weight,
    seed_output,
    output,
):
    """
    Accumulate the genetic value of every causal site in one sweep of the nodes.

    Each node holds the seeds whose effect has reached it, seeded at the node
    of each causal mutation. Sweeping the nodes from the past to the present
    visits every node after all of its ancestors, because a parent is always
    older than its child, so a seed can be passed from a node to its children
    and is guaranteed to have arrived before that child is reached. For each
    seed the outbound edges spanning its causal site are found, and the seed is
    credited to the edge and to the node at the far end before being passed on
    to it.

    A node is credited when a seed arrives rather than when it is swept, so a
    node that is never a parent never holds anything. On a large tree sequence
    that is about half of the work.

    ``child_index`` is the tskit child index, in which a node that is never a
    parent has the range ``(-1, -1)``. ``nodes_by_time`` lists the nodes from
    the oldest to the youngest. ``edges_output`` and ``seed_output`` give the
    index in ``output`` that each contribution is added to, which is how the
    same sweep serves both node and edge genetic values; a negative index
    discards the contribution.
    """
    num_nodes = len(child_index)
    # A node's list is made when something first reaches it. Most nodes are
    # never reached when the causal alleles are rare, and making a list for
    # every one of them up front then costs more than the sweep does.
    empty = List.empty_list(types.int32)
    pending = List.empty_list(_SEED_LIST)
    for _ in range(num_nodes):
        pending.append(empty)
    reached = np.zeros(num_nodes, dtype=np.bool_)

    for j in range(len(seed_node)):
        u = seed_node[j]
        if seed_output[j] >= 0:
            output[seed_output[j]] += seed_weight[j]
        if child_index[u, 0] < 0:
            continue
        if not reached[u]:
            pending[u] = List.empty_list(types.int32)
            reached[u] = True
        pending[u].append(np.int32(j))

    for i in range(len(nodes_by_time)):
        parent = nodes_by_time[i]
        if not reached[parent]:
            continue
        items = pending[parent]
        edge_start = child_index[parent, 0]
        edge_stop = child_index[parent, 1]
        for k in range(len(items)):
            item = items[k]
            site = seed_site[item]
            weight = seed_weight[item]
            for e in range(edge_start, edge_stop):
                if edges_site_start[e] <= site and site < edges_site_stop[e]:
                    if edges_output[e] >= 0:
                        output[edges_output[e]] += weight
                    child = edges_child[e]
                    if child_index[child, 0] < 0:
                        # Nothing below, so there is nothing to hold.
                        continue
                    if not reached[child]:
                        pending[child] = List.empty_list(types.int32)
                        reached[child] = True
                    pending[child].append(np.int32(item))
        # A node is swept once, so dropping the reference here hands its
        # storage back for the nodes still to come.
        pending[parent] = empty

    return output


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
