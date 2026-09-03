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

import numba
import numpy as np
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
