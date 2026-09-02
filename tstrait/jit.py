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
import tskit

# The arena starts at this many tuples and doubles whenever it fills up.
INITIAL_ARENA_SIZE = 1024


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

    Each node conceptually holds a set of ``(causal site, effect)`` tuples,
    seeded at the node of each causal mutation. Sweeping the nodes from the
    past to the present visits every node after all of its ancestors, because
    a parent is always older than its child, so a tuple can be pushed from a
    node to its children and is guaranteed to have arrived before that child is
    reached. For each tuple the outbound edges spanning its causal site are
    found, and the tuple is credited to the edge and to the node at the far end
    before being passed on to it.

    A node is credited when a tuple arrives rather than when it is swept, so a
    node that is never a parent never holds a tuple. On a large tree sequence
    that is about half of the work.

    ``child_index`` is the tskit child index, in which a node that is never a
    parent has the range ``(-1, -1)``. ``nodes_by_time`` lists the nodes from
    the oldest to the youngest. ``edges_output`` and ``seed_output`` give the
    index in ``output`` that each contribution is added to, which is how the
    same sweep serves both node and edge genetic values; a negative index
    discards the contribution.
    """
    num_nodes = len(child_index)
    head = np.full(num_nodes, tskit.NULL, dtype=np.int32)
    arena_size = INITIAL_ARENA_SIZE
    tuple_site = np.empty(arena_size, dtype=np.int32)
    tuple_weight = np.empty(arena_size, dtype=np.float64)
    tuple_next = np.empty(arena_size, dtype=np.int32)
    # Slots are taken from the free list first, and from the top of the arena
    # only when it is empty. Every tuple of a node is dead once that node has
    # been swept, so the arena only ever holds the tuples still in flight.
    arena_top = 0
    free = tskit.NULL

    for j in range(len(seed_node)):
        u = seed_node[j]
        weight = seed_weight[j]
        if seed_output[j] >= 0:
            output[seed_output[j]] += weight
        if child_index[u, 0] < 0:
            continue
        if free != tskit.NULL:
            slot = free
            free = tuple_next[slot]
        else:
            if arena_top == arena_size:
                arena_size *= 2
                grown_site = np.empty(arena_size, dtype=np.int32)
                grown_weight = np.empty(arena_size, dtype=np.float64)
                grown_next = np.empty(arena_size, dtype=np.int32)
                grown_site[:arena_top] = tuple_site
                grown_weight[:arena_top] = tuple_weight
                grown_next[:arena_top] = tuple_next
                tuple_site = grown_site
                tuple_weight = grown_weight
                tuple_next = grown_next
            slot = arena_top
            arena_top += 1
        tuple_site[slot] = seed_site[j]
        tuple_weight[slot] = weight
        tuple_next[slot] = head[u]
        head[u] = slot

    for i in range(len(nodes_by_time)):
        parent = nodes_by_time[i]
        slot = head[parent]
        if slot == tskit.NULL:
            continue
        head[parent] = tskit.NULL
        edge_start = child_index[parent, 0]
        edge_stop = child_index[parent, 1]
        while slot != tskit.NULL:
            site = tuple_site[slot]
            weight = tuple_weight[slot]
            for e in range(edge_start, edge_stop):
                if edges_site_start[e] <= site and site < edges_site_stop[e]:
                    if edges_output[e] >= 0:
                        output[edges_output[e]] += weight
                    child = edges_child[e]
                    if child_index[child, 0] < 0:
                        # Nothing below, so there is no tuple to hold.
                        continue
                    if free != tskit.NULL:
                        child_slot = free
                        free = tuple_next[child_slot]
                    else:
                        if arena_top == arena_size:
                            arena_size *= 2
                            grown_site = np.empty(arena_size, dtype=np.int32)
                            grown_weight = np.empty(arena_size, dtype=np.float64)
                            grown_next = np.empty(arena_size, dtype=np.int32)
                            grown_site[:arena_top] = tuple_site
                            grown_weight[:arena_top] = tuple_weight
                            grown_next[:arena_top] = tuple_next
                            tuple_site = grown_site
                            tuple_weight = grown_weight
                            tuple_next = grown_next
                        child_slot = arena_top
                        arena_top += 1
                    tuple_site[child_slot] = site
                    tuple_weight[child_slot] = weight
                    tuple_next[child_slot] = head[child]
                    head[child] = child_slot
            # This tuple is finished with, so its slot can be reused.
            next_slot = tuple_next[slot]
            tuple_next[slot] = free
            free = slot
            slot = next_slot

    return output


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
