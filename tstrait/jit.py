"""
Numba compiled kernels.

Two conventions apply throughout this module:

1. Genetic values are accumulated by descending the ARG rather than by
   working through the trees one at a time. Positions are expressed as
   indexes into the sorted array of causal sites, so that the descent can
   prune whole subtrees with integer comparisons and never looks at a
   genomic coordinate.

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

# The stacks start at this size and double whenever they fill up. A few
# hundred entries is typical, so this is usually the only allocation.
INITIAL_STACK_SIZE = 1024


@numba.njit
def _search_sorted(values, start, stop, key):
    """
    Return the first index in ``[start, stop)`` at which ``values`` is greater
    than or equal to ``key``. The values in that range must be sorted.
    """
    while start < stop:
        mid = (start + stop) // 2
        if values[mid] < key:
            start = mid + 1
        else:
            stop = mid
    return start


@numba.njit
def _group_weight(offset, site, weight_sum, group, start, stop):
    """
    Return the total weight of the entries in ``group`` whose causal site index
    is in ``[start, stop)``. Entries are grouped by ``offset`` and sorted by
    site within each group, and ``weight_sum`` is their cumulative sum with a
    leading zero.
    """
    group_start = offset[group]
    group_stop = offset[group + 1]
    if group_start == group_stop:
        return 0.0
    first = _search_sorted(site, group_start, group_stop, start)
    last = _search_sorted(site, group_start, group_stop, stop)
    return weight_sum[last] - weight_sum[first]


@numba.njit
def _descend_arg(
    child_index,
    edges_child,
    edges_output,
    edges_site_start,
    edges_site_stop,
    mutations_offset,
    mutations_site,
    mutations_weight_sum,
    ancestral_sum,
    roots,
    roots_output,
    roots_site_start,
    roots_site_stop,
    roots_mutations_offset,
    roots_mutations_site,
    roots_mutations_weight_sum,
    output,
):
    """
    Accumulate the genetic value of every causal site in one descent of the ARG.

    Each entry on the stack is a node together with the half open range of
    causal sites over which its path back to the root is unchanged, and the
    genetic value accumulated along that path. Descending an edge intersects
    the range with the edge's own range of causal sites, so a subtree spanning
    no causal site is never entered, and adds the effects of the causal
    mutations the edge carries within the intersected range.

    A node is reached once per such range, and the ranges partition the causal
    sites the node spans, so the values accumulate to the total over every
    causal site.

    The value carried down is a sum over the sites in the current range, so
    when an edge narrows the range it is no longer the value the child should
    inherit: the mutations above at the sites that dropped out do not apply to
    the child. The path back to the root is therefore kept in ``path_edge``, and
    the value is recomputed over the narrowed range whenever that happens. This
    is rare, a hundredth of a percent of descents on a large tree sequence,
    because an edge usually spans the whole of the range reaching it.

    ``child_index`` is the tskit child index, in which a node that is never a
    parent has the range ``(-1, -1)``. ``ancestral_sum`` is the cumulative
    effect size of the causal sites whose ancestral state is the causal allele,
    which every node in a tree carries. ``edges_output`` and ``roots_output``
    give the index in ``output`` that each contribution is added to, which is
    how the same descent serves both node and edge genetic values; a negative
    index discards the contribution.
    """
    stack_size = INITIAL_STACK_SIZE
    stack_node = np.empty(stack_size, dtype=np.int32)
    stack_start = np.empty(stack_size, dtype=np.int32)
    stack_stop = np.empty(stack_size, dtype=np.int32)
    stack_value = np.empty(stack_size, dtype=np.float64)
    stack_depth = np.empty(stack_size, dtype=np.int32)
    stack_edge = np.empty(stack_size, dtype=np.int32)
    path_size = INITIAL_STACK_SIZE
    path_edge = np.empty(path_size, dtype=np.int32)

    for j in range(len(roots)):
        root_start = roots_site_start[j]
        root_stop = roots_site_stop[j]
        if root_start >= root_stop:
            continue
        root = roots[j]
        root_value = (
            ancestral_sum[root_stop]
            - ancestral_sum[root_start]
            + _group_weight(
                roots_mutations_offset,
                roots_mutations_site,
                roots_mutations_weight_sum,
                j,
                root_start,
                root_stop,
            )
        )
        if roots_output[j] >= 0:
            output[roots_output[j]] += root_value
        if child_index[root, 0] < 0:
            continue
        stack_node[0] = root
        stack_start[0] = root_start
        stack_stop[0] = root_stop
        stack_value[0] = root_value
        stack_depth[0] = 0
        stack_edge[0] = tskit.NULL
        top = 1

        while top > 0:
            top -= 1
            parent = stack_node[top]
            start = stack_start[top]
            stop = stack_stop[top]
            parent_value = stack_value[top]
            depth = stack_depth[top]
            # Depth first order guarantees that the entries below this one in
            # path_edge are the edges back to the root.
            if depth > 0:
                path_edge[depth - 1] = stack_edge[top]

            for e in range(child_index[parent, 0], child_index[parent, 1]):
                edge_start = edges_site_start[e]
                child_start = start if start > edge_start else edge_start
                edge_stop = edges_site_stop[e]
                child_stop = stop if stop < edge_stop else edge_stop
                if child_start >= child_stop:
                    continue

                if child_start == start and child_stop == stop:
                    child_value = parent_value
                else:
                    child_value = (
                        ancestral_sum[child_stop]
                        - ancestral_sum[child_start]
                        + _group_weight(
                            roots_mutations_offset,
                            roots_mutations_site,
                            roots_mutations_weight_sum,
                            j,
                            child_start,
                            child_stop,
                        )
                    )
                    for k in range(depth):
                        child_value += _group_weight(
                            mutations_offset,
                            mutations_site,
                            mutations_weight_sum,
                            path_edge[k],
                            child_start,
                            child_stop,
                        )
                child_value += _group_weight(
                    mutations_offset,
                    mutations_site,
                    mutations_weight_sum,
                    e,
                    child_start,
                    child_stop,
                )

                if edges_output[e] >= 0:
                    output[edges_output[e]] += child_value

                # A node that is never a parent has nothing below it, so there
                # is no point putting it on the stack.
                child = edges_child[e]
                if child_index[child, 0] < 0:
                    continue
                if top == stack_size:
                    stack_size *= 2
                    grown_node = np.empty(stack_size, dtype=np.int32)
                    grown_start = np.empty(stack_size, dtype=np.int32)
                    grown_stop = np.empty(stack_size, dtype=np.int32)
                    grown_value = np.empty(stack_size, dtype=np.float64)
                    grown_depth = np.empty(stack_size, dtype=np.int32)
                    grown_edge = np.empty(stack_size, dtype=np.int32)
                    grown_node[: len(stack_node)] = stack_node
                    grown_start[: len(stack_start)] = stack_start
                    grown_stop[: len(stack_stop)] = stack_stop
                    grown_value[: len(stack_value)] = stack_value
                    grown_depth[: len(stack_depth)] = stack_depth
                    grown_edge[: len(stack_edge)] = stack_edge
                    stack_node = grown_node
                    stack_start = grown_start
                    stack_stop = grown_stop
                    stack_value = grown_value
                    stack_depth = grown_depth
                    stack_edge = grown_edge
                if depth + 1 == path_size:
                    path_size *= 2
                    grown_path = np.empty(path_size, dtype=np.int32)
                    grown_path[: len(path_edge)] = path_edge
                    path_edge = grown_path
                stack_node[top] = child
                stack_start[top] = child_start
                stack_stop[top] = child_stop
                stack_value[top] = child_value
                stack_depth[top] = depth + 1
                stack_edge[top] = e
                top += 1

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
