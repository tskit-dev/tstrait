import numpy as np
import pandas as pd
import tskit
import tskit.jit.numba as tskit_numba

from . import jit
from .base import _check_dataframe, _check_instance, _check_non_decreasing  # noreorder


def _row_mutations(ts, trait_df):
    """
    Expand the trait dataframe to one entry per (row, mutation at that row's
    site) pair, returning the row, the mutation, whether the mutation carries
    the causal allele and whether the state it replaced did.

    Every mutation at a causal site is returned, including those that leave the
    causal allele state unchanged. A descent of the trees needs all of them,
    because a mutation blocks the inheritance of the allele above it whatever
    it changes the state to; ``_causal_mutations`` drops the ones that a sum of
    state changes does not need.
    """
    site_id = trait_df["site_id"].to_numpy()
    # Mutations are sorted by site, so each site owns a contiguous run of IDs.
    offset = np.searchsorted(ts.mutations_site, np.arange(ts.num_sites + 1))
    start = offset[site_id]
    count = offset[site_id + 1] - start
    row = np.repeat(np.arange(len(trait_df)), count)
    mutation = np.arange(count.sum()) + np.repeat(
        start - (np.cumsum(count) - count), count
    )

    derived_state = ts.mutations_derived_state
    causal_allele = np.asarray(
        trait_df["causal_allele"].to_numpy(), dtype=derived_state.dtype
    )[row]
    had_causal_allele = ts.mutations_inherited_state[mutation] == causal_allele
    has_causal_allele = derived_state[mutation] == causal_allele
    return row, mutation, has_causal_allele, had_causal_allele


def _causal_mutations(ts, trait_df):
    """
    Expand the trait dataframe to one entry per (row, mutation at that row's
    site) pair, and return the row, the mutation and the change in causal
    allele state for the mutations that change the state. Mutations that leave
    the state unchanged contribute nothing and are dropped.
    """
    row, mutation, has_causal_allele, had_causal_allele = _row_mutations(ts, trait_df)
    state_change = has_causal_allele.astype(np.int8) - had_causal_allele.astype(np.int8)
    changed = state_change != 0
    return row[changed], mutation[changed], state_change[changed]


def _root_runs(ts):
    """
    Return the roots of the tree sequence as ``(node, left, right)`` arrays,
    where each root spans a maximal interval over which it is a root.

    The roots are taken as the children of the virtual root, so that a node
    counts as being in a tree exactly when the tree based implementation would
    have reached it from the virtual root. Isolated samples are roots.
    """
    node = []
    left = []
    right = []
    tree = tskit.Tree(ts)
    left_child_array = tree.left_child_array
    right_sib_array = tree.right_sib_array
    virtual_root = tree.virtual_root
    tree.first()
    while True:
        interval_left, interval_right = tree.interval
        u = left_child_array[virtual_root]
        while u != tskit.NULL:
            if len(node) > 0 and node[-1] == u and right[-1] == interval_left:
                right[-1] = interval_right
            else:
                node.append(u)
                left.append(interval_left)
                right.append(interval_right)
            u = right_sib_array[u]
        if not tree.next():
            break
    return (
        np.array(node, dtype=np.int32),
        np.array(left, dtype=float),
        np.array(right, dtype=float),
    )


def _check_trait_df(ts, trait_df):
    """
    Check the trait dataframe against the tree sequence, returning the required
    columns with trait_id cast to int.
    """
    trait_df = _check_dataframe(
        trait_df, ["site_id", "effect_size", "trait_id", "causal_allele"], "trait_df"
    )
    if len(trait_df) == 0:
        raise ValueError("trait_df must contain at least one row")
    _check_non_decreasing(trait_df["site_id"], "site_id")
    site_id = trait_df["site_id"].to_numpy()
    if site_id.min() < 0 or site_id.max() >= ts.num_sites:
        raise ValueError(f"site_id must be in [0, {ts.num_sites})")

    trait_id = trait_df["trait_id"].unique()

    if np.min(trait_id) != 0 or np.max(trait_id) != len(trait_id) - 1:
        raise ValueError("trait_id must be consecutive and start from 0")

    return trait_df.astype({"trait_id": int})


class _GeneticValue:
    """
    GeneticValue class to compute genetic values of individuals, nodes, or edges.

    The genetic values of every causal site are accumulated in a single descent
    of the ARG. Each causal mutation changes the causal allele state by
    ``[derived == causal] - [inherited == causal]``, and that change applies to
    every node below it. These changes telescope down a path, so a node's value
    is the sum of the changes on the mutations above it, plus the effect size
    of every causal site whose ancestral state is itself the causal allele.
    That makes the value additive along a root to node path, which is what the
    descent accumulates. Nested and back mutations need no special handling.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Tree sequence data with mutation
    trait_df : pandas.DataFrame
        Dataframe that includes causal site ID, causal allele, simulated effect
        size, and trait ID.
    """

    def __init__(self, ts, trait_df):
        self.trait_df = trait_df[["site_id", "effect_size", "trait_id", "causal_allele"]]
        self.ts = ts
        self.child_index = tskit_numba.jitwrap(ts).child_index()

        site_id = self.trait_df["site_id"].to_numpy()
        effect_size = self.trait_df["effect_size"].to_numpy()
        self.trait_id = self.trait_df["trait_id"].to_numpy()
        self.num_trait = np.max(self.trait_id) + 1

        # Causal sites are identified by their rank among the causal sites, so
        # that matching an edge to one is an integer comparison and a position
        # that cannot matter is never considered.
        causal_site = np.unique(site_id)
        causal_position = ts.sites_position[causal_site]
        rows_site = np.searchsorted(causal_site, site_id)
        self.edges_site_start = np.searchsorted(causal_position, ts.edges_left).astype(
            np.int32
        )
        self.edges_site_stop = np.searchsorted(causal_position, ts.edges_right).astype(
            np.int32
        )
        # tskit does not require the node IDs to be in time order.
        self.nodes_by_time = np.argsort(-ts.nodes_time, kind="stable").astype(np.int32)

        row, mutation, state_change = _causal_mutations(ts, self.trait_df)
        # The row a seed came from, so that a subset of the rows can be
        # selected without recomputing the seeds.
        self.seed_row = row
        self.seed_trait = self.trait_id[row]
        self.seed_node = ts.mutations_node[mutation].astype(np.int32)
        self.seed_site = rows_site[row].astype(np.int32)
        self.seed_weight = effect_size[row] * state_change
        # A mutation above a root has no edge, and needs no special handling:
        # seeding at its node and pushing down is already right, and the
        # missing edge contribution matches the tree based implementation.
        self.seed_edge = ts.mutations_edge[mutation]

        # When the ancestral state of a site is the causal allele, every node
        # in its tree carries it. There is no mutation to seed from, so the
        # roots are seeded instead and the effect reaches the same nodes.
        ancestral = np.flatnonzero(
            ts.sites_ancestral_state[site_id]
            == self.trait_df["causal_allele"].to_numpy()
        )
        if len(ancestral) > 0:
            roots, roots_left, roots_right = _root_runs(ts)
            start = np.searchsorted(causal_position, roots_left)
            stop = np.searchsorted(causal_position, roots_right)
            # Each root run takes the ancestral rows spanned by its interval.
            ancestral_site = rows_site[ancestral]
            first = np.searchsorted(ancestral_site, start)
            count = np.searchsorted(ancestral_site, stop) - first
            run = np.repeat(np.arange(len(roots)), count)
            index = np.arange(count.sum()) + np.repeat(
                first - (np.cumsum(count) - count), count
            )
            self.seed_row = np.concatenate([self.seed_row, ancestral[index]])
            self.seed_trait = np.concatenate(
                [self.seed_trait, self.trait_id[ancestral[index]]]
            )
            self.seed_node = np.concatenate([self.seed_node, roots[run]])
            self.seed_site = np.concatenate(
                [self.seed_site, ancestral_site[index].astype(np.int32)]
            )
            self.seed_weight = np.concatenate(
                [self.seed_weight, effect_size[ancestral[index]]]
            )
            # A root has no edge above it, so it contributes nothing at the
            # edge level, which is what the tree based implementation does too.
            self.seed_edge = np.concatenate(
                [self.seed_edge, np.full(len(run), tskit.NULL, dtype=np.int32)]
            )

    def _output_size(self, level):
        return {
            "individual": self.ts.num_individuals,
            "node": self.ts.num_nodes,
            "edge": self.ts.num_edges,
        }[level]

    def _node_output(self, level):
        """
        Return the output slot that a contribution arriving at each node is
        credited to, with a negative slot discarding it.

        Nodes and individuals differ only in this mapping, so the individual
        values fall out of the same descent rather than needing a pass of their
        own. A node belonging to no individual is tskit.NULL, which is already
        negative.
        """
        if level == "individual":
            return self.ts.nodes_individual
        return np.arange(self.ts.num_nodes, dtype=np.int32)

    def _descent_arguments(self, level, trait, output):
        """
        Return the arguments to the push down kernel for one trait, with the
        contributions directed at nodes, individuals or edges according to
        ``level`` and accumulated into ``output``.
        """
        ts = self.ts
        in_trait = self.seed_trait == trait
        if level == "edge":
            edges_output = np.arange(ts.num_edges, dtype=np.int32)
            # A seed is credited to the edge above the mutation, which does not
            # exist when the mutation is above a root.
            seed_output = self.seed_edge[in_trait].astype(np.int32)
        else:
            node_output = self._node_output(level)
            edges_output = node_output[ts.edges_child]
            seed_output = node_output[self.seed_node[in_trait]]

        return {
            "child_index": self.child_index,
            "edges_child": ts.edges_child,
            "edges_output": edges_output,
            "edges_site_start": self.edges_site_start,
            "edges_site_stop": self.edges_site_stop,
            "nodes_by_time": self.nodes_by_time,
            "seed_node": self.seed_node[in_trait],
            "seed_site": self.seed_site[in_trait],
            "seed_weight": self.seed_weight[in_trait],
            "seed_output": seed_output,
            "output": output,
        }

    def _run(self, level):
        """
        Computes genetic values of individuals, nodes, or edges
        depending on the value of "level"

        Returns
        -------
        pandas.DataFrame
            Dataframe with trait ID, [individual|node|edge] ID, and genetic value.
        """
        N = self._output_size(level)
        genetic_value_table = np.zeros((self.num_trait, N))
        for trait in range(self.num_trait):
            jit._push_down_arg(
                **self._descent_arguments(level, trait, genetic_value_table[trait])
            )

        return pd.DataFrame(
            {
                "trait_id": np.repeat(np.arange(self.num_trait), N),
                f"{level}_id": np.tile(np.arange(N), self.num_trait),
                "genetic_value": genetic_value_table.flatten(),
            }
        )


def genetic_value(ts, trait_df, level="individual"):
    """
    Compute genetic values for a tree sequence given a trait dataframe.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence data that will be used in the quantitative trait
        simulation.
    trait_df : pandas.DataFrame
        Trait dataframe. See :ref:`req_trait_df` for column and data
        requirements.
    level : {"individual", "node", "edge"}, default "individual"
        The level (entity) at which genetic values are returned.

    Returns
    -------
    pandas.DataFrame
        Genetic values for each trait and level (entity) in the tree sequence.
        The dataframe columns are ``trait_id``, the ID column corresponding to
        ``level`` (``individual_id``, ``node_id``, or ``edge_id``), and
        ``genetic_value``.

    See Also
    --------
    trait_model : Return a trait model, which can be used as `model` input.
    sim_trait : Return a trait dataframe, which can be used as a `trait_df` input.
    sim_env : Simulate environmental noise on top of genetic values.
    edge_effect : Compute effects introduced on each edge.

    Examples
    --------
    See :ref:`genetic_value_doc` and :ref:`genetic_individual_node_edge_doc` for
    worked examples, while :ref:`phenotype_model` describes
    quantitative genetics model assumptions.
    """

    ts = _check_instance(ts, "ts", tskit.TreeSequence)
    valid_levels = ("individual", "node", "edge")
    if level not in valid_levels:
        raise ValueError("level must be one of 'individual', 'node', or 'edge'")
    if level == "individual" and ts.num_individuals == 0:
        raise ValueError("No individuals in the provided tree sequence dataset")
    trait_df = _check_trait_df(ts, trait_df)

    genetic = _GeneticValue(ts=ts, trait_df=trait_df)

    genetic_result = genetic._run(level)

    return genetic_result


def edge_effect(ts, trait_df):
    """
    Compute effects introduced on each edge for a tree sequence
    given a trait dataframe.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence that will be used in the trait simulation.
    trait_df : pandas.DataFrame
        Trait dataframe. See :ref:`req_trait_df` for column and data
        requirements.

    Returns
    -------
    pandas.DataFrame
        Edge effects for each trait and edge in the tree sequence. The
        dataframe contains ``trait_id``, ``edge_id``, and ``effect_size``.

    Raises
    ------
    ValueError
        If a causal-allele transition occurs on a root node,
        which has no immediately ancestral edge to which an edge effect can be assigned.
    ValueError
        If a ``site_id`` in `trait_df` is not a valid site ID in `ts`.

    See Also
    --------
    genetic_value : Compute trait genetic values for individuals, nodes, or edges.

    Examples
    --------
    See :ref:`genetic_individual_node_edge_doc` for a worked example,
    while :ref:`phenotype_model` describes quantitative genetics model assumptions.
    """
    ts = _check_instance(ts, "ts", tskit.TreeSequence)
    trait_df = _check_trait_df(ts, trait_df)

    N = ts.num_edges
    num_trait = np.max(trait_df.trait_id) + 1
    effect_size = trait_df["effect_size"].to_numpy()
    trait_id = trait_df["trait_id"].to_numpy()

    row, mutation, state_change = _causal_mutations(ts, trait_df)
    edge = ts.mutations_edge[mutation]
    if np.any(edge == tskit.NULL):
        bad_mutation = mutation[edge == tskit.NULL][0]
        raise ValueError(
            "Cannot assign an edge effect to a mutation on a root node "
            f"(mutation {bad_mutation})"
        )
    edge_effect_table = np.bincount(
        trait_id[row] * N + edge,
        weights=effect_size[row] * state_change,
        minlength=num_trait * N,
    )

    df = pd.DataFrame(
        {
            "trait_id": np.repeat(np.arange(num_trait), N),
            "edge_id": np.tile(np.arange(N), num_trait),
            "effect_size": edge_effect_table,
        }
    )
    return df


def normalise_genetic_value(genetic_df, mean=0, var=1, ddof=1):
    """
    Normalise genetic value dataframe.

    Parameters
    ----------
    genetic_df : pandas.DataFrame
        Genetic value dataframe.
    mean : float, default 0
        Mean of the resulting genetic value.
    var : float, default 1
        Variance of the resulting genetic value.
    ddof : int, default 1
        Delta degrees of freedom. The divisor used in computing the variance
        is N - ddof, where N represents the number of elements.

    Returns
    -------
    pandas.DataFrame
        Dataframe with normalised genetic value.

    Raises
    ------
    ValueError
        If `var` <= 0.

    Notes
    -----
    The following columns must be included in `genetic_df`:

        * **trait_id**: Trait ID.
        * **individual_id**: Individual ID inside the tree sequence input.
        * **genetic_value**: Simulated genetic values.

    The dataframe output has the following columns:

        * **trait_id**: Trait ID.
        * **individual_id**: Individual ID inside the tree sequence input.
        * **genetic_value**: Normalised genetic values.

    Examples
    --------
    See :ref:`normalise_genetic_value` section for worked examples.
    """
    if var <= 0:
        raise ValueError("Variance must be greater than 0.")
    genetic_df = _check_dataframe(
        genetic_df, ["individual_id", "trait_id", "genetic_value"], "genetic_df"
    )
    grouped = genetic_df.groupby("trait_id")[["genetic_value"]]
    transformed_genetic = grouped.transform(lambda x: (x - x.mean()) / x.std(ddof=ddof))
    transformed_genetic = transformed_genetic * np.sqrt(var) + mean
    genetic_df.loc[:, "genetic_value"] = transformed_genetic

    return genetic_df
