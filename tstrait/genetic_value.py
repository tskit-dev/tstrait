import numpy as np
import pandas as pd
import tskit
import tskit.jit.numba as tskit_numba

from . import jit
from .base import _check_dataframe, _check_instance, _check_non_decreasing  # noreorder


def _row_mutations(ts, trait_df):
    """
    Expand the trait dataframe to one entry per (row, mutation at that row's
    site) pair, returning the row and the mutation.

    Every mutation at a causal site is returned. The descent needs all of them,
    because a mutation blocks the inheritance of the allele above it whatever
    it changes the state to.
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

    return row, mutation


def _row_causal_allele(ts, trait_df, row, mutation):
    """
    Whether each (row, mutation) pair's mutation carries the row's causal
    allele.
    """
    derived_state = ts.mutations_derived_state
    causal_allele = np.asarray(
        trait_df["causal_allele"].to_numpy(), dtype=derived_state.dtype
    )[row]
    return derived_state[mutation] == causal_allele, causal_allele


def _causal_mutations(ts, trait_df):
    """
    Expand the trait dataframe to one entry per (row, mutation at that row's
    site) pair, and return the row, the mutation and the change in causal
    allele state for the mutations that change the state. Mutations that leave
    the state unchanged contribute nothing and are dropped.
    """
    row, mutation = _row_mutations(ts, trait_df)
    has_causal_allele, causal_allele = _row_causal_allele(ts, trait_df, row, mutation)
    had_causal_allele = ts.mutations_inherited_state[mutation] == causal_allele
    state_change = has_causal_allele.astype(np.int8) - had_causal_allele.astype(np.int8)
    changed = state_change != 0
    return row[changed], mutation[changed], state_change[changed]


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

    The genetic values of every causal site are accumulated in a single pass
    over the trees. Each tree is built from the one before it, and for each
    causal site it carries the effect is added to the nodes that inherit the
    causal allele: those reached by descending from a mutation carrying it,
    stopping wherever another mutation at that site replaces the state. Where
    the causal allele is the site's ancestral state the roots are descended
    from instead, which reaches exactly the nodes of the tree.

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
        self.numba_ts = tskit_numba.jitwrap(ts)

        site_id = self.trait_df["site_id"].to_numpy()
        self.trait_id = self.trait_df["trait_id"].to_numpy()
        self.num_trait = np.max(self.trait_id) + 1

        # One entry per row of the trait dataframe, walked in step with the
        # trees, and one per mutation at each row's site.
        self.row_site = site_id.astype(np.int32)
        self.row_trait = self.trait_id.astype(np.int32)
        self.row_effect = self.trait_df["effect_size"].to_numpy().astype(float)
        self.row_ancestral = (
            ts.sites_ancestral_state[site_id]
            == self.trait_df["causal_allele"].to_numpy()
        )
        pair_row, pair_mutation = _row_mutations(ts, self.trait_df)
        self.pair_offset = np.searchsorted(
            pair_row, np.arange(len(self.trait_df) + 1)
        ).astype(np.int64)
        self.pair_node = ts.mutations_node[pair_mutation].astype(np.int32)
        self.pair_carries, _ = _row_causal_allele(
            ts, self.trait_df, pair_row, pair_mutation
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

    def _descend_arguments(self, level, output):
        """
        Return the arguments to the descent kernel, with the contributions
        directed at nodes, individuals or edges according to ``level`` and
        accumulated into ``output``.
        """
        ts = self.ts
        return {
            "numba_ts": self.numba_ts,
            "edges_parent": ts.edges_parent,
            "edges_child": ts.edges_child,
            "row_site": self.row_site,
            "row_trait": self.row_trait,
            "row_effect": self.row_effect,
            "row_ancestral": self.row_ancestral,
            "pair_offset": self.pair_offset,
            "pair_node": self.pair_node,
            "pair_carries": self.pair_carries,
            "samples": ts.samples().astype(np.int32),
            "node_output": self._node_output(level),
            "edge_level": level == "edge",
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
        jit._descend_trees(**self._descend_arguments(level, genetic_value_table))

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
