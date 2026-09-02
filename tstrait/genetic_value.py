import numpy as np
import pandas as pd
import tskit

from . import jit
from .base import _check_dataframe, _check_instance, _check_non_decreasing  # noreorder


def _accumulate_edge_values(nodes_genetic_value, nodes_edge, num_nodes, num_edges):
    """
    Accumulate the edge genetic values by summing their node contributions.
    """
    nodes_edge = nodes_edge[:num_nodes]
    nodes_genetic_value = nodes_genetic_value[:num_nodes]
    has_edge = nodes_edge != tskit.NULL
    return np.bincount(
        nodes_edge[has_edge],
        weights=nodes_genetic_value[has_edge],
        minlength=num_edges,
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

    def _node_genetic_values(self, tree, site, causal_allele, effect_size):
        """
        Returns a numpy array with node genetic values.
        """
        has_mutation = np.zeros(self.ts.num_nodes + 1, dtype=bool)
        state_transitions = {tree.virtual_root: site.ancestral_state}
        for m in site.mutations:
            state_transitions[m.node] = m.derived_state
            has_mutation[m.node] = True
        causal_nodes = np.array(
            [
                node
                for node, allele in state_transitions.items()
                if allele == causal_allele
            ],
            dtype=np.int32,
        )
        return jit._compute_nodes_genetic_value(
            left_child_array=tree.left_child_array,
            right_sib_array=tree.right_sib_array,
            causal_nodes=causal_nodes,
            has_mutation=has_mutation,
            effect_size=effect_size,
        )

    def _run(self, level):
        """
        Computes genetic values of individuals, nodes, or edges
        depending on the value of "level"

        Returns
        -------
        pandas.DataFrame
            Dataframe with trait ID, [individual|node|edge] ID, and genetic value.
        """

        ts = self.ts
        size_map = {
            "individual": ts.num_individuals,
            "node": ts.num_nodes,
            "edge": ts.num_edges,
        }
        N = size_map[level]

        num_trait = np.max(self.trait_df.trait_id) + 1
        genetic_value_table = np.zeros((num_trait, N))
        tree = tskit.Tree(self.ts)

        for data in self.trait_df.itertuples():
            site = self.ts.site(data.site_id)
            tree.seek(site.position)
            genetic_value = self._node_genetic_values(
                tree=tree,
                site=site,
                causal_allele=data.causal_allele,
                effect_size=data.effect_size,
            )
            if level == "individual":
                genetic_value = jit._accumulate_individual_values(
                    genetic_value, ts.nodes_individual, ts.num_individuals
                )
            elif level == "edge":
                genetic_value = _accumulate_edge_values(
                    genetic_value, tree.edge_array, ts.num_nodes, ts.num_edges
                )
            genetic_value_table[data.trait_id, :] += genetic_value

        df = pd.DataFrame(
            {
                "trait_id": np.repeat(np.arange(num_trait), N),
                f"{level}_id": np.tile(np.arange(N), num_trait),
                "genetic_value": genetic_value_table.flatten(),
            }
        )

        return df


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
    site_id = trait_df["site_id"].to_numpy()
    effect_size = trait_df["effect_size"].to_numpy()
    trait_id = trait_df["trait_id"].to_numpy()

    # Mutations are sorted by site, so each site owns a contiguous run of IDs.
    offset = np.searchsorted(ts.mutations_site, np.arange(ts.num_sites + 1))
    start = offset[site_id]
    count = offset[site_id + 1] - start
    # Expand to one entry per (trait_df row, mutation at that row's site) pair.
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
    state_change = has_causal_allele.astype(np.int8) - had_causal_allele.astype(np.int8)

    # Mutations that do not change the causal allele state contribute nothing, and
    # are allowed to sit above a root.
    changed = state_change != 0
    row = row[changed]
    mutation = mutation[changed]
    state_change = state_change[changed]
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
