import collections
from dataclasses import dataclass

import numba
import numpy as np
import pandas as pd
import tskit

from .base import _check_instance, _check_dataframe, _check_non_decreasing  # noreorder


@dataclass
class GeneticResult:
    """
    Data class that contains effect size and genetic value dataframe.

    Attributes
    ----------
    effect_size : pandas.DataFrame
        Dataframe that includes simulated effect sizes.
    genetic : pandas.DataFrame
        Dataframe that includes simulated genetic values.

    See Also
    --------
    sim_genetic : Use this dataclass as a simulation output.

    Examples
    --------
    See :ref:`effect_size_genetic` for details on extracting the effect size
    dataframe, and :ref:`genetic_value_output` for details on extracting the
    genetic value dataframe.
    """

    effect_size: pd.DataFrame
    genetic: pd.DataFrame


@numba.njit
def _traversal_genotype(
    nodes_individual,
    left_child_array,
    right_sib_array,
    stack,
    has_mutation,
    num_individuals,
    num_nodes,
):  # pragma: no cover
    """
    Numba to speed up the tree traversal algorithm to determine the genotype of
    individuals.
    Stack has to be Typed List in numba to use numba.
    """

    genotype = np.zeros(num_individuals)
    while len(stack) > 0:
        parent_node_id = stack.pop()
        if parent_node_id == num_nodes:
            individual_id = -1
        else:
            individual_id = nodes_individual[parent_node_id]
        if individual_id > -1:
            genotype[individual_id] += 1
        child_node_id = left_child_array[parent_node_id]
        while child_node_id != -1:
            if not has_mutation[child_node_id]:
                stack.append(child_node_id)
            child_node_id = right_sib_array[child_node_id]

    return genotype


class _GeneticValue:
    """GeneticValue class to compute genetic values of individuals.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Tree sequence data with mutation
    trait_df : pandas.DataFrame
        Dataframe that includes causal site ID, causal allele, simulated effect
        size, and trait ID.
    alpha : float
        Parameter that determines the relative weight on rarer variants.
    """

    def __init__(self, ts, trait_df, alpha, random_seed):
        self.trait_df = trait_df[["site_id", "effect_size", "trait_id"]]
        self.ts = ts
        self.alpha = alpha
        self.rng = np.random.default_rng(random_seed)

    def _obtain_allele_count(self, tree, site):
        """Obtain a dictionary of allele counts, and the ancestral state is not
        included in the dictionary. Input is the tree sequence site (`ts.site(ID)`)
        instead of site ID, as obtaining `ts.site(ID)` can be time consuming. The
        ancestral state is not deleted if the ancestral state is the only allele
        at that site.
        """
        counts = collections.Counter({site.ancestral_state: self.ts.num_samples})
        for m in site.mutations:
            current_state = site.ancestral_state
            if m.parent != tskit.NULL:
                current_state = self.ts.mutation(m.parent).derived_state
            # Silent mutations do nothing
            if current_state != m.derived_state:  # pragma: no cover
                num_samples = tree.num_samples(m.node)
                counts[m.derived_state] += num_samples
                counts[current_state] -= num_samples
        del counts[site.ancestral_state]
        counts = {x: y for x, y in counts.items() if y != 0}
        if len(counts) == 0:
            counts = {site.ancestral_state: self.ts.num_samples}
        return counts

    def _frequency_dependence(self, allele_freq):
        if allele_freq == 0 or allele_freq == 1:
            const = 0
        else:
            const = np.sqrt(pow(2 * allele_freq * (1 - allele_freq), self.alpha))
        return const

    def _individual_genotype(self, tree, site, causal_state, num_nodes):
        """
        Returns a numpy array that describes the number of causal mutation in an
        individual.
        """
        has_mutation = np.zeros(num_nodes + 1, dtype=bool)
        state_transitions = {tree.virtual_root: site.ancestral_state}
        for m in site.mutations:
            state_transitions[m.node] = m.derived_state
            has_mutation[m.node] = True
        stack = numba.typed.List()
        for node, state in state_transitions.items():
            if state == causal_state:
                stack.append(node)

        if len(stack) == 0:  # pragma: no cover
            genotype = np.zeros(self.ts.num_individuals)
        else:
            genotype = _traversal_genotype(
                nodes_individual=self.ts.nodes_individual,
                left_child_array=tree.left_child_array,
                right_sib_array=tree.right_sib_array,
                stack=stack,
                has_mutation=has_mutation,
                num_individuals=self.ts.num_individuals,
                num_nodes=num_nodes,
            )

        return genotype

    def _compute_genetic_value(self):
        """Computes genetic values of individuals.

        Returns
        -------
        pandas.DataFrame
            Dataframe with simulated genetic value, individual ID, and trait ID.
        """

        num_ind = self.ts.num_individuals
        num_samples = self.ts.num_samples
        num_trait = np.max(self.trait_df.trait_id) + 1
        genetic_val_array = np.zeros((num_trait, num_ind))
        causal_state_array = np.zeros(len(self.trait_df), dtype=object)
        freq_dep = np.zeros(len(self.trait_df))
        allele_freq_array = np.zeros(len(self.trait_df))
        num_nodes = self.ts.num_nodes
        tree = tskit.Tree(self.ts)

        for i, data in enumerate(self.trait_df.itertuples()):
            site = self.ts.site(data.site_id)
            tree.seek(site.position)
            counts = self._obtain_allele_count(tree, site)
            causal_state = self.rng.choice(list(counts))
            causal_state_array[i] = causal_state
            allele_freq = counts[causal_state] / num_samples
            freq_dep[i] = self._frequency_dependence(allele_freq)
            allele_freq_array[i] = allele_freq

            individual_genotype = self._individual_genotype(
                tree=tree,
                site=site,
                causal_state=causal_state,
                num_nodes=num_nodes,
            )
            genetic_val_array[data.trait_id, :] += (
                individual_genotype * data.effect_size * freq_dep[i]
            )

        df = pd.DataFrame(
            {
                "trait_id": np.repeat(np.arange(num_trait), num_ind),
                "individual_id": np.tile(np.arange(num_ind), num_trait),
                "genetic_value": genetic_val_array.flatten(),
            }
        )

        self.trait_df["effect_size"] = np.multiply(self.trait_df.effect_size, freq_dep)
        self.trait_df["causal_state"] = causal_state_array
        self.trait_df["allele_frequency"] = allele_freq_array

        genetic_result = GeneticResult(effect_size=self.trait_df, genetic=df)

        return genetic_result


def sim_genetic(ts, trait_df, alpha=0, random_seed=None):
    """
    Simulates genetic values from a trait dataframe.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence data that will be used in the quantitative trait
        simulation.
    trait_df : pandas.DataFrame
        Trait dataframe.
    alpha : float, default 0
        Parameter that determines the degree of the frequency dependence model. Please
        see :ref:`frequency_dependence` for details on how this parameter influences
        effect size simulation.
    random_seed : int, default None
        Random seed of simulation. If None, simulation will be conducted randomly.

    Returns
    -------
    GeneticResult
        Dataclass object that includes effect size and genetic value dataframe.

    See Also
    --------
    trait_model : Return a trait model, which can be used as `model` input.
    sim_trait : Return a trait dataframe, whch can be used as a `trait_df` input.
    GeneticResult : Dataclass object that will be used as an output.
    sim_env : Genetic value dataframe output can be used as an input to simulate
        environmental noise.

    Notes
    -----
    The `trait_df` input has some requirements that will be noted below.

    1. Columns

    The following columns must be included in `trait_df`:

        * **site_id**: Site IDs that have causal mutation.
        * **effect_size**: Simulated effect size of causal mutation.
        * **trait_id**: Trait ID.

    2. Data requirements

        * Site IDs in **site_id** column must be sorted in an ascending order. Please
          refer to :py:meth:`pandas.DataFrame.sort_values` for details on sorting
          values in a :class:`pandas.DataFrame`.

        * Trait IDs in **trait_id** column must start from zero and be consecutive.

    The simulation outputs of effect sizes and phenotypes are given as a
    :py:class:`pandas.DataFrame`.

    The effect size dataframe can be extracted by using ``.effect_size`` in the
    resulting object and contains the following columns:

        * **site_id**: ID of sites that have causal mutation.
        * **effect_size**: Simulated effect size of causal mutation.
        * **trait_id**: Trait ID.
        * **causal_state**: Causal state.
        * **allele_frequency**: Allele frequency of causal mutation.

    The genetic value dataframe can be extracted by using ``.genetic`` in the resulting
    object and contains the following columns:

        * **trait_id**: Trait ID.
        * **individual_id**: Individual ID inside the tree sequence input.
        * **genetic_value**: Simulated genetic values.

    Examples
    --------
    See :ref:`genetic_value` for worked examples.
    """

    ts = _check_instance(ts, "ts", tskit.TreeSequence)
    trait_df = _check_dataframe(
        trait_df, ["site_id", "effect_size", "trait_id"], "trait_df"
    )
    _check_non_decreasing(trait_df["site_id"], "site_id")

    trait_id = trait_df["trait_id"].unique()

    if np.min(trait_id) != 0 or np.max(trait_id) != len(trait_id) - 1:
        raise ValueError("trait_id must be consecutive and start from 0")

    genetic = _GeneticValue(
        ts=ts, trait_df=trait_df, alpha=alpha, random_seed=random_seed
    )

    genetic_result = genetic._compute_genetic_value()

    return genetic_result
