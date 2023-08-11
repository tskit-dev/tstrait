import numba
import numpy as np
import pandas as pd
import tskit

from .base import _check_instance, _check_dataframe, _check_non_decreasing  # noreorder


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


class GeneticValue:
    """GeneticValue class to compute genetic values of individuals.

    :param ts: Tree sequence data with mutation.
    :type ts: tskit.TreeSequence
    :param trait_df: Pandas dataframe that includes causal site ID, causal allele,
        simulated effect size, and trait ID.
    :type effect_size_df: pandas.DataFrame
    """

    def __init__(self, ts, trait_df):
        self.trait_df = trait_df[["site_id", "causal_state", "effect_size", "trait_id"]]
        self.ts = ts

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

        if len(stack) == 0:
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

    def compute_genetic_value(self):
        """Computes genetic values of individuals.

        :return: Returns a pandas dataframe with genetic value, individual ID,
            and trait ID
        :rtype: pandas.DataFrame
        """

        num_ind = self.ts.num_individuals
        num_trait = np.max(self.trait_df.trait_id) + 1
        genetic_val_array = np.zeros((num_trait, num_ind))

        num_nodes = self.ts.num_nodes
        tree = tskit.Tree(self.ts)

        for data in self.trait_df.itertuples():
            site = self.ts.site(data.site_id)
            tree.seek(site.position)
            individual_genotype = self._individual_genotype(
                tree=tree,
                site=site,
                causal_state=data.causal_state,
                num_nodes=num_nodes,
            )
            genetic_val_array[data.trait_id, :] += (
                individual_genotype * data.effect_size
            )

        df = pd.DataFrame(
            {
                "trait_id": np.repeat(np.arange(num_trait), num_ind),
                "individual_id": np.tile(np.arange(num_ind), num_trait),
                "genetic_value": genetic_val_array.flatten(),
            }
        )

        return df


def genetic_value(ts, trait_df):
    """Calculates genetic values of individuals based on the inputted tree sequence
    and the dataframe with simulated effect sizes of causal mutations.
    :param ts: The tree sequence data that will be used in the quantitative trait
        simulation. The tree sequence data must include a mutation.
    :type ts: tskit.TreeSequence
    :param trait_df: The dataframe that includes simulated effect sizes of causal
        mutations. It must include site_id, causal allele, simulated effect size,
        and trait_id, and the data must be aligned based on site_id.
    :type trait_df: pandas.DataFrame
    :return: Returns a pandas dataframe that includes individual ID and genetic
        value.
    :rtype: pandas.DataFrame
    """

    ts = _check_instance(ts, "ts", tskit.TreeSequence)
    trait_df = _check_dataframe(
        trait_df, ["site_id", "causal_state", "effect_size", "trait_id"], "trait_df"
    )
    _check_non_decreasing(trait_df["site_id"], "site_id")

    trait_id = trait_df["trait_id"].unique()

    if np.min(trait_id) != 0 or np.max(trait_id) != len(trait_id) - 1:
        raise ValueError("trait_id must be consecutive and start from 0")

    genetic = GeneticValue(ts=ts, trait_df=trait_df)

    genetic_df = genetic.compute_genetic_value()

    return genetic_df
