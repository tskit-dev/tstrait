import numba
import numpy as np
import pandas as pd
import tskit


@numba.njit
def _traversal_genotype(
    nodes_individual,
    left_child_array,
    right_sib_array,
    stack,
    has_mutation,
    num_individuals,
    num_nodes,
):
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
        trait_df = trait_df.sort_values(by="site_id")
        self.ts = ts
        self.num_trait = len(trait_df.trait_id.unique())
        self.trait_df = trait_df

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

    def compute_genetic_value_single(self, df):
        """Computes genetic values of individuals for a single trait.

        This method computes genetic values of individuals based on the simulated
        effect sizes and the corresponding site ID and causal allele.

        :return: Returns a numpy array of genetic values.
        :rtype: numpy.ndarray(float)
        """
        causal_site_array = df["site_id"]
        causal_state_array = df["causal_state"]
        beta_array = df["effect_size"]

        num_nodes = self.ts.num_nodes
        tree = tskit.Tree(self.ts)

        individual_genetic_array = np.zeros(self.ts.num_individuals)

        for i, single_id in enumerate(causal_site_array):
            site = self.ts.site(single_id)
            tree.seek(site.position)
            individual_genotype = self._individual_genotype(
                tree=tree,
                site=site,
                causal_state=causal_state_array.iloc[i],
                num_nodes=num_nodes,
            )
            individual_genetic_array += individual_genotype * beta_array.iloc[i]

        return individual_genetic_array

    def compute_genetic_value(self):
        """Computes genetic values of individuals.

        This method computes genetic values of individuals based on the simulated
        effect sizes and the corresponding site ID and causal allele.

        :return: Returns a numpy array of genetic values.
        :rtype: numpy.ndarray(float)
        """
        num_ind = self.ts.num_individuals
        trait_id_array = self.trait_df.trait_id.unique()
        df = pd.DataFrame(columns=["individual_id", "genetic_value", "trait_id"])
        for trait_id in trait_id_array:
            genetic_value = self.compute_genetic_value_single(
                self.trait_df[self.trait_df.trait_id == trait_id]
            )
            df_add = pd.DataFrame(
                {
                    "individual_id": np.arange(num_ind),
                    "genetic_value": genetic_value,
                    "trait_id": np.ones(num_ind) * trait_id,
                }
            )
            df = pd.concat([df, df_add])
        df = df.sort_values(by=["trait_id", "individual_id"])
        df = df.reset_index()
        del df["index"]

        return df


def calculate_genetic_value(ts, trait_df):
    """Calculates genetic values of individuals based on the inputted tree sequence
    and the dataframe with simulated effect sizes of causal mutations.

    :param ts: The tree sequence data that will be used in the quantitative trait
        simulation. The tree sequence data must include a mutation.
    :type ts: tskit.TreeSequence
    :param trait_df: The dataframe that includes simulated effect sizes of causal
        mutations. It must include site ID, causal allele, simulated effect size,
        and trait_id.
    :type trait_df: pandas.DataFrame
    :return: Returns a pandas dataframe that includes individual ID and genetic
        value.
    :rtype: pandas.DataFrame
    """
    if not isinstance(ts, tskit.TreeSequence):
        raise TypeError("Input must be a tree sequence data")
    if not isinstance(trait_df, pd.DataFrame):
        raise TypeError("trait_df must be a pandas dataframe")
    if "site_id" not in trait_df.columns:
        raise ValueError("site_id is not included in the trait dataframe")
    if "causal_state" not in trait_df.columns:
        raise ValueError("causal_state is not included in the trait dataframe")
    if "effect_size" not in trait_df.columns:
        raise ValueError("effect_size is not included in the trait dataframe")
    if "trait_id" not in trait_df.columns:
        raise ValueError("trait_id is not included in the trait dataframe")

    genetic = GeneticValue(
        ts=ts,
        trait_df=trait_df,
    )

    df = genetic.compute_genetic_value()

    return df
