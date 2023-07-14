import numbers

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


class PhenotypeSimulator:
    """Simulator class to simulate quantitative traits of individuals.

    :param ts: Tree sequence data with mutation.
    :type ts: tskit.TreeSequence
    :param effect_size_df: Pandas dataframe that includes causal site ID, causal
        allele, and simulated effect size.
    :type effect_size_df: pandas.DataFrame
    :param h2: Narrow-sense heritability.
    :type h2: float
    :param random_seed: The random seed. If this is not specified or None, simulation
        will be done randomly.
    :type random_seed: None or int
    """

    def __init__(self, ts, effect_size_df, h2, random_seed):
        effect_size_df = effect_size_df.sort_values(by="SiteID")
        self.ts = ts
        self.siteID = effect_size_df["SiteID"]
        self.causal_state = effect_size_df["CausalState"]
        self.effect_size = effect_size_df["EffectSize"]
        self.h2 = h2
        self.rng = np.random.default_rng(random_seed)

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

        This method computes genetic values of individuals based on the simulated
        effect sizes and the corresponding site ID and causal allele.

        :return: Returns a numpy array of genetic values.
        :rtype: numpy.ndarray(float)
        """
        num_nodes = self.ts.num_nodes
        tree = tskit.Tree(self.ts)

        individual_genetic_array = np.zeros(self.ts.num_individuals)

        for i, single_id in enumerate(self.siteID):
            site = self.ts.site(single_id)
            tree.seek(site.position)
            individual_genotype = self._individual_genotype(
                tree=tree,
                site=site,
                causal_state=self.causal_state[i],
                num_nodes=num_nodes,
            )
            individual_genetic_array += individual_genotype * self.effect_size[i]

        return individual_genetic_array

    def _sim_environment_noise(self, individual_genetic_array):
        """Simulates environmental noise based on the genetic values of individuals
        and the narrow-sense heritability. This method also returns the phenotype,
        which is a sum of environmental noise and genetic value.
        """
        num_ind = len(individual_genetic_array)
        if self.h2 == 1:
            E = np.zeros(num_ind)
            phenotype = individual_genetic_array
        else:
            if num_ind > 1:
                env_std = np.sqrt(
                    (1 - self.h2) / self.h2 * np.var(individual_genetic_array)
                )
                E = self.rng.normal(loc=0.0, scale=env_std, size=num_ind)
                phenotype = individual_genetic_array + E
            else:
                E = np.zeros(num_ind)
                phenotype = individual_genetic_array + E

        return phenotype, E

    def sim_environment(self, genetic_value):
        """Simulates environmental noise of individuals and returns the phenotype.

        This method simulates the environmental noise of individuals based on their
        genetic values that are passed into the method. The narrow-sense heritability
        in :class:`PhenotypeSimulator` object will be used to simulate environmental
        noise assuming the additive model.

        The simulated environmental noise and phenotype will be returned by using a
        pandas dataframe, which includes individual ID, phenotype, environmental
        noise and genetic value.

        :param genetic_value: Genetic value of individuals.
        :type genetic_value: numpy.ndarray(float)
        :return: Returns a pandas dataframe object, which includes individual
            ID, phenotype, environmental noise and genetic value.
        :rtype: pandas.DataFrame
        """
        phenotype, E = self._sim_environment_noise(genetic_value)
        individual = np.arange(self.ts.num_individuals, dtype=int)
        phenotype_df = pd.DataFrame(data=[individual, phenotype, E, genetic_value]).T

        phenotype_df = phenotype_df.set_axis(
            ["IndividualID", "Phenotype", "EnvironmentalNoise", "GeneticValue"],
            axis="columns",
        )

        return phenotype_df


def sim_phenotype(ts, effect_size, h2, random_seed=None):
    """Simulates quantitative traits of individuals based on the inputted tree sequence
    and the dataframe with simulated effect sizes of causal mutations.

    :param ts: The tree sequence data that will be used in the quantitative trait
        simulation. The tree sequence data must include a mutation.
    :type ts: tskit.TreeSequence
    :param effect_size: The dataframe that includes simulated effect sizes of causal
        mutations. It must include site ID, causal allele, and simulated effect size.
    :type effect_size: pandas.DataFrame
    :param h2: Narrow-sense heritability, which will be used to simulate environmental
        noise. Narrow-sense heritability must be between 0 and 1.
    :type h2: float
    :param random_seed: The random seed. If this is not specified or None, simulation
        will be done randomly.
    :type random_seed: None or int
    :return: Returns a pandas dataframe that includes individual ID, simulated phenotype,
        simulated environmental noise, and genetic value.
    :rtype: pandas.DataFrame
    """

    if not isinstance(ts, tskit.TreeSequence):
        raise TypeError("Input must be a tree sequence data")
    if not isinstance(effect_size, pd.DataFrame):
        raise TypeError("Effect size input must be a pandas dataframe")
    if "SiteID" not in effect_size.columns:
        raise ValueError("SiteID is not included in the effect size dataframe")
    if "CausalState" not in effect_size.columns:
        raise ValueError("CausalState is not included in the effect size dataframe")
    if "EffectSize" not in effect_size.columns:
        raise ValueError("EffectSize is not included in the effect size dataframe")
    if not isinstance(h2, numbers.Number):
        raise TypeError("Heritability must be a number")
    if h2 > 1 or h2 <= 0:
        raise ValueError("Heritability must be 0 < h2 <= 1")

    simulator = PhenotypeSimulator(
        ts=ts,
        effect_size_df=effect_size,
        h2=h2,
        random_seed=random_seed,
    )
    genetic_value = simulator.compute_genetic_value()
    phenotype_df = simulator.sim_environment(genetic_value)

    return phenotype_df
