import numbers
import numpy as np
import tskit
import numba
import collections
from dataclasses import dataclass
import tstrait.trait_model as trait_model


@dataclass
class PhenotypeResult:
    """Data class that contains simulated phenotypic information of individuals.

    For each individual in the tree sequence data, this data class object returns the
    simulated value of quantitative traits, environmental noise and genetic value, which
    are aligned based on individual IDs. See the :ref:`sec_output_phenotype` section for
    more details on the output of this object.

    :param individual_id: Individual IDs
    :type individual_id: numpy.ndarray(int)
    :param phenotype: Simulated quantitative trait
    :type phenotype: numpy.ndarray(float)
    :param environment_noise: Simulated environmental noise
    :type environment_noise: numpy.ndarray(float)
    :param genetic_value: Simulated genetic values
    :type genetic_value: numpy.ndarray(float)
    """

    individual_id: np.ndarray
    phenotype: np.ndarray
    environment_noise: np.ndarray
    genetic_value: np.ndarray

    def __str__(self):
        output = (
            f"individual_id: {self.individual_id}"
            f"\nphenotype: {self.phenotype}"
            f"\nenvironmental_noise: {self.environment_noise}"
            f"\ngenetic_value: {self.genetic_value}"
        )
        return output


@dataclass
class GenotypeResult:
    """Data class that contains simulated genotypic information.

    For each randomly chosen causal site, this data class object returns causal allele,
    frequency of the causal allele, and simulated effect size, which are aligned based
    on site IDs. See the :ref:`sec_output_genetic` section for more details on the
    output of this object.

    :param site_id: Causal site IDs
    :type site_id: numpy.ndarray(int)
    :param causal_allele: Causal allele
    :type causal_allele: numpy.ndarray(object)
    :param effect_size: Effect size
    :type effect_size: numpy.ndarray(float)
    :param allele_frequency: Frequency of causal allele
    :type allele_frequency: numpy.ndarray(float)
    """

    site_id: np.ndarray
    causal_allele: np.ndarray
    effect_size: np.ndarray
    allele_frequency: np.ndarray

    def __str__(self):
        output = (
            f"site_id: {self.site_id}"
            f"\ncausal_allele: {self.causal_allele}"
            f"\neffect_size: {self.effect_size}"
            f"\nallele_frequency: {self.allele_frequency}"
        )
        return output    


@dataclass
class Result:
    """Data class that contains the simulated result. See the ...

    :param phenotype: A :class:`PhenotypeResult` object that contains simulated
        phenotypic information of individuals.
    :type phenotype: PhenotypeResult
    :param genotype: A :class:`GenotypeResult` object that contains simulated genotypic
        information of causal sites.
    :type genotype: GenotypeResult
    """

    phenotype: PhenotypeResult
    genotype: GenotypeResult


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

    :param ts: Tree sequence data with mutation
    :type ts: tskit.TreeSequence
    :param num_causal: Number of causal sites
    :type num_causal: int
    :param h2: Narrow-sense heritability
    :type h2: float
    :param model: Trait model
    :type model: TraitModel
    """

    def __init__(self, ts, num_causal, h2, model, random_seed):
        self.ts = ts
        self.num_causal = num_causal
        self.h2 = h2
        self.model = model
        self.rng = np.random.default_rng(random_seed)

    def _choose_causal_site(self):
        """
        Obtain site ID based on their position (site IDs are aligned
        based on their positions in tree sequence data requirement).
        """
        site_id = self.rng.choice(
            range(self.ts.num_sites), size=self.num_causal, replace=False
        )
        site_id = np.sort(site_id)

        return site_id

    def _obtain_allele_frequency(self, tree, site):
        """
        Obtain a dictionary of allele frequency counts, excluding the ancestral state
        Input is the tree sequence site (ts.site(ID)).
        Remove sites from dictionary having no items.
        If only the ancestral state exists, don't delete the ancestral state.
        """
        counts = collections.Counter({site.ancestral_state: self.ts.num_samples})
        for m in site.mutations:
            current_state = site.ancestral_state
            if m.parent != tskit.NULL:
                current_state = self.ts.mutation(m.parent).derived_state
            # Silent mutations do nothing
            if current_state != m.derived_state:
                num_samples = tree.num_samples(m.node)
                counts[m.derived_state] += num_samples
                counts[current_state] -= num_samples
        del counts[site.ancestral_state]
        counts = {x: y for x, y in counts.items() if y != 0}
        if len(counts) == 0:
            counts = {site.ancestral_state: self.ts.num_samples}
        return counts

    def _individual_genotype(self, tree, site, causal_state, num_nodes):
        """
        Returns a numpy array that describes the number of causal mutation in an individual.
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

    def sim_genetic_value(self):
        """Simulates genetic values of individuals.

        This method randomly chooses causal sites and the corresponding causal state
        based on the `num_causal` input. Afterwards, effect size of each causal site
        is simulated based on the trait model given by the `model` input. Genetic
        values are computed by using the simulated effect sizes and mutation
        information of individuals.

        :return: Returns a :class:`Genotype` object that includes simulated
            genetic information of each causal site, and a numpy array of simulated
            genetic values.
        :rtype: (GenotypeResult, numpy.ndarray(float))
        """
        causal_site_array = self._choose_causal_site()
        num_nodes = self.ts.num_nodes
        tree = tskit.Tree(self.ts)

        individual_genetic_array = np.zeros(self.ts.num_individuals)
        causal_state_array = np.zeros(self.num_causal, dtype=object)
        beta_array = np.zeros(self.num_causal)
        allele_frequency = np.zeros(self.num_causal)

        for i, single_id in enumerate(causal_site_array):
            site = self.ts.site(single_id)
            tree.seek(site.position)
            counts = self._obtain_allele_frequency(tree, site)
            causal_state_array[i] = self.rng.choice(list(counts))
            individual_genotype = self._individual_genotype(
                tree=tree,
                site=site,
                causal_state=causal_state_array[i],
                num_nodes=num_nodes,
            )
            allele_frequency[i] = np.sum(individual_genotype) / (
                2 * len(individual_genotype)
            )
            beta_array[i] = self.model.sim_effect_size(
                self.num_causal, allele_frequency[i], self.rng
            )
            individual_genetic_array += individual_genotype * beta_array[i]

        genotypic_effect_data = GenotypeResult(
            site_id=causal_site_array,
            causal_allele=causal_state_array,
            effect_size=beta_array,
            allele_frequency=allele_frequency,
        )

        return genotypic_effect_data, individual_genetic_array

    def _sim_environment_noise(self, individual_genetic_array):
        """
        Add environmental noise to the genetic value of individuals given the genetic value
        of individuals. The simulation assumes the additive model.
        """
        trait_sd = self.model.trait_sd
        num_ind = len(individual_genetic_array)
        if self.h2 == 1:
            E = np.zeros(num_ind)
            phenotype = individual_genetic_array
        elif self.h2 == 0:
            E = self.rng.normal(loc=0.0, scale=trait_sd, size=num_ind)
            phenotype = E
        else:
            env_std = np.sqrt(
                (1 - self.h2) / self.h2 * np.var(individual_genetic_array)
            )
            E = self.rng.normal(loc=0.0, scale=env_std, size=num_ind)
            phenotype = individual_genetic_array + E

        return phenotype, E

    def sim_environment(self, individual_genetic_value):
        """Simulates environmental noise of individuals and returns the phenotype.

        This method simulates the environmental noise of individuals based on their
        genetic values that are passed into the method. The narrow-sense heritability
        in :class:`PhenotypeSimulator` object will be used to simulate environmental
        noise assuming the additive model.

        The simulated environmental noise and phenotype will be returned by using the
        :class:`PhenotypeResult` object, which includes individual ID, phenotype,
        environmental noise and genetic value.

        :param individual_genetic_value: Genetic value of individuals.
        :type individual_genetic_value: numpy.ndarray(float)
        :return: Returns the :class:`PhenotypeResult` object, which includes individual
            ID, phenotype, environmental noise and genetic value.
        :rtype: PhenotypeResult
        """
        phenotype, E = self._sim_environment_noise(individual_genetic_value)
        phenotype_individuals = PhenotypeResult(
            individual_id=np.arange(self.ts.num_individuals),
            phenotype=phenotype,
            environment_noise=E,
            genetic_value=individual_genetic_value,
        )

        return phenotype_individuals


def sim_phenotype(ts, num_causal, model, h2=0.3, random_seed=None):
    """Simulates quantitative traits of individuals based on the inputted tree sequence
    and the specified trait model, and returns a :class:`Result` object. See the
    :ref:`sec_output` section for more details on the output of the simulation model.

    :param ts: The tree sequence data that will be used in the quantitative trait simulation.
        The tree sequence data must include a mutation.
    :type ts: tskit.TreeSequence
    :param num_causal: Number of causal sites that will be chosen randomly. It should be a
        positive integer that is greater than the number of sites in the tree sequence data.
    :type num_causal: int
    :param model: Trait model from :class:`TraitModel` class, which will be used to simulate
        effect sizes of causal sites. See the :ref:`sec_trait_model` section for more details
        on the available models and examples.
    :type model: TraitModel
    :param h2: Narrow-sense heritability, which will be used to simulate environmental noise.
        Narrow-sense heritability must be between 0 and 1.
    :type h2: float
    :param random_seed: The random seed. If this is not specified or None, simulation will
        be done randomly.
    :type random_seed: None or int
    :return: Returns the :class:`Result` object that includes the simulated information
        obtained from `tstrait`. The :class:`Result` object includes :class:`PhenotypeResult`
        object to describe the simulated information regarding the individuals, and the
        :class:`GenotypeResult` object to describe the simulated information regarding the
        causal sites.
    :rtype: Result
    """

    if not isinstance(ts, tskit.TreeSequence):
        raise TypeError("Input should be a tree sequence data")
    if not isinstance(num_causal, numbers.Number):
        raise TypeError("Number of causal sites should be an integer")
    if int(num_causal) != num_causal or num_causal <= 0:
        raise ValueError("Number of causal sites should be a positive integer")
    if not isinstance(model, trait_model.TraitModel):
        raise TypeError("Trait model must be an instance of TraitModel")
    if not isinstance(h2, numbers.Number):
        raise TypeError("Heritability should be a number")
    if h2 > 1 or h2 < 0:
        raise ValueError("Heritability should be 0 <= h2 <= 1")
    num_sites = ts.num_sites
    if num_sites == 0:
        raise ValueError("No mutation in the provided data")
    if num_causal > num_sites:
        raise ValueError(
            "There are less number of sites in the tree sequence than the inputted number"
            "of causal sites"
        )

    individual_node = ts.nodes_individual[ts.nodes_individual > -1]
    individual_node_count = np.bincount(individual_node)
    if len(individual_node_count) != ts.num_individuals:
        raise ValueError("All samples must be associated with an individual")
    if np.any(individual_node_count == 0):
        raise ValueError("All individuals must be associated with sample nodes")

    simulator = PhenotypeSimulator(
        ts=ts, num_causal=num_causal, h2=h2, model=model, random_seed=random_seed
    )
    genotypic_effect_data, individual_genetic_array = simulator.sim_genetic_value()
    phenotype_data = simulator.sim_environment(individual_genetic_array)
    sim_result = Result(phenotype=phenotype_data, genotype=genotypic_effect_data)
    return sim_result
