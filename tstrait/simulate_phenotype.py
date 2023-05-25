import msprime
import numbers
import numpy as np
import tskit
import pandas as pd
from numba import jit
import collections
from dataclasses import dataclass
import tstrait.trait_model as trait_model


@dataclass
class PhenotypeResult:
    """Data class that contains simulated phenotypic information of individuals
    
    For each individual in the tree sequence data, this data class oject returns the
    simulated value of phenotype, environmental noise and genetic value, which are
    aligned based on the individual IDs.
    
    :param individual_id: Numpy array of individual IDs
    :type individual_id: np.ndarray(int)
    :param phenotype: Simulated phenotypes of individuals
    :type phenotype: np.ndarray(float)
    :param environment_noise: Simulated environmental noise of individuals
    :type environment_noise: np.ndarray(float)
    :param genetic_value: Simulated genetic value of individuals
    :type genetic_value: np.ndarray(float)
    """
    # Phenotype result class
    individual_id: np.ndarray
    phenotype: np.ndarray
    environment_noise: np.ndarray
    genetic_value: np.ndarray


@dataclass
class GeneticValueResult:
    """Data class that contains simulated genetic information
    
    For each randomly chosen causal site inside the simulation model, this data class
    object returns the randomly chosen causal state, the allele frequency of the causal
    state, and the simulated value of effect size, which are aligned based on the
    site IDs.
    
    :param site_id: Numpy array of site IDs that were randomly chosen to be causal
    :type site_id: np.ndarray(int)
    :param causal_state: Numpy array of causal state inside each causal site
    :type causal_state: np.ndarray(object)
    :param effect_size: Effect size of causal mutation
    :type effect_size: np.ndarray(float)
    :param allele_frequency: Frequency of causal mutation
    :type allele_frequency: np.ndarray(float)
    """
    site_id: np.ndarray
    causal_state: np.ndarray
    effect_size: np.ndarray
    allele_frequency: np.ndarray


class PhenotypeSimulator:
    """Simulator class of phenotypes
    
    :param ts: Tree sequence data with mutation
    :type ts: tskit.TreeSequence
    :param num_causal: Number of causal sites associated with a trait
    :type num_causal: int
    :param h2: Narrow-sense heritability of a trait
    :type h2: float
    :param model: Mutation model in simulation
    :type model: class `tstrait.TraitModel`
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
        based on their positions in tree sequence data requirement)
        """
        num_sites = self.ts.num_sites
        if num_sites == 0:
            raise ValueError("No mutation in the provided data")
        if self.num_causal > num_sites:
            raise ValueError(
                "There are less number of sites in the tree sequence than the inputted number of causal sites"
            )
        site_id = self.rng.choice(range(num_sites), size=self.num_causal, replace=False)
        site_id = np.sort(site_id)

        return site_id

    def _obtain_allele_frequency(self, tree, site):
        """
        Obtain a dictionary of allele frequency counts, excluding the ancestral state
        Input is the tree sequence site (ts.site(ID))
        Remove sites from dictionary having no items
        If only the ancestral state exists, don't delete the ancestral state
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
        counts = {x: y for x, y in counts.items() if y != 0}
        if len(counts) > 1:
            del counts[site.ancestral_state]
        return counts

    def _individual_genotype(self, tree, site, causal_state):
        """
        Returns a numpy array that describes the number of causal mutation in an individual
        """
        has_mutation = np.zeros(self.ts.num_nodes + 1, dtype=bool)
        state_transitions = {tree.virtual_root: site.ancestral_state}
        for m in site.mutations:
            state_transitions[m.node] = m.derived_state
            has_mutation[m.node] = True
        stack = []
        for node, state in state_transitions.items():
            if state == causal_state:
                stack.append(node)

        genotype = np.zeros(self.ts.num_individuals)
        while len(stack) > 0:
            u = stack.pop()
            j = self.ts.nodes_individual[u]
            if j > -1:
                genotype[j] += 1
            for v in tree.children(u):
                if not has_mutation[v]:
                    stack.append(v)
        return genotype

    def sim_genetic_value(self):
        """Simulates the genetic value of individuals
        
        This method randomly chooses the causal sites and the corresponding causal state based
        on the `num_causal` input. Afterwards, this computes the allele frequency of causal state,
        and simulates the effect size of each causal mutation based on the mutation model given by
        the `model` input.
        
        The genetic value of individuals are computed by using the simulated effect size of
        causal mutation.
        
        The results of this method are returned through the dataclass `tstrait.GeneticValueResult`
        object and includes simulated results of each causal site. The method also returns the
        simulated genetic value of indviduals, which will be used to simulate the phenotypes.

        :return: Returns class `tstrait.GeneticValueResult` object to indicate the simulated
            results of each causal site, and the simulated genetic value of individuals which
            are aligned by individual IDs
        :rtype: (class `tstrait.GeneticValueResult`, np.ndarray)
        """
        tree = tskit.Tree(self.ts)
        causal_site_array = self._choose_causal_site()

        individual_genetic_value = np.zeros(self.ts.num_individuals)
        causal_state_array = np.zeros(self.num_causal, dtype=object)
        beta_array = np.zeros(self.num_causal)
        allele_frequency = np.zeros(self.num_causal)

        for i, single_id in enumerate(causal_site_array):
            site = self.ts.site(single_id)
            tree.seek(site.position)
            counts = self._obtain_allele_frequency(tree, site)
            causal_state_array[i] = self.rng.choice(list(counts))
            individual_genotype = self._individual_genotype(tree = tree, site = site,
                                                            causal_state = causal_state_array[i])
            allele_frequency[i] = np.sum(individual_genotype) / (2 * len(individual_genotype))
            beta_array[i] = self.model.sim_effect_size(
                self.num_causal, allele_frequency[i], self.rng
            )
            individual_genetic_value += individual_genotype * beta_array[i]

        genotypic_effect_sizes = GeneticValueResult(
            site_id = causal_site_array, causal_state = causal_state_array,
            effect_size = beta_array, allele_frequency= allele_frequency
        )

        return genotypic_effect_sizes, individual_genetic_value

    def _sim_environment_noise(self, individual_genetic_value):
        """
        Add environmental noise to the genetic value of individuals given the genetic value
        of individuals. The simulation assumes the additive model.
        """
        trait_sd = self.model.trait_sd
        num_ind = len(individual_genetic_value)
        if self.h2 == 1:
            E = np.zeros(num_ind)
            phenotype = individual_genetic_value
        elif self.h2 == 0:
            E = self.rng.normal(loc=0.0, scale=trait_sd, size=num_ind)
            phenotype = E
        else:
            env_std = np.sqrt(
                (1 - self.h2) / self.h2 * np.var(individual_genetic_value)
            )
            E = self.rng.normal(loc=0.0, scale=env_std, size=num_ind)
            phenotype = individual_genetic_value + E

        return phenotype, E

    def sim_environment(self, individual_genetic_value):
        """Simulates environmental noise of individuals and return the phenotype
        
        This method simulates the environmental noise of individuals based on their
        genetic values. The inputted narrow-sense heritability will be used to simulate
        environmental noise assuming the additive model.
        
        The simulated environmental noise and phenotype will be returned by using the
        dataclass `tstrait.PhenotypeResult` object, which includes individual IDs,
        phenotype, environmental noise and genetic value.

        :param individual_genetic_value: Genetic value of individuals
        :type individual_genetic_value: np.ndarray(float)
        :return: Returns the class `tstrait.PhenotypeResult` object to describe the simulated
        phenotypes
        :rtype: class `tstrait.PhenotypeResult`
        """
        phenotype, E = self._sim_environment_noise(individual_genetic_value)
        phenotype_individuals = PhenotypeResult(
            individual_id = np.arange(self.ts.num_individuals), phenotype = phenotype,
            environment_noise = E, genetic_value = individual_genetic_value
        )

        return phenotype_individuals


def sim_phenotype(ts, num_causal, model, h2=0.3, random_seed=None):
    """Simulates phenotypes of individuals based on the inputted tree sequence data with
    mutation
    
    The input of the function will be number of causal sites, narrow-sense heritability,
    and the model from class `tstrait.TraitModel`. Model will be used to simulate the
    effect sizes of randomly chosen causal mutation, and environmental noise will be
    simulated by using the narrow-sense heritability. The random seed will be used to
    produce a `np.random.Generator` object to conduct the simulation.
    
    The simulated phenotypes of individuals will be returned by using the dataclass
    `tstrait.PhenotypeResult` object, and it includes the simulated value of phenotype,
    genetic value and environmental noise, which are aligned based on the individual IDs.
    Since we are considering the additive model in simulation, the phenotype value will be
    a sum of genetic value and environmental noise.
    
    The simulated genetic information will be returned by using the dataclass
    `tstrait.GeneticValueResult` object, and it includes the randomly chosen causal state,
    the allele frequency of the causal state, and the simulated value of effect size, which
    are aligned based on the site IDs.

    :param ts: Tree sequence data with mutation
    :type ts: tskit.TreeSequence
    :param num_causal: Number of causal sites associated with a trait
    :type num_causal: int or array_like(int)[ints]
    :param model: Mutation model in simulation
    :type model: class `tstrait.TraitModel`
    :param h2: Narrow-sense heritability of a trait
    :type h2: float or array_like(float)[ints]
    :param random_seed: The random seed value used to generate the `np.random.Generator`
        object
    :type random_seed: None or int or array_like(int or None)[ints] or SeedSequence or
        BitGenerator or Generator
    :return: Returns the class `tstrait.PhenotypeResult` object to describe the simulated
        phenotypes and the class `tstrait.GeneticValueResult` object to describe the
        simulated genetic information
    :rtype: (class `tstrait.PhenotypeResult`, class `tstrait.GeneticValueResult`)
    """
    
    if not isinstance(ts, tskit.TreeSequence):
        raise TypeError("Input should be a tree sequence data")
    if not isinstance(num_causal, numbers.Number):
        raise TypeError("Number of causal sites should be an integer")
    if int(num_causal) != num_causal or num_causal <= 0:
        raise ValueError("Number of causal sites should be a positive integer")
    if not isinstance(model, trait_model.TraitModel):
        raise TypeError("Mutation model must be an instance of TraitModel")
    if not isinstance(h2, numbers.Number):
        raise TypeError("Heritability should be a number")
    if h2 > 1 or h2 < 0:
        raise ValueError("Heritability should be 0 <= h2 <= 1")
    
    simulator = PhenotypeSimulator(ts = ts, num_causal = num_causal, h2 = h2,
                                   model = model, random_seed = random_seed)
    genotypic_effect_sizes, individual_genetic_value = simulator.sim_genetic_value()
    phenotype_individuals = simulator.sim_environment(individual_genetic_value)
    return phenotype_individuals, genotypic_effect_sizes