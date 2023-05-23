import msprime
import numpy as np
import tskit
import pandas as pd
from numba import jit
import collections
from dataclasses import dataclass
import tstrait.trait_model as trait_model

@dataclass
class PhenotypeResult:
# Phenotype result class
    individual_id: np.ndarray
    phenotype: np.ndarray
    environment_noise: np.ndarray
    genetic_value: np.ndarray

@dataclass
class GeneticValueResult:
# Genetic result class
    site_id: np.ndarray
    causal_state: np.ndarray
    effect_size: np.ndarray
    allele_frequency: np.ndarray

class SimPhenotype:
# Phenotype simulation class
    def __init__(self, ts, num_causal, h2, model, rng):
        self.ts = ts
        self.num_causal = num_causal
        self.h2 = h2
        self.model = model
        self.rng = rng
        self.sample_index_map = self._obtain_sample_index_map()
        self.site_id_array = self._choose_causal_site()
        
    def _obtain_sample_index_map(self):
        num_nodes = self.ts.num_nodes
        if num_nodes == 0:
            raise ValueError("No nodes in the tree sequence data")
        sample_index_map = np.zeros(num_nodes + 1, dtype=int) - 1
        for j, u in enumerate(self.ts.samples()):
            sample_index_map[u] = j
        return sample_index_map
    
    def _choose_causal_site(self):
        """
        Obtain site ID based on their position (site IDs are aligned based on their positions in tree sequence data requirement)
        """
        num_sites = self.ts.num_sites
        if num_sites == 0:
            raise ValueError("No mutation in the provided data")
        if self.num_causal > num_sites:
            raise ValueError("There are less number of sites in the tree sequence than the inputted number of causal sites")
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
        counts = {x:y for x,y in counts.items() if y!=0}
        if len(counts) > 1:
            del counts[site.ancestral_state]
        return counts
        
    def _obtain_causal_state(self):
        """
        Obtain causal alleles from causal sites, and return the corresponding causal state
        It is aligned based on site ID array
        """ 
        causal_state_array = np.zeros(len(self.site_id_array), dtype=object)
        tree = tskit.Tree(self.ts)
    
        for i, single_id in enumerate(self.site_id_array):
            site = self.ts.site(single_id)
            tree.seek(site.position)
            counts = self._obtain_allele_frequency(tree, site)
            causal_state_array[i] = self.rng.choice(list(counts))
        
        return causal_state_array

    def _site_genotypes(self, tree, site, causal_state):
        """
        Returns a numpy array of 0 and 1, where 1 represents if a node includes the causal mutation or not
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
                
        genotype = np.zeros(self.ts.num_samples)
        while len(stack) > 0:
            u = stack.pop()
            j = self.sample_index_map[u]
            if j > -1:
                genotype[j] = 1
            for v in tree.children(u):
                if not has_mutation[v]:
                    stack.append(v)
        return genotype
                
    def _group_by_individual_genetic_value(self, node_genetic_value):
        """
        Convert genetic value of nodes to be genetic value of individuals
        """
        individual_genetic_value = np.zeros(self.ts.num_individuals)

        for j in range(self.ts.num_samples):
            ind = self.ts.nodes_individual[self.sample_index_map[j]]
            individual_genetic_value[ind] += node_genetic_value[j]
            
        return individual_genetic_value

    def _sim_environment_noise(self, individual_genetic_value):
        """
        Add environmental noise
        G should be values of individuals
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
            env_std = np.sqrt((1-self.h2)/self.h2 * np.var(individual_genetic_value))
            E = self.rng.normal(loc=0.0, scale=env_std, size=num_ind)
            phenotype = individual_genetic_value + E
        
        return phenotype, E
    
    def genetic_value_simulation(self):
        causal_state_array = self._obtain_causal_state()
        tree = tskit.Tree(self.ts)
        node_genetic_value = np.zeros(self.ts.num_samples)
        beta_array = np.zeros(self.num_causal)
        allele_frequency = np.zeros(self.num_causal)

        for i, site_id in enumerate(self.site_id_array):
            site = self.ts.site(site_id)
            tree.seek(site.position)
            site_genotype = self._site_genotypes(tree, site, causal_state_array[i])
            allele_frequency[i] = np.sum(site_genotype) / len(site_genotype)
            beta_array[i] = self.model.sim_effect_size(self.num_causal, allele_frequency[i], self.rng)
            node_genetic_value += site_genotype * beta_array[i]
        
        genotypic_effect_sizes = GeneticValueResult(self.site_id_array, causal_state_array, beta_array, allele_frequency)
        
        return node_genetic_value, genotypic_effect_sizes
    
    def phenotype_environemnt_simulation(self, node_genetic_value):
        individual_genetic_value = self._group_by_individual_genetic_value(node_genetic_value)
        phenotype, E = self._sim_environment_noise(individual_genetic_value)
        phenotype_individuals = PhenotypeResult(list(range(self.ts.num_individuals)), phenotype, E, individual_genetic_value)
        
        return phenotype_individuals

def _sim_phenotype(ts, num_causal, h2, model, rng):
    simulateClass = SimPhenotype(ts, num_causal, h2, model, rng)
    
    node_genetic_value, genotypic_effect_sizes = simulateClass.genetic_value_simulation()
    
    phenotype_individuals = simulateClass.phenotype_environemnt_simulation(node_genetic_value)
    # Ancestral state and location can easily be determined
    
    return phenotype_individuals, genotypic_effect_sizes
    
def sim_phenotype(ts, num_causal, model, h2=0.3, random_seed=None):
    if not isinstance(ts, tskit.TreeSequence):
        raise TypeError("Input should be a tree sequence data")
    try:
        num_causal > 0
    except:
        raise TypeError("Number of causal sites should be a positive integer")  
    if int(num_causal) != num_causal or num_causal <= 0:
        raise ValueError("Number of causal sites should be a positive integer")   
    if not isinstance(model, trait_model.TraitModel):
        raise TypeError("Mutation model must be an instance of TraitModel")
    try:
        h2 > 0
    except:
        raise TypeError("Heritability should be 0 <= h2 <= 1")
    if h2 > 1 or h2 < 0:
        raise ValueError("Heritability should be 0 <= h2 <= 1")
    
    rng = np.random.default_rng(random_seed)
    
    return _sim_phenotype(ts, num_causal, h2, model, rng)
