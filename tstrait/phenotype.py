import msprime
import numpy as np
import tskit
import pandas as pd
from numba import jit
import collections
from dataclasses import dataclass
import tstrait.trait as trait

@dataclass
class PhenotypeResult:
# Phenotype result class
    individual_id: np.ndarray
    phenotype: np.ndarray
    environment: np.ndarray
    genetic: np.ndarray

@dataclass
class GeneticResult:
# Genetic result class
    site_id: np.ndarray
    causal_state: np.ndarray
    effect_size: np.ndarray
    frequency: np.ndarray
    
@dataclass
class SiteGenetic:
# Genetic information of a site
    beta: float
    allele_freq: float
    site_genetic: np.ndarray
    

class SimPhenotype:
# Phenotype simulation class
    def __init__(self, ts, num_causal, h2, model, rng):
        self.ts = ts
        self.num_causal = num_causal
        self.h2 = h2
        self.model = model
        self.rng = rng
        self.sample_index_map = self._obtain_sample_index_map()
        self.site_id_list = self._choose_causal()
        
    def _obtain_sample_index_map(self):
        num_nodes = self.ts.num_nodes
        if num_nodes == 0:
            raise ValueError("No nodes in the tree sequence data")
        sample_index_map = np.zeros(num_nodes + 1, dtype=int) - 1
        for j, u in enumerate(self.ts.samples()):
            sample_index_map[u] = j
        return sample_index_map
    
    def _choose_causal(self):
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
        
    def _count_site_alleles(self, tree, site):
        """
        Obtain a dictionary of allele frequency counts, excluding the ancestral state
        Input is the tree sequence site (ts.site(ID))
        If only state exists, don't delete the ancestral state
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
        if len(counts) > 1:
            del counts[site.ancestral_state]
        return counts
        
    def _obtain_causal_state(self):
        """
        Obtain causal alleles from causal sites, and return the site ID and the corresponding causal state
        They are aligned based on their genomic positions
        """ 
        causal_state_list = np.zeros(len(self.site_id_list), dtype=object)
        tree = tskit.Tree(self.ts)
    
        for i, single_id in enumerate(self.site_id_list):
            site = self.ts.site(single_id)
            tree.seek(site.position)
            counts = self._count_site_alleles(tree, site)
            causal_state_list[i] = self.rng.choice(list(counts))
        
        return causal_state_list

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
            if j > 0:
                genotype[j] = 1
            for v in tree.children(u):
                if not has_mutation[v]:
                    stack.append(v)
        return genotype
        

    def _effect_size_simulation_site(self, site_genotype):
        """
        Simulate effect size of sample nodes in a single site
        """ 
        allele_freq = np.sum(site_genotype) / len(site_genotype)
        beta = self.model.sim_effect_size(self.num_causal, allele_freq, self.rng)
        site_genetic = site_genotype * beta
        result = SiteGenetic(beta, allele_freq, site_genetic)
        
        return result
        
    def _individual_genetic(self, G):
        """
        Convert genetic value of nodes to be genetic value of individuals
        """
        I = np.zeros(self.ts.num_individuals)

        for j in range(self.ts.num_samples):
            ind = self.ts.nodes_individual[self.sample_index_map[j]]
            I[ind] += G[j]
            
        return I

    def _environment(self, G):
        """
        Add environmental noise
        G should be values of individuals
        """
        trait_sd = self.model.trait_sd
        num_ind = len(G)
        if self.h2 == 1:
            E = np.zeros(num_ind)
            phenotype = G
        elif self.h2 == 0:
            E = self.rng.normal(loc=0.0, scale=trait_sd, size=num_ind)
            phenotype = E
        else:
            E = self.rng.normal(loc=0.0, scale=np.sqrt((1-self.h2)/self.h2 * np.var(G)), size=num_ind)
            phenotype = G + E
        
        return phenotype, E
    
    def effect_size_simulation(self):
        causal_state_list = self._obtain_causal_state()
        tree = tskit.Tree(self.ts)
        G = np.zeros(self.ts.num_samples)
        beta_list = np.zeros(self.num_causal)
        frequency = np.zeros(self.num_causal)

        for i, site_id in enumerate(self.site_id_list):
            site = self.ts.site(site_id)
            tree.seek(site.position)
            site_genotype = self._site_genotypes(tree, site, causal_state_list[i])
            genetic_info_site = self._effect_size_simulation_site(site_genotype)
            beta_list[i] = genetic_info_site.beta
            frequency[i] = genetic_info_site.allele_freq
            G += genetic_info_site.site_genetic
        
        G = self._individual_genetic(G)
        
        phenotype, E = self._environment(G)
        
        genetic_result = GeneticResult(self.site_id_list, causal_state_list, beta_list, frequency)
        phenotype_result = PhenotypeResult(list(range(self.ts.num_individuals)), phenotype, E, G)
        
        return genetic_result, phenotype_result

def _sim_phenotype(ts, num_causal, h2, model, rng):
    simulateClass = SimPhenotype(ts, num_causal, h2, model, rng)
    
    genetic_result, phenotype_result = simulateClass.effect_size_simulation()
    # Ancestral state and location can easily be determined
    
    return phenotype_result, genetic_result
    
def sim_phenotype(ts, num_causal, model, h2=0.3, random_seed=None):
    if not isinstance(ts, tskit.TreeSequence):
        raise TypeError("Input should be a tree sequence data")
    if int(num_causal) != num_causal or num_causal < 0:
        raise ValueError("Number of causal sites should be a non-negative integer")   
    if not isinstance(model, trait.TraitModel):
        raise TypeError("Mutation model must be an instance of TraitModel")
    if h2 > 1 or h2 < 0:
        raise ValueError("Heritability should be 0 <= h2 <= 1")
    
    rng = np.random.default_rng(random_seed)
    
    return _sim_phenotype(ts, num_causal, h2, model, rng)
