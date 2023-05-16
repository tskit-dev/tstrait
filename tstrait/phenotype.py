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
# Genetic result class
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
    
def environment(G, h2, model, rng):
    """
    Add environmental noise
    """
    trait_sd = model.trait_sd
    num_ind = len(G)
    if h2 == 1:
        E = np.zeros(num_ind)
        phenotype = G
    elif h2 == 0:
        E = rng.normal(loc=0.0, scale=trait_sd, size=num_ind)
        phenotype = E
    else:
        E = rng.normal(loc=0.0, scale=np.sqrt((1-h2)/h2 * np.var(G)), size=num_ind)
        phenotype = G + E
    return phenotype, E

def count_site_alleles(ts, tree, site):
    """
    Obtain a dictionary of allele frequency counts, excluding the ancestral state
    Input is the tree sequence site (ts.site(ID))
    If only state exists, don't delete the ancestral state
    """
    counts = collections.Counter({site.ancestral_state: ts.num_samples})
    for m in site.mutations:
        current_state = site.ancestral_state
        if m.parent != tskit.NULL:
            current_state = ts.mutation(m.parent).derived_state
        # Silent mutations do nothing
        if current_state != m.derived_state:
            num_samples = tree.num_samples(m.node)
            counts[m.derived_state] += num_samples
            counts[current_state] -= num_samples
    if len(counts) > 1:
        del counts[site.ancestral_state]
    return counts

def obtain_causal_state(ts, site_id_list, rng):
    """
    Obtain causal alleles from causal sites, and return the site ID and the corresponding causal state
    They are aligned based on their genomic positions
    """ 
    causal_state_list = np.zeros(len(site_id_list), dtype=object)
    tree = tskit.Tree(ts)
    
    for i, single_id in enumerate(site_id_list):
        site = ts.site(single_id)
        tree.seek(site.position)
        counts = count_site_alleles(ts, tree, site)
        causal_state_list[i] = rng.choice(list(counts))
    
    return causal_state_list

def examine_mutation(ts, site_id, causal_state):
    """
    Input is the tree and site data.
    The function will obtain a list of nodes with mutation
    1 represents causal mutation, -1 represents non-causal mutation, and 0 represents no mutation
    and the vector to show if a node has a mutation or not
    """
    site = ts.site(site_id)
    has_mutation = np.zeros(ts.num_nodes + 1)
    for m in site.mutations:
        has_mutation[m.node] = -1
        if m.derived_state == causal_state:
            has_mutation[m.node] = 1
    
    return has_mutation
    

def site_genotypes(ts, tree, site_id, causal_state, sample_index_map):
    """
    Obtain list of 0 and 1, where 1 represents sample nodes with the causal mutation
    """
    has_mutation = examine_mutation(ts, site_id, causal_state)
    causal_node_id = np.where(has_mutation == 1)[0].tolist()
    has_causal_mutation = np.zeros(ts.num_samples)
    while len(causal_node_id) > 0:
        u = causal_node_id.pop()
        j = sample_index_map[u]
        if j > 0:
            has_causal_mutation[j] = 1
        for v in tree.children(u):
            if has_mutation[v] == 0:
                causal_node_id.append(v)
    
    return has_causal_mutation

def obtain_beta(model, num_causal, allele_freq, rng):
    """
    Obtain effect size from the model
    """
    kwargs = {"num_causal": num_causal, "rng": rng}
    if model._require_allele_freq:
        kwargs["allele_freq"] = allele_freq
    
    beta = model.sim_effect_size(**kwargs)
    
    return beta

def effect_size_simulation_site(ts, tree, site_id, causal_state, num_causal, sample_index_map, model, rng):
    """
    Simulate effect size of sample nodes in a single site
    """ 
    has_causal_mutation = site_genotypes(ts, tree, site_id, causal_state, sample_index_map)
    allele_freq = np.sum(has_causal_mutation) / len(has_causal_mutation)
    if allele_freq > 0 and allele_freq < 1:
        beta = obtain_beta(model, num_causal, allele_freq, rng)
    else:
        beta = 0
    genetic_causal_node = has_causal_mutation * beta
    
    return beta, allele_freq, genetic_causal_node
    
def effect_size_simulation(ts, causal_state_list, site_id_list, sample_index_map, model, rng):
    tree = ts.first()
    G = np.zeros(ts.num_samples)
    beta_list = np.zeros(len(site_id_list))
    frequency = np.zeros(len(site_id_list))
    num_causal = len(site_id_list)
    
    for i, site_id in enumerate(site_id_list):
        location = ts.site(site_id).position
        tree.seek(location)
        beta, allele_freq, genetic_causal_node = effect_size_simulation_site(ts, tree, site_id, causal_state_list[i], num_causal, sample_index_map, model, rng)
        beta_list[i] = beta
        frequency[i] = allele_freq
        G += genetic_causal_node
    
    return G, beta_list, frequency
    

def individual_genetic(ts, G):
    """
    Convert genetic value of nodes to be genetic value of individuals
    """
    G = G[ts.samples()]
    G = G[::2] + G[1::2]
    return G

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
        if num_causal > num_sites:
            raise ValueError("There are less number of sites in the tree sequence than the inputted number of causal sites")
        site_id = self.rng.choice(range(num_sites), size=self.num_causal, replace=False)
        site_id = np.sort(site_id)
        
        return site_id
        
    def obtain_causal_state(self):
    """
    Obtain causal alleles from causal sites, and return the site ID and the corresponding causal state
    They are aligned based on their genomic positions
    """ 
    causal_state_list = np.zeros(len(self.site_id_list), dtype=object)
    tree = tskit.Tree(ts)
    
        for i, single_id in enumerate(self.site_id_list):
            site = ts.site(single_id)
            tree.seek(site.position)
            counts = count_site_alleles(self.ts, tree, site)
            causal_state_list[i] = rng.choice(list(counts))
        
        return causal_state_list

@dataclass
class Result:
# Genetic result class
    G: np.ndarray
    beta_list: np.ndarray
    frequency: np.ndarray
    


def _sim_phenotype(ts, num_causal, h2, model, rng):
    simulateClass = SimPhenotype(ts, num_causal, h2, model, rng)
    
    causal_state_list = simulateClass.obtain_causal_state()
    
    G, beta_list, frequency = effect_size_simulation(ts, causal_state_list, site_id_list, sample_index_map, model, rng)
    
    G = individual_genetic(ts, G)
    
    phenotype, E = environment(G, h2, model, rng)
    
    phenotype_result = PhenotypeResult(list(range(ts.num_individuals)), phenotype, E, G)
    genetic_result = GeneticResult(site_id_list, causal_state_list, beta_list, frequency)
    # Ancestral state and location can easily be determined
    
    return phenotype_result, genetic_result
    
def sim_phenotype(ts, num_causal, model, h2=0.3, random_seed=None):
    if type(ts) != tskit.trees.TreeSequence:
        raise TypeError("Input should be a tree sequence data")
    if not isinstance(num_causal, int):
        raise TypeError("Number of causal sites should be an integer")
    if num_causal < 0:
        raise ValueError("Number of causal sites should be a non-negative integer")        
    if not isinstance(model, trait.TraitModel):
        raise TypeError("Mutation model must be an instance of TraitModel")         
    if (not isinstance(h2, int) and not isinstance(h2, float)):
        raise TypeError("Heritability should be a float or an integer")
    if h2 > 1 or h2 < 0:
        raise ValueError("Heritability should be 0 <= h2 <= 1")
    if (not isinstance(random_seed, int) and random_seed != None):
        raise TypeError("Random seed should be None or an integer")
    if random_seed != None:
        if random_seed < 0:
            raise ValueError("Random seed should be None or a non-negative integer")
    
    rng = np.random.default_rng(random_seed)
    
    return _sim_phenotype(ts, num_causal, h2, model, rng)
