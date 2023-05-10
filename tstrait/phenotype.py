import msprime
import numpy as np
import tskit
import pandas as pd
from numba import jit


class TraitModel:
# Trait model class
    def __init__(self, model_name, trait_mean, trait_sd, num_causal, allele_freq):
        self._model_name = model_name
        self.trait_mean = trait_mean
        self.trait_sd = trait_sd
        self.num_causal = num_causal
        self.allele_freq = allele_freq
    
    def sim_effect_size(self, random_seed=None):
        rng = np.random.default_rng(random_seed)
        if self.trait_sd == 0:
            beta = self.trait_mean
        else:
            beta = rng.normal(loc=self.trait_mean, scale=self.trait_sd / np.sqrt(self.num_causal))
        return beta   
    @property
    def name(self):
        return self._model_name

class TraitModelGCTA(TraitModel):
# GCTA model (Effect size simulation won't be affected by allele frequency)
    def __init__(self, trait_mean, trait_sd, num_causal, allele_freq):
        super().__init__('gcta', trait_mean, trait_sd, num_causal, allele_freq)

class TraitModelAllele(TraitModel):
# Allele model (Effect size will be affected by allele frequency)
    def __init__(self, trait_mean, trait_sd, num_causal, allele_freq):
        super().__init__('allele', trait_mean, trait_sd, num_causal, allele_freq)
    def sim_effect_size(self, random_seed=None):
        beta = super().sim_effect_size(random_seed)
        beta /= np.sqrt(2 * self.allele_freq * (1 - self.allele_freq))
        return beta
    
class TraitModelLDAK(TraitModel):
# LDAK model (Effect size will be affected by allele frequency and alpha parameter)
    def __init__(self, trait_mean, trait_sd, num_causal, allele_freq, alpha):
        super().__init__('ldak', trait_mean, trait_sd, num_causal, allele_freq)
        self.alpha = alpha
    def sim_effect_size(self, random_seed=None):
        beta = super().sim_effect_size(random_seed)
        beta *= pow(self.allele_freq * (1 - self.allele_freq), self.alpha)
        return beta
        
        
MODEL_MAP = {
    "gcta": TraitModelGCTA,
    "allele": TraitModelAllele,
    # "ldak": TraitModelLDAK, Needs alpha argument
}


def effect_size_model(model, trait_mean, trait_sd, num_causal, allele_freq):
    """
    Returns a mutation model corresponding to the specified model.
    - If model is None, the default mutation model is returned.
    - If model is a string, return the corresponding model instance.
    - If model is an instance of MutationModel, return it.
    - Otherwise raise a type error.
    """

    if model is None:
        model_instance = TraitModelGCTA(trait_mean, trait_sd, num_causal, allele_freq)
    elif isinstance(model, str):
        lower_model = model.lower()
        if lower_model not in MODEL_MAP:
            raise ValueError(
                "Model '{}' unknown. Choose from {}".format(
                    model, sorted(MODEL_MAP.keys())
                )
            )
        model_instance = MODEL_MAP[lower_model](trait_mean, trait_sd, num_causal, allele_freq)
    elif isinstance(model, TraitModel):
        model_instance = model
    else:
        raise TypeError(
            "Mutation model must be a string or an instance of TraitModel"
        )
    return model_instance


def obtain_sample_index_map(num_nodes, sample_id_list):
    sample_index_map = np.zeros(num_nodes + 1, dtype=int) - 1
    for j, u in enumerate(sample_id_list):
        sample_index_map[u] = j
    return sample_index_map

def choose_causal(ts, num_causal, random_seed = None):
    """
    Choose causal sites from tree sequence data and return the site ID
    """
    rng = np.random.default_rng(random_seed)
    if type(ts) != tskit.trees.TreeSequence:
        raise TypeError("Input should be a tree sequence data")  
    if not isinstance(num_causal, int):
        raise TypeError("Number of causal sites should be a non-negative integer")
    if num_causal < 0:
        raise ValueError("Number of causal sites should be a non-negative integer")
    
    num_sites = ts.num_sites
    
    if num_sites == 0:
        raise ValueError("No mutation in the provided data")
    if num_causal > num_sites:
        raise ValueError("There are more causal sites than the number of mutations inside the tree sequence")

    site_id = rng.choice(range(num_sites), size=num_causal, replace=False)
    
    return site_id
    
def environment(G, h2, trait_sd, random_seed = None):
    """
    Add environmental noise
    """
    rng = np.random.default_rng(random_seed)
    if len(G) == 0:
        raise ValueError("No individuals in the simulation model")
    if (not isinstance(h2, int) and not isinstance(h2, float)) or h2 > 1 or h2 < 0:
        raise ValueError("Heritability should be 0 <= h2 <= 1")
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

def obtain_causal_state(ts, site_id_list, random_seed = None):
    """
    Obtain causal alleles from causal sites, and return the ancestral state, causal allele, and genomic location
    They are aligned based on their genomic positions
    """
    rng = np.random.default_rng(random_seed)
    if type(ts) != tskit.trees.TreeSequence:
        raise TypeError("Input should be a tree sequence data")  
    location = np.zeros(len(site_id_list))
    ancestral = np.array([])
    causal_state_list = np.array([])
    
    for i, single_id in enumerate(site_id_list):
        allele_list = np.array([])
        for m in ts.site(single_id).mutations:
            if m.derived_state != ts.site(single_id).ancestral_state:
                allele_list = np.append(allele_list, m.derived_state)

        causal_state_list = np.append(causal_state_list, rng.choice(np.unique(allele_list)))
        location[i] = ts.site(single_id).position
        ancestral = np.append(ancestral, ts.site(single_id).ancestral_state)
    
    coordinate = np.argsort(location)
    location = location[coordinate]
    site_id_list = site_id_list[coordinate]
    ancestral = ancestral[coordinate]
    causal_state_list = causal_state_list[coordinate]
    
    return site_id_list, ancestral, causal_state_list, location

def check_new_mutation(site, mutation):
    """
    Check if the mutation inside the edge is the newest mutation or not
    site = ts.site(..) and mutation = ts.mutation(...)
    """
    newest = True
    time = mutation.time
    edge = mutation.edge
    for m in site.mutations:
        if m.edge == edge and m.time < time:
            newest = False
            break
    
    return newest

def examine_mutation(num_nodes, ts, site, causal_state):
    """
    Input is the tree and site data.
    The function will obtain a list of nodes with the causal state that is the newest in the edge,
    and the vector to show if a node has a mutation or not
    site = ts.site(..)
    """
    causal_node_id = []
    has_mutation = np.zeros(ts.num_nodes + 1, dtype=bool)
    for m in site.mutations:
        has_mutation[m.node] = True
        if m.derived_state == causal_state:
            if check_new_mutation(site, m):
                causal_node_id.append(m.node)
    
    return has_mutation, causal_node_id
    

def site_genotypes(ts, tree, site, causal_state, sample_index_map):
    """
    Obtain list of 0 and 1, where 1 represents sample nodes with the causal mutation
    site = ts.site(..)
    """
    has_mutation, causal_node_id = examine_mutation(ts.num_nodes, ts, site, causal_state)
    
    has_causal_mutation = np.zeros(ts.num_samples)
    while len(causal_node_id) > 0:
        u = causal_node_id.pop()
        j = sample_index_map[u]
        if j > 0:
            has_causal_mutation[j] = 1
        for v in tree.children(u):
            if not has_mutation[v]:
                causal_node_id.append(v)
    
    return has_causal_mutation

def effect_size_simulation_site(ts, tree, site, causal_state, sample_index_map, model, trait_mean, trait_sd, num_causal, random_seed):
    """
    Simulate effect size of sample nodes in a single site
    site = ts.site(..)
    """ 
    has_causal_mutation = site_genotypes(ts, tree, site, causal_state, sample_index_map)
    allele_freq = np.sum(has_causal_mutation) / len(has_causal_mutation)
    beta_model = effect_size_model(model, trait_mean, trait_sd, num_causal, allele_freq)
    beta = beta_model.sim_effect_size(random_seed)
    genetic_causal_node = has_causal_mutation * beta
    
    return beta, allele_freq, genetic_causal_node
    
    
    
def effect_size_simulation(ts, causal_state_list, site_id_list, location, sample_index_map, model, trait_mean, trait_sd, num_causal, random_seed):
    tree = ts.first()
    G = np.zeros(ts.num_samples)
    beta_list = np.zeros(len(location))
    frequency = np.zeros(len(location))
    
    for i, loc in enumerate(location):
        tree.seek(loc)
        site = ts.site(site_id_list[i])
        beta, allele_freq, genetic_causal_node = effect_size_simulation_site(ts, tree, site, causal_state_list[i], sample_index_map, model, trait_mean, trait_sd, num_causal, random_seed)
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


def phenotype_sim(ts, num_causal, trait_mean=0, trait_sd=1, h2=0.3, model = None, random_seed=None):
    if type(ts) != tskit.trees.TreeSequence:
        raise TypeError("Input should be a tree sequence data")  
    
    sample_index_map = obtain_sample_index_map(ts.num_nodes, ts.samples())
    
    site_id_list = choose_causal(ts, num_causal, random_seed)
    
    site_id_list, ancestral, causal_state_list, location = obtain_causal_state(ts, site_id_list, random_seed)
    
    G, beta_list, frequency = effect_size_simulation(ts, causal_state_list, site_id_list, location,
        sample_index_map, model, trait_mean, trait_sd, num_causal, random_seed)
    
    G = individual_genetic(ts, G)
    
    phenotype, E = environment(G, h2, trait_sd, random_seed)
    assert len(phenotype) == ts.num_individuals
    
    
    # Phenotype dataframe
#    pheno_df = pd.DataFrame({"Individual ID": [s.id for s in ts.individuals()],
#                             "Phenotype": phenotype,"Environment":E,"Genotype": G})
    pheno_df = pd.DataFrame({"Individual ID": list(range(ts.num_individuals)),
                              "Phenotype": phenotype,"Environment":E,"Genetic Value": G})
    
    # Genotype dataframe
    gene_df = pd.DataFrame({
        "Site ID": site_id_list,
        "Ancestral State": ancestral,
        "Causal State": causal_state_list,
        "Location": location,
        "Effect Size": beta_list,
        "Frequency": frequency
    })

    return pheno_df, gene_df