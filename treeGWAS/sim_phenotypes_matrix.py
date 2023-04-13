import msprime
import numpy as np
import tskit
import pandas as pd


def choose_causal(num_mutations, num_causal, trait_sd, rng):
    if not isinstance(num_causal, int) or num_causal < 0:
        raise ValueError("Number of causal sites should be a non-negative integer")
    if num_mutations == 0:
        raise ValueError("No mutation in the provided data")
    if not isinstance(num_mutations, int) or num_mutations < 0:
        raise ValueError("Number of mutation sites should be a non-negative integer")
    if num_causal > num_mutations:
        raise ValueError("There are more causal sites than the number of mutations inside the tree sequence")
    if (not isinstance(trait_sd, int) and not isinstance(trait_sd, float)) or trait_sd <= 0:
        raise ValueError("Standard deviation should be a non-negative number")
    mutation_id = rng.choice(range(num_mutations), size=num_causal, replace=False)
    mutation_id = np.sort(mutation_id)
    if num_causal > 0:
        beta = rng.normal(loc=0, scale=trait_sd/ np.sqrt(num_causal), size=num_causal)
    else:
        beta = np.array([])
    return mutation_id, beta

def environment(G, h2, trait_sd, rng):
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

def genetic_value(ts, mutation_id, beta, num_causal):
    if type(ts) != tskit.trees.TreeSequence:
        raise TypeError("Input should be a tree sequence data")
    
    G = np.zeros((num_causal, int(ts.num_samples/2)))    
    var = tskit.Variant(ts)
    mutation_list = np.zeros(len(mutation_id), dtype=int)
    location = np.zeros(len(mutation_id))
    for i, ID in enumerate(mutation_id):
        var.decode(ts.mutation(ID).site)
        G[i] = var.genotypes[::2] + var.genotypes[1::2]
        location[i] = var.site.position
        mutation_list[i] = ID
    
    G = np.inner(beta, G.T)
    return G, location, mutation_list

def phenotype_sim(ts, num_causal, trait_sd=1, h2=0.3, seed=1):
    if type(ts) != tskit.trees.TreeSequence:
        raise TypeError("Input should be a tree sequence data")    
    rng = np.random.default_rng(seed)
    mutation_id, beta = choose_causal(ts.num_mutations, num_causal, trait_sd, rng)
    # This G is genotype of individuals
    G, location, mutation_list = genetic_value(ts, mutation_id, beta, num_causal)
    phenotype, E = environment(G, h2, trait_sd, rng)
    assert len(phenotype) == ts.num_individuals
    
    
    # Phenotype dataframe
    # 1st column = Individual ID
    # 2nd column = Phenotypic value
    pheno_df = pd.DataFrame({"Individual ID": [s.id for s in ts.individuals()],
                             "Phenotype": phenotype,"Environment":E,"Genotype": G})
    
    # Genotype dataframe
    # 1st column = site ID
    # 2nd column = location of sites in the genome
    # 3rd column = Effect size
    # 4th column = Reference allele
    # 5th column = Minor allele frequency
    gene_df = pd.DataFrame({
        "Mutation ID": mutation_list,
        "Location": location,
        "Effect Size": beta,
        #"Frequency": frequency
    })

    return pheno_df, gene_df