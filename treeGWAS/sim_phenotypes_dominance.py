import msprime
import numpy as np
import tskit
import pandas as pd
from numba import jit


def choose_causal(num_mutations, num_causal, trait_mean, trait_sd, dominance_mean, dominance_sd, rng):
    if not isinstance(num_causal, int) or num_causal < 0:
        raise ValueError("Number of causal sites should be a non-negative integer")
    if num_mutations == 0:
        raise ValueError("No mutation in the provided data")
    if not isinstance(num_mutations, int) or num_mutations < 0:
        raise ValueError("Number of mutation sites should be a non-negative integer")
    if num_causal > num_mutations:
        raise ValueError("There are more causal sites than the number of mutations inside the tree sequence")
    if (not isinstance(trait_sd, int) and not isinstance(trait_sd, float)):
        raise TypeError("Standard deviation is in an incorrect type")
    if trait_sd <= 0:
        raise ValueError("Standard deviation should be a non-negative number")
    if (not isinstance(dominance_sd, int) and not isinstance(dominance_sd, float)) or dominance_sd <= 0:
        raise ValueError("Standard deviation of dominance should be a non-negative number")
    if (not isinstance(trait_mean, int) and not isinstance(trait_mean, float)):
        raise TypeError("Trait mean is in an incorrect type")      
    if (not isinstance(dominance_mean, int) and not isinstance(dominance_mean, float)):
        raise TypeError("Dominance mean is in an incorrect type")        
    mutation_id = rng.choice(range(num_mutations), size=num_causal, replace=False)
    mutation_id = np.sort(mutation_id)
    if num_causal > 0:
        beta = rng.normal(loc=trait_mean, scale=trait_sd/ np.sqrt(num_causal), size=num_causal)
    else:
        beta = np.array([])
    return mutation_id, beta, dominance

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

def update_node_values_tree_object(tree, node_values, G):
    stack = [tree.root]
    while stack:
        parent = stack.pop()
        child = tree.left_child(parent)
        if child != tskit.NULL:
            node_values[child] += node_values[parent]
            stack.append(child)
            right_sib = tree.right_sib(child)
            node_values[right_sib] += node_values[parent]
            stack.append(right_sib)
        else:
            G[parent] += node_values[parent]
    return G

@jit(nopython=True)
def update_node_values_array_access(root, left_child_array, right_sib_array, node_values, G):
    stack = [root]
    while stack:
        parent = stack.pop()
        child = left_child_array[parent]
        if child != -1:
            while child != -1:
                node_values[child] += node_values[parent]
                stack.append(child)
                child = right_sib_array[child]
        else:
            G[parent] += node_values[parent]

def genetic_value(ts, mutation_id, beta):
    if type(ts) != tskit.trees.TreeSequence:
        raise TypeError("Input should be a tree sequence data")  
    size_G = np.max(ts.samples())+1
    size_mutation = np.max(mutation_id)+1
    G = np.zeros(size_G)
    location = np.zeros(len(mutation_id))
    mutation_list = np.zeros(len(mutation_id), dtype=int)
    
    for i, mut_id in enumerate(mutation_id):
        mut = ts.mutation(mut_id)
        location[i] = ts.site(mut.site).position
    coordinate = np.argsort(location)
    location = np.sort(location)
    
    N = ts.num_nodes
    tree = ts.first()
    
    for i, loc in enumerate(location):
        tree.seek(loc)
        node_values = np.zeros(N)
        mut = ts.mutation(mutation_id[coordinate[i]])
        node_values[mut.node] += beta[coordinate[i]]
        update_node_values_array_access(mut.node, tree.left_child_array, tree.right_sib_array, node_values, G)

    # Convert node values to individual values
    G = G[ts.samples()]
    G = G[::2] + G[1::2]
    location = location[location != 0]
    
    return G, location, mutation_id


def phenotype_sim(ts, num_causal, trait_mean=0, trait_sd=1, dominance_mean=0, dominance_sd=0.1, h2=0.3, seed=1):
    if type(ts) != tskit.trees.TreeSequence:
        raise TypeError("Input should be a tree sequence data")    
    rng = np.random.default_rng(seed)
    mutation_id, beta, dominance = choose_causal(ts.num_mutations, num_causal, trait_mean, trait_sd, dominance_mean, dominance_sd, rng)
    # This G is genotype of individuals
    G, D, location, mutation_list = genetic_value(ts, mutation_id, beta, dominance)
    phenotype, E = environment(G, D, h2, trait_sd, rng)
    assert len(phenotype) == ts.num_individuals
    
    
    # Phenotype dataframe
    # 1st column = Individual ID
    # 2nd column = Phenotypic value
#    pheno_df = pd.DataFrame({"Individual ID": [s.id for s in ts.individuals()],
#                             "Phenotype": phenotype,"Environment":E,"Genotype": G})
    pheno_df = pd.DataFrame({"Individual ID": list(range(ts.num_individuals)),
                              "Phenotype": phenotype,"Environment":E,"Genotype": G, "Dominance": D})
    
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