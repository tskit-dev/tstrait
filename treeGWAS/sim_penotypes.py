import msprime
import numpy as np
import tskit
import pandas as pd

def choose_causal(ts, num_sites, trait_sd, rng):
    assert num_sites <= ts.num_mutations
    mutation_id = rng.choice(range(ts.num_mutations), size=num_sites, replace=False)
    beta = rng.normal(loc=0, scale=trait_sd, size=num_sites)
    return mutation_id, beta

def environment(ts, G, h2):
    assert len(G) == ts. num_individuals
    E = np.random.normal(loc=0.0, scale=np.sqrt((1-h2)/h2 * np.var(G)), size=ts.num_individuals)        
    phenotype = G + E
    return phenotype, E

def update_node_values(tree, node_values, G):
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

def parse_genotypes(ts, mutation_id, beta):
    size_G = np.max(ts.samples())+1
    size_mutation = np.max(mutation_id)+1
    G = np.zeros(size_G)
    location = np.zeros(len(mutation_id))
    mutation_list = np.zeros(len(mutation_id))
    snp_idx = 0
    for tree in ts.trees():
        node_values = np.zeros(ts.num_nodes)
        for mut in tree.mutations():
            if mut.id in mutation_id:
                node_values[mut.node] += beta[snp_idx]
                location[snp_idx] = ts.site(mut.site).position
                mutation_list[snp_idx] = mut.id
                snp_idx += 1
        G = update_node_values(tree, node_values, G)
    # Convert node values to individual values
    G = G[::2] + G[1::2]
    G = G[G != 0]
    location = location[location != 0]
    assert len(location) == len(mutation_id)
    assert len(G) == ts.num_individuals
    
    return G, location, mutation_list

def phenotype_sim(ts, num_sites, trait_sd=1, h2=0.3, seed=1, addVar=1):
    rng = np.random.default_rng(seed)
    mutation_id, beta = choose_causal(ts, num_sites, trait_sd, rng)
    # This G is genotype of node (not individuals)
    G, location, mutation_list = parse_genotypes(ts, mutation_id, beta)
    phenotype, E = environment(ts, G, h2)
    
    
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