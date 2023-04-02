import msprime
import numpy as np
import tskit
import pandas as pd


def choose_causal(num_mutations, num_causal, trait_sd, rng):
    if num_causal > num_mutations:
        raise ValueError("Too many causal sites")
    if trait_sd <= 0:
        raise ValueError("Standard deviation should be greater than 0")
    mutation_id = rng.choice(range(num_mutations), size=num_causal, replace=False)
    beta = rng.normal(loc=0, scale=trait_sd/ np.sqrt(num_causal), size=num_causal)
    return mutation_id, beta

def environment(G, h2):
    if h2 >= 1 or h2 <= 0:
        raise ValueError("Heritability should be 0 < h2 < 1")
    num_ind = len(G)
    E = np.random.normal(loc=0.0, scale=np.sqrt((1-h2)/h2 * np.var(G)), size=num_ind)
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

def update_node_values_array_access(root, left_child_array, right_sib_array, node_values, G):
    stack = [root]
    while stack:
        parent = stack.pop()
        child = left_child_array[parent]
        if child != -1:
            node_values[child] += node_values[parent]
            stack.append(child)
            right_sib = right_sib_array[child]
            assert right_sib != -1
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
        #G = update_node_values_old(tree, node_values, G)
        G = update_node_values_array_access(tree.root, tree.left_child_array, tree.right_sib_array, node_values, G)
    # Convert node values to individual values
    G = G[ts.samples()]
    G = G[::2] + G[1::2]
    location = location[location != 0]
    
    return G, location, mutation_list

def phenotype_sim(ts, num_causal, trait_sd=1, h2=0.3, seed=1):
    rng = np.random.default_rng(seed)
    mutation_id, beta = choose_causal(ts.num_mutations, num_causal, trait_sd, rng)
    # This G is genotype of individuals
    G, location, mutation_list = parse_genotypes(ts, mutation_id, beta)
    phenotype, E = environment(G, h2)
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