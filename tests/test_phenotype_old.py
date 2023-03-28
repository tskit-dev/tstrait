import pytest

import msprime
import numpy as np
import tskit
import pandas as pd

import treeGWAS.sim_phenotypes as sim_pheno

"""

class Test_environment:
    def test_environment(self):
        seeds = np.random.randint(1, 2**16, 10)
        num_causal = np.array([1, 100, 300])
        size = 100
        for seed in seeds:
            rng = np.random.default_rng(seed)
            ts = msprime.sim_ancestry(size, sequence_length=1_000_000, recombination_rate=1e-8,population_size=10**4, random_seed=seed)
            ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed, model="binary", discrete_genome=False)
            for causal in num_causal:
                mutation_id, beta = sim_pheno.choose_causal(ts, causal, 1, rng)
                G, location, mutation_list = sim_pheno.parse_genotypes(ts, mutation_id, beta)
                phenotype, E = sim_pheno.environment(ts, G, 0.3)
                assert len(phenotype) == ts.num_individuals
                assert len(E) == ts.num_individuals
      
class Test_num_sites:
    def test_num_sites(self):
        seeds = np.random.randint(1, 2**16, 10)
        num_causal = np.array([1, 2, 3, 4, 5])
        size = 100
        for seed in seeds:
            rng = np.random.default_rng(seed)
            ts = msprime.sim_ancestry(size, sequence_length=100_000, recombination_rate=1e-8,population_size=10**4, random_seed=seed)
            ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed, model="binary", discrete_genome=False)
            for causal in num_causal:
                with pytest.raises(ValueError):
                    sim_pheno.choose_causal(ts, causal + ts.num_sites, 1, rng)
              
class Test_update_node_values_array_access:
    def test_update_node_values_array_access(self):
        seeds = np.random.randint(1, 2**16, 10)
        num_causal = np.array([1, 10, 100])
        size = 100
        for seed in seeds:
            ts = msprime.sim_ancestry(size, sequence_length=100_000, recombination_rate=1e-8,population_size=10**4, random_seed=seed)
            ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed, model="binary", discrete_genome=False)
            for causal in num_causal:
                mutation_id, beta = sim_pheno.choose_causal(ts, causal, 1, np.random.default_rng(seed))
                snp_idx = 0
                size_G = np.max(ts.samples())+1
                G = np.zeros(size_G)
                for tree in ts.trees():
                    node_values = np.zeros(ts.num_nodes)
                    for mut in tree.mutations():
                        if mut.id in mutation_id:
                            node_values[mut.node] += beta[snp_idx]
                            snp_idx += 1
                    G = sim_pheno.update_node_values_array_access(tree.root, tree.left_child_array, tree.right_sib_array, node_values, G)
                assert snp_idx == causal
            G = G[ts.samples()]
            assert len(G) == ts.num_samples

# Check if all leaves and edges are accessed
class Test_update_node_values_array_access_leaf:
    def test_update_node_values_array_access(self):
        seeds = np.random.randint(1, 2**16, 10)
        num_causal = np.array([1, 10, 100])
        size = 10
        for seed in seeds:
            ts = msprime.sim_ancestry(size, sequence_length=1_000_000, recombination_rate=1e-8,population_size=10**4, random_seed=seed)
            ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed, model="binary", discrete_genome=False)
            for causal in num_causal:
                mutation_id, beta = sim_pheno.choose_causal(ts, causal, 1, np.random.default_rng(seed))
                snp_idx = 0
                size_G = np.max(ts.samples())+1
                G = np.zeros(size_G)
                for tree in ts.trees():
                    node_values = np.zeros(ts.num_nodes)
                    for mut in tree.mutations():
                        if mut.id in mutation_id:
                            node_values[mut.node] += beta[snp_idx]
                            snp_idx += 1
                    left_child_array = tree.left_child_array
                    right_sib_array = tree.right_sib_array
                    stack = [tree.root]
                    child_count = ts.samples()
                    edge_list = tree.edge_array
                    edge_list = np.setdiff1d(edge_list, -1)
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
                            edge_list = np.setdiff1d(edge_list, [tree.edge(child), tree.edge(right_sib)])
                        else:
                            G[parent] += node_values[parent]
                            child_count = np.setdiff1d(child_count, parent)
                    assert child_count.size == 0
                    assert edge_list.size == 0
 
class Test_parse_genotypes:
    def test_parse_genotypes(self):
        seeds = np.random.randint(1, 2**16, 10)
        num_causal = np.array([1, 10, 100])
        size = 1000
        for seed in seeds:
            ts = msprime.sim_ancestry(size, sequence_length=100_000, recombination_rate=1e-8,population_size=10**4, random_seed=seed)
            ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed, model="binary", discrete_genome=False)
            assert ts.num_sites == ts.num_mutations
            for causal in num_causal:
                mutation_id, beta = sim_pheno.choose_causal(ts, causal, 1, np.random.default_rng(seed))
                G, location, mutation_list = sim_pheno.parse_genotypes(ts, mutation_id, beta)
                assert len(location) == len(mutation_id)
                assert len(G) == ts.num_individuals



class Test_parse_genotypes_comparison:
    def old_code(self, ts, mutation_id, beta):
        G = np.zeros(np.max(ts.samples())+1)
        mutation_list = np.zeros(len(mutation_id))
        snp_idx = 0
        for tree in ts.trees():
            for mut in tree.mutations():
                if mut.id in mutation_id:
                    for sample in tree.samples(mut.node):
                        G[sample] += beta[snp_idx]
                        mutation_list[snp_idx] = mut.id
                    snp_idx += 1
        G = G[::2] + G[1::2]
        return G, mutation_list
    
    def test_genotype(self):
        seeds = np.random.randint(1, 2**16, 10)
        num_causal = np.array([1, 10, 100, 1000])
        size = 100
        for seed in seeds:
            ts = msprime.sim_ancestry(size, sequence_length=1_000_000, recombination_rate=1e-8,population_size=10**4, random_seed=seed)
            ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed, model="binary", discrete_genome=False)
            for causal in num_causal:
                mutation_id, beta = sim_pheno.choose_causal(ts, causal, 1, np.random.default_rng(seed))
                G1, mutation_list1 = self.old_code(ts, mutation_id, beta)
                G, location, mutation_list = sim_pheno.parse_genotypes(ts, mutation_id, beta)
                assert (mutation_list1 == mutation_list).all()
                assert np.isclose(G1, G)
         
"""                
          
     