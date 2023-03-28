import pytest

import msprime
import numpy as np
import tskit
import pandas as pd

import treeGWAS.sim_phenotypes as sim_pheno

class Test_update_node_values_array_access:
    def test_update(self):
        # Construct an artificial tree with 6 leaves with node 7 and 9 missing from the tree
        # We will be having mutation in every edge having genetic value 1 to be sure that all the edges are captured in the tree traversal algorithm
        root = 12
        left_child_array = np.array([-1,-1,-1,-1,-1,-1,0,-1,2,-1,4,6,11,12])
        right_sib_array = np.array([1,-1,3,-1,5,-1,8,-1,-1,-1,-1,10,-1,-1])
        node_values = np.array([1,1,1,1,1,1,1,0,1,0,1,1,1])
        
        # Create an arbitrary genotype vector
        G = np.random.normal(loc=0, scale=1, size=6)        
        G1 = np.copy(G)
        G = sim_pheno.update_node_values_array_access(root, left_child_array, right_sib_array, node_values, G)
        
        actual_G = np.array([4,4,4,4,3,3]) + G1
        
        assert np.array_equal(G, actual_G)
        

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
"""