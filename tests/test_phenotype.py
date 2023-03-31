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
        

class Test_environment:
    def test_environment(self):
        size_list = np.array([1, 10, 100, 1000, 10000])
        h2_list = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        for size in size_list:
            for h2 in h2_list:
                G = np.random.normal(size = size)
                phenotype, E = sim_pheno.environment(G, h2)
                assert len(phenotype) == size
                assert len(E) == size
    def test_error(self):
        size_list = np.array([1, 10, 100, 1000, 10000])
        h2_list = np.array([-0.2, 0, 1, 1.2])
        for h2 in h2_list:
            for size in size_list:
                G = np.random.normal(size = size)            
                with pytest.raises(ValueError):
                    phenotype, E = sim_pheno.environment(G, h2)    

class Test_choose_causal:
    def test_choose_causal(self):
        rng = np.random.default_rng(np.random.randint(10000))
        num_mutations_list = np.array([1000, 2000, 5000, 10000])
        num_causal_list = np.array([1, 10, 20, 100, 1000])
        trait_sd_list = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        for num_mutations in num_mutations_list:
            for num_causal in num_causal_list:
                for trait_sd in trait_sd_list:
                    mutation_id, beta = sim_pheno.choose_causal(num_mutations, num_causal, trait_sd, rng)
                    assert len(beta) == num_causal
                    assert len(mutation_id) == num_causal
                    assert min(mutation_id) >= 0
                    assert max(mutation_id) < num_mutations
    def test_error_num_sites(self):
        rng = np.random.default_rng(np.random.randint(10000))
        num_mutations_list = np.array([100, 300, 500, 700])
        addition_list = np.array([1, 10, 100, 1000])
        trait_sd_list = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        for num_mutations in num_mutations_list:
            for addition in addition_list:
                for trait_sd in trait_sd_list:
                    with pytest.raises(ValueError):
                        num_causal = num_mutations + addition
                        mutation_id, beta = sim_pheno.choose_causal(num_mutations, num_causal, trait_sd, rng)
    def test_error_trait_sd(self):
        rng = np.random.default_rng(np.random.randint(10000))
        num_mutations_list = np.array([1000, 2000, 5000, 10000])
        num_causal_list = np.array([1, 10, 20, 100, 1000])
        trait_sd_list = np.array([0, -0.1, -1, -10])
        for num_mutations in num_mutations_list:
            for num_causal in num_causal_list:
                for trait_sd in trait_sd_list:
                    with pytest.raises(ValueError):
                        mutation_id, beta = sim_pheno.choose_causal(num_mutations, num_causal, trait_sd, rng)

            

class Test_parse_genotypes:
    def test_parse_genotypes(self):
        seeds = np.random.randint(1, 2**16, 3)
        num_causal_list = np.array([1, 10, 50, 100])
        size = 1000
        for seed in seeds:
            rng = np.random.default_rng(seed)
            ts = msprime.sim_ancestry(size, sequence_length=100_000, recombination_rate=1e-8,population_size=10**4, random_seed=seed)
            ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed, model="binary", discrete_genome=False)
            for num_causal in num_causal_list:
                mutation_id = rng.choice(range(ts.num_mutations), size=num_causal, replace=False)
                beta = rng.normal(size=num_causal)
                G, location, mutation_list = sim_pheno.parse_genotypes(ts, mutation_id, beta)
                assert len(G) == ts.num_individuals
                assert len(location) == num_causal
                assert len(mutation_list) == num_causal
                
class Test_phenotype_sim:
    def test_phenotype_sim(self):
        seeds = np.random.randint(1, 2**16, 3)
        num_causal_list = np.array([1, 10, 50, 100])
        size = 1000
        for seed in seeds:
            rng = np.random.default_rng(seed)
            ts = msprime.sim_ancestry(size, sequence_length=100_000, recombination_rate=1e-8,population_size=10**4, random_seed=seed)
            ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed, model="binary", discrete_genome=False)
            for num_causal in num_causal_list:
                pheno_df, gene_df = sim_pheno.phenotype_sim(ts, num_causal, trait_sd=1, h2=0.3, seed=seed)
                
                # Check if any values of pheno_df / gene_df is null
                assert pheno_df.isnull().values.any() == False
                assert gene_df.isnull().values.any() == False