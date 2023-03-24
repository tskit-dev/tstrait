import pytest

import msprime
import numpy as np
import tskit
import pandas as pd

import treeGWAS.sim_phenotypes as sim_pheno

class Test_parse_genotypes:
    def old_code(ts, mutation_id, beta):
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
        size = 1000
        for seed in seeds:
            ts = msprime.sim_ancestry(size, sequence_length=1_000_000, recombination_rate=1e-8,population_size=10**4, random_seed=seed)
            ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed, model="binary", discrete_genome=False)
            for causal in num_causal:
                mutation_id, beta = sim_pheno.choose_causal(ts, causal, 1, np.random.default_rng(seed))
                G1, mutation_list1 = self.old_code(ts, mutation_id, beta)
                G, location, mutation_list = sim_pheno.parse_genotypes(ts, mutation_id, beta)
                assert (G1 == G).all()
                assert (mutation_list1 == mutation_list).all()