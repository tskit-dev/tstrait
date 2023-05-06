"""
import pytest

import msprime
import numpy as np
import tskit
import pandas as pd

import tstrait.phenotype as sim_pheno
import tstrait.phenotype_matrix as sim_matrix

          
class Test_phenotype_sim:
    @pytest.mark.parametrize("seed", [1,2,3,4,5])
    @pytest.mark.parametrize("num_causal", [1,2,5])
    @pytest.mark.parametrize("size", [1, 2, 5, 10])
    @pytest.mark.parametrize("seq_len", [500_000, 700_000, 1_000_000])
    def test_phenotype_sim(self, seed, num_causal, size, seq_len):
        ts = msprime.sim_ancestry(size, sequence_length=seq_len, recombination_rate=1e-8,population_size=10**4, random_seed=seed)
        ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed, model="binary", discrete_genome=False)
        pheno_df, gene_df = sim_pheno.phenotype_sim(ts, num_causal, trait_sd=1, h2=0.3, seed=seed)
        pheno_df1, gene_df1 = sim_matrix.phenotype_sim(ts, num_causal, trait_sd=1, h2=0.3, seed=seed)
        
        pheno_df = pheno_df.sort_values(by=['Individual ID']).reset_index(drop=True)
        pheno_df1 = pheno_df1.sort_values(by=['Individual ID']).reset_index(drop=True)
        
        gene_df = gene_df.sort_values(by=['Mutation ID']).reset_index(drop=True)
        gene_df1 = gene_df1.sort_values(by=['Mutation ID']).reset_index(drop=True)
        
        assert np.allclose(pheno_df["Individual ID"], pheno_df1["Individual ID"])
        assert np.allclose(pheno_df["Phenotype"], pheno_df1["Phenotype"])
        assert np.allclose(pheno_df["Environment"], pheno_df1["Environment"])
        assert np.allclose(pheno_df["Genotype"], pheno_df1["Genotype"])
        
        assert np.allclose(gene_df["Mutation ID"], gene_df1["Mutation ID"])
        assert np.allclose(gene_df["Location"], gene_df1["Location"])
        assert np.allclose(gene_df["Effect Size"], gene_df1["Effect Size"])
        
"""