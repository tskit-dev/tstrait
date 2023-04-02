import msprime
import numpy as np
import tskit
import pandas as pd

import argparse
import inspect
import scipy
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import sys
from tqdm import tqdm

matplotlib.use("Agg")
import statsmodels.api as sm

import treeGWAS.sim_phenotypes as sim_pheno

class Test:
    def __init__(self, basedir, cl_name):
        self.basedir = basedir
        self.name = cl_name
        self.set_output_dir(basedir, cl_name)

    def set_output_dir(self, basedir, cl_name):
        output_dir = pathlib.Path(self.basedir) / cl_name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

    def require_output_dir(self, folder_name):
        output_dir = self.output_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

    def _get_tests(self):
        return [
            value
            for name, value in inspect.getmembers(self)
            if name.startswith("test_")
        ]

    def _run_tests(self):
        all_results = self._get_tests()
        print(f"[+] Running: {self.name}.")
        print(f"[+] Collected {len(all_results)} subtest(s).")
        for method in all_results:
            method()

    def _build_filename(self, filename, extension=".png"):
        return self.output_dir / (filename + extension)

    def plot_qq(self, v1, v2, x_label, y_label, filename, info=""):
        sm.graphics.qqplot(v1)
        sm.qqplot_2samples(v1, v2, x_label, y_label, line="45")
        #plt.xlabel(x_label)
        #plt.ylabel(y_label)
        plt.title(info)
        f = self._build_filename(filename)
        plt.savefig(f, dpi=72)
        plt.close("all")

    def get_seeds(self, num_replicates, seed=None):
        rng = np.random.default_rng(seed)
        max_seed = 2**16
        return rng.integers(1, max_seed, size=num_replicates)

    def run_msprime(self, n, seq, seed):
        ts = msprime.sim_ancestry(samples = n, sequence_length=seq, recombination_rate=1e-8, population_size=10**4, random_seed=seed)
        ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed, model="binary", discrete_genome=False)
            
        return ts

    def log_run(self, filename, info_labels, info_array):
        with open(filename, "w") as logfile:
            print(
                "\t".join(l for l in info_labels),
                file=logfile,
            )
            for line in info_array:
                print(
                    "\t".join(str(entry) for entry in line),
                    file=logfile,
                )

class TestPhenotypeSim(Test):
    def test_normal(self):
        n = 1000
        seq = 1_000_000
        num_replicates = 5
        seeds = self.get_seeds(num_replicates)
        num_causal_list = np.array([100, 500, 1000, 2000])
        for seed in tqdm(seeds, desc="Phenotype Test"):
            for num_causal in num_causal_list:
                ts = self.run_msprime(n, seq, seed)
                pheno_df, gene_df = sim_pheno.phenotype_sim(ts, num_causal=num_causal)
                G = pheno_df["Genotype"]
                meanG = np.mean(G)
                stdG = np.std(G)
                # Compare against normal distribution
                normalG = np.random.normal(loc=meanG, scale=stdG, size=n)
                self.plot_qq(
                    G, normalG, "Genotype", "Normal", f"genotype_normal_qq_n{n}", info=f"Genotype Q-Q plot, num_causal = {num_causal}"
                )
                
                pheno = pheno_df["Phenotype"]
                meanP = np.mean(pheno)
                stdP = np.std(pheno)
                
                normalP = np.random.normal(loc=meanP, scale=stdP, size=n)
                self.plot_qq(
                    pheno, normalP, "Phenotype", "Normal", f"penotype_normal_qq_n{n}", info=f"Penotype Q-Q plot, num_causal = {num_causal}"
                )
                
                n += 1


# The bottom two tests might not be necessary

class TestBeta(Test):
    def test_beta_normal(self):
        n = 0
        num_mutations = 1_000_000
        num_causal_list = np.array([100, 500, 1000, 2000])
        trait_sd_list = np.array([0.1, 0.5, 1, 2])
        for trait_sd in trait_sd_list:
            for num_causal in num_causal_list:
                rng = np.random.default_rng(np.random.randint(10000))
                mutation_id, beta = sim_pheno.choose_causal(num_mutations, num_causal, trait_sd, rng)
        
                # Compare against normal distribution
                normal = np.random.normal(loc=0, scale=trait_sd/ np.sqrt(num_causal), size=num_causal)
                
                self.plot_qq(
                    beta, normal, "Beta", "Normal", f"beta_normal_qq_n{n}", info=f"Beta Q-Q plot, num_causal = {num_causal}, trait_sd = {trait_sd}"
                )
                n += 1

class TestEnvironment(Test):
    def test_environment_normal(self):
        n = 0
        num_ind_list = np.array([1000, 5000, 10000, 20000])
        h2_list = np.array([0.01, 0.1, 0.5, 0.7, 0.99])
        for num_ind in num_ind_list:
            for h2 in h2_list:
                G = np.random.normal(size = num_ind)
                phenotype, E = sim_pheno.environment(G, h2)
                
                # Compare against normal distribution
                normal = np.random.normal(loc=0, scale=np.sqrt((1-h2)/h2 * np.var(G)), size=num_ind)
                
                self.plot_qq(
                    E, normal, "Beta", "Normal", f"environment_normal_qq_n{n}", info=f"Environment Q-Q plot, num_individual = {num_ind}, h2 = {h2}"
                )
                n += 1
                
        
        
        
    


def run_tests(suite, output_dir):
    print(f"[+] Test suite contains {len(suite)} tests.")
    for cl_name in suite:
        instance = getattr(sys.modules[__name__], cl_name)(output_dir, cl_name)
        instance._run_tests()

def main():
    parser = argparse.ArgumentParser()
    choices = [
        "TestPhenotypeSim",
        "TestBeta",
        "TestEnvironment"
    ]

    parser.add_argument(
        "--test-class",
        "-c",
        nargs="*",
        default=choices,
        choices=choices,
        help="Run all tests for specified test class",
    )

    parser.add_argument(
        "--output-dir",
        "-d",
        type=str,
        default="_output/stats_tests_output",
        help="specify the base output directory",
    )

    args = parser.parse_args()

    run_tests(args.test_class, args.output_dir)


if __name__ == "__main__":
    main()