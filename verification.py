import inspect
import itertools
import pathlib
import random
import subprocess
import sys

import matplotlib.pyplot as plt
import msprime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import tstrait
from tqdm import tqdm

"""
This code compares the tstrait simulation with AlphaSimR, which is an R
package for simulating animal breeding.
Please read the comments in data/alphasim.R for details on the
quantitative trait simulation algorithm in AlphaSimR.

This simulates a tree sequence with 1000 individuals for each combination
of mean and variance, and samples an individual at random. For that
simulated tree sequence, tstrait and AlphaSimR simulate quantitative traits
for 1000 times and extracts the simulated genetic value for the sampled
individual. Afterwards, a QQplot is generated to compare tstrait and
AlphaSimR's simulation.
"""


class Test:
    def __init__(self, basedir, cl_name):
        self.basedir = basedir
        self.name = cl_name
        self.set_output_dir(cl_name)

    def set_output_dir(self, cl_name):
        output_dir = pathlib.Path(self.basedir) / cl_name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

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

    def plot_qq_compare(self, v1, v2, x_label, y_label, filename, title=""):
        sm.qqplot_2samples(v1, v2, x_label, y_label, line="45")
        plt.title(title)
        f = self._build_filename(filename)
        plt.savefig(f, dpi=72)
        plt.close("all")

    def get_random_seed(self):
        return random.randint(1, 2**16)

    def sim_tree_seq(self, name):
        random_seed = self.get_random_seed()
        ts = msprime.sim_ancestry(
            samples=1000,
            recombination_rate=1e-8,
            sequence_length=100_000,
            population_size=10_000,
            random_seed=random_seed,
        )
        ts = msprime.sim_mutations(
            ts, rate=5e-8, random_seed=random_seed, discrete_genome=False
        )

        ts.dump(f"data/tmp/tree_sequence_{name}.trees")

        return ts


class TestAlphaSim(Test):
    """
    Compare tstrait simulation with AlphaSimR simulation.
    We must scale the tstrait simulated genetic values, as scaling is being done
    in AlphaSimR simulation.
    """

    def test_comparison(self):
        mean_array = np.array([0, 1])
        var_array = np.array([1, 4, 6])

        # We are not testing environmental noise, as they are validated through
        # tstrait's unit test. We will not be getting the same environmental
        # noise as AlphaSimR, as they are using a different algorithm to
        # simulate environmental noise (scale genetic value and then simulate
        # environmental noise)
        h2 = 0.3
        num_causal = 1000

        count = 0

        # Command for running AlphaSimR
        cmd = ["Rscript", "data/alphasim.R"]

        prod = itertools.product(mean_array, var_array)
        for element in tqdm(prod, total=6):
            mean = element[0]
            var = element[1]
            ts = self.sim_tree_seq(f"AlphaSim_{count}")
            num_causal = ts.num_sites
            # We will only examine normal distribution, as it is the supported
            # distribution in AlphaSimR
            # Gamma distribution is also supported in AlphaSimR, but scaling can
            # be challenging, as they are using a different parameter as us to
            # simulate a Gamma distribution
            model = tstrait.trait_model(distribution="normal", mean=mean, var=var)

            # Sampled individual
            ind_id = np.random.choice(ts.num_individuals)
            num_rep = 200
            genetic = np.zeros(num_rep)

            for i in range(num_rep):
                sim_result = tstrait.sim_phenotype(
                    ts=ts, num_causal=num_causal, model=model, h2=h2
                )
                # Conduct scaling, as simulated genetic values are scaled in AlphaSimR
                # These codes make sure that the simulated genetic values in tstrait will
                # have exactly the same mean and variance as the input
                tstrait_result = (
                    sim_result.phenotype["genetic_value"]
                    - np.mean(sim_result.phenotype["genetic_value"])
                ) / np.std(sim_result.phenotype["genetic_value"])
                tstrait_result = (tstrait_result + mean) * np.sqrt(var)

                genetic[i] = tstrait_result[ind_id]

            # Arguments to be fed inside AlphaSimR simulation
            # count parameter is used to make sure that we are using the same tree
            # sequence data in both simulations
            args = [
                str(mean),
                str(var),
                str(h2),
                str(num_causal),
                str(count),
                str(ind_id),
            ]
            input_cmd = cmd + args
            subprocess.check_output(input_cmd)

            alphasim_df = pd.read_csv(f"data/alphasim_result/sim_{count}.csv")
            title = f"AlphaSimR_mean_{mean}_var_{var}_ind_{ind_id}"

            self.plot_qq_compare(
                genetic,
                alphasim_df["genetic"],
                "tstrait simulation",
                "AlphaSimR simulation",
                filename=title,
                title=title,
            )

            count += 1


def run_tests(suite, output_dir):
    print(f"[+] Test suite contains {len(suite)} tests.")
    for cl_name in suite:
        instance = getattr(sys.modules[__name__], cl_name)(output_dir, cl_name)
        instance._run_tests()


if __name__ == "__main__":
    run_tests(["TestAlphaSim"], "data/stats_tests_output")
