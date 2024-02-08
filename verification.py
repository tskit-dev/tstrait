"""
Script to automate verification of tstrait against known statistical
results and benchmark programs such as AlphaSimR and simplePHENOTYPES.

We have conducted the following tests:

1. Exact tests
We simulated effect sizes and phenotypes by using AlphaSimR, simplePHENOTYPES
and the simulation framework described in ARG-Needle paper, and used the
simulated effect sizes in tstrait to simulate phenotypes, while setting the
environmental noise to be zero in all simulations. We then tested if the
simulated phenotypes in tstrait exactly match the simulated phenotypes
of external programs.

This test aims to examine whether tstrait can correctly use the genetic information
of individuals to accurately compute the genetic values. We have validated the
tstrait's output for single trait simulation and pleiotropic trait simulation.
These tests are implemented in `ExactTest` class.

2. Comparison tests
We simulated phenotypes in AlphaSimR, simplePHENOTYPES and the simulation
framework described in ARG-Needle paper by using the same parameters as the
tstrait simulation. We have simulated traits for a single individual in the
tree sequence multiple times and examined if their phenotype distributions match
by using a QQ-plot.

This test serves as an end to end testing of tstrait with environmental noise
simulation and tries to examine if the statistical properties of the
simulated traits matches the output of different simulation packages.
We have examined the tstrait output for different values of
heritability and the alpha parameter that is used in the frequency dependence
architecture. These tests are implemented in ComparisonTest.

3. Statistical tests
We have examined the statistical properties of tstrait's simulation output.
The tests in `EffectSizeDistribution` examine the statistical properties of
simulated effect sizes and the tests in `EnvironmentalNoise` examine the
simulated environmental noise.

NOTE: The properties of tstrait's simulation algorithm (such as whether it
can correctly detect mutations in a tree sequence) are validated in unit tests.

THe differences between each simulators are highlighted as the following:
1. simplePHENOTYPES
- Effect sizes can only be simulated from geometric series, so a normal
distribution must be specified if we want to simulate traits where effect
sizes are drawn from a normal distribution
- Ancestral state is set as a causal state in simplePHENOTYPES

2. AlphaSimR
- Genetic values are normalized in the simulation process

3. Simulation framework in ARG-Needle paper
- We assume that all sites are causal

These codes are largely adapted from msprime/verification.py. Please
see its documentation for usage.

"""
import argparse
import concurrent.futures
import inspect
import logging
import pathlib
import subprocess
import sys
import tempfile
from collections import namedtuple
from dataclasses import dataclass

import attr
import daiquiri
import matplotlib
import msprime
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import stdpopsim
import tqdm
import tstrait
from matplotlib import pyplot

matplotlib.use("Agg")
import statsmodels.api as sm  # noqa: E402


def sample_ts():
    ts = msprime.sim_ancestry(10_000, sequence_length=100_000, random_seed=1)
    ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
    return ts


@attr.s
class Test:
    """
    The superclass of all tests. The only attribute defined is the output
    directory for the test, which is guaranteed to exist when the
    test method is called.
    """

    output_dir = attr.ib(type=str, default=None)

    def _build_filename(self, *args):
        return self.output_dir / ("_".join(args) + ".png")

    def _plot_qq_dist(
        self, data, data_name, dist, dist_name, distargs=(), loc=0, scale=1
    ):
        sm.qqplot(data, dist=dist, distargs=distargs, loc=loc, scale=scale, line="45")
        f = self._build_filename(data_name, dist_name, "qqplot")
        pyplot.title(f"data={data_name}, dist={dist_name}")
        pyplot.savefig(f, dpi=72)
        pyplot.close("all")

        sns.kdeplot(data, color="b", fill=True, legend=False)
        pyplot.xlabel(data_name)
        f = self._build_filename(data_name, dist_name, "histogram")
        pyplot.savefig(f, dpi=72)
        pyplot.close("all")

    def _plot_qq_compare(self, data1, data1_name, data2, data2_name):
        sm.qqplot_2samples(
            data1, data2, xlabel=data1_name, ylabel=data2_name, line="45"
        )
        f = self._build_filename("compare", data1_name, data2_name)
        pyplot.title(f"compare_{data1_name}_{data2_name}")
        pyplot.savefig(f, dpi=72)
        pyplot.close("all")

        sns.kdeplot(data1, color="b", fill=True, legend=False, label=data1_name)
        sns.kdeplot(data2, color="r", fill=True, legend=False, label=data2_name)
        pyplot.legend()
        f = self._build_filename(data1_name, data2_name, "density_histogram")
        pyplot.savefig(f, dpi=72)
        pyplot.close("all")


@dataclass
class SimulationResult:
    """
    Dataclass that contains simulated effect sizes and phenotypes.

    Attributes
    ----------
    trait : pandas.DataFrame
        Trait dataframe.
    phenotype : pandas.DataFrame
        Phenotype dataframe.
    """

    phenotype: pd.DataFrame
    trait: pd.DataFrame


def _simulate_simplePHENOTYPES_qqplot(
    ts, num_causal, h2, random_seed, num_rep, mean=0, var=1
):
    """
    The input of this function is a tree sequence and the output is a phenotype
    dataframe. This function is used for comparing the simulation output in
    a QQ-plot.
    The output of this function will be a phenotype dataframe.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(f"{tmpdirname}/tree_seq.vcf", "w") as vcf_file:
            ts.write_vcf(
                vcf_file, individual_names=np.arange(ts.num_individuals).astype(str)
            )
        cmd = ["Rscript", "scripts/simulate_simplePHENOTYPES_qqplot.R"]
        args = [
            str(num_causal),
            str(h2),
            tmpdirname,
            str(num_rep),
            str(mean),
            str(var),
            str(random_seed),
        ]
        input_cmd = cmd + args
        subprocess.check_output(input_cmd)

        phenotype_df = pd.read_csv(f"{tmpdirname}/simplePHENOTYPES.csv")

    return phenotype_df


def _simulate_simplePHENOTYPES_exact(
    ts, num_causal, random_seed, add_effect=1, num_trait=1, add_effect_2=1
):
    """
    The function to simulate quantitative traits by using simplePHENOTYPES.
    We will specify the number of causal sites and the parameter for the
    geometric series where the effect sizes are determined.
    The output of this function will be a SimulationResult object.
    """

    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(f"{tmpdirname}/tree_seq.vcf", "w") as vcf_file:
            ts.write_vcf(
                vcf_file, individual_names=np.arange(ts.num_individuals).astype(str)
            )
        cmd = ["Rscript", "scripts/simulate_simplePHENOTYPES_exact.R"]
        args = [
            str(num_causal),
            str(num_trait),
            str(add_effect),
            str(add_effect_2),
            tmpdirname,
            str(random_seed),
        ]
        input_cmd = cmd + args
        subprocess.check_output(input_cmd)

        qtn_df = pd.read_csv(f"{tmpdirname}/Additive_Selected_QTNs.txt", sep="\t")

        if num_trait == 1:
            phenotype_df = pd.read_csv(
                f"{tmpdirname}/Simulated_Data_1_Reps_Herit_1.txt", sep="\t"
            )
            del phenotype_df["reps"]
            phenotype_df = phenotype_df.rename(
                columns={"<Trait>": "individual_id", "Pheno": "phenotype"}
            )
        else:
            phenotype_df = pd.read_csv(
                f"{tmpdirname}/Simulated_Data_1_Reps_Herit_1_1.txt", sep="\t"
            )
            del phenotype_df["Rep"]
            phenotype_df = pd.melt(
                phenotype_df,
                value_vars=["Trait_1_H2_1", "Trait_2_H2_1"],
                id_vars=["<Trait>"],
            )
            phenotype_df = phenotype_df.rename(
                columns={
                    "<Trait>": "individual_id",
                    "value": "phenotype",
                    "variable": "trait_id",
                }
            )
            phenotype_df = phenotype_df.replace({"Trait_1_H2_1": 0, "Trait_2_H2_1": 1})

    # Obtain the list of causal allele
    causal_allele = []
    effect_size = []
    effect_size_2 = []
    for i, site_id in enumerate(qtn_df["snp"].values, start=1):
        # simplePHENOTYPES uses ancestral state as a causal allele
        allele = ts.site(site_id).ancestral_state
        causal_allele.append(allele)
        effect_size.append(add_effect**i)
        effect_size_2.append(add_effect_2**i)

    if num_trait == 2:
        effect_size = np.append(effect_size, effect_size_2)

    trait_df = pd.DataFrame(
        {
            "site_id": np.tile(qtn_df["snp"].values, num_trait),
            "causal_allele": np.tile(causal_allele, num_trait),
            "effect_size": effect_size,
            "trait_id": np.repeat(np.arange(num_trait), len(causal_allele)),
        }
    )

    simulation_result = SimulationResult(phenotype=phenotype_df, trait=trait_df)

    return simulation_result


def _simulate_AlphaSimR(
    ts, num_causal, h2, random_seed, num_rep=1, h2_2=1, corA=1, num_trait=1
):
    """
    The function to simulate quantitative traits by using AlphaSimR. We will
    specify the number of causal sites, such that the AlphaSimR simulation
    will be conducted randomly. corA is used to specify the correlation
    coefficient of pleiotropic traits. The output of this function is a
    PhenotypeResult object.
    """

    with tempfile.TemporaryDirectory() as tmpdirname:
        ts.dump(f"{tmpdirname}/tree_seq.tree")
        cmd = ["Rscript", "scripts/simulate_AlphaSimR.R"]
        args = [
            str(num_causal),
            tmpdirname,
            str(corA),
            str(num_trait),
            str(h2),
            str(h2_2),
            str(num_rep),
            str(random_seed),
        ]
        input_cmd = cmd + args
        subprocess.check_output(input_cmd)

        phenotype_df = pd.read_csv(f"{tmpdirname}/phenotype_alphasimr.csv")
        trait_df = pd.read_csv(f"{tmpdirname}/trait_alphasimr.csv")

    # Obtain the list of causal allele
    causal_allele = []
    for site_id in trait_df["site_id"]:
        allele = ts.mutation(site_id).derived_state
        causal_allele.append(allele)

    trait_df["causal_allele"] = causal_allele

    simulation_result = SimulationResult(phenotype=phenotype_df, trait=trait_df)

    return simulation_result


def _simulate_arg_needle(ts, alpha, h2, random_seed, num_rep=1):
    """
    Simulate phenotypes by using the quantitative trait simulation framework that is
    adapted from the ARG-Needle paper.
    https://zenodo.org/records/7745746
    """

    rng = np.random.default_rng(random_seed)
    num_ind = ts.num_individuals

    phenotype_result = pd.DataFrame({"individual_id": [], "phenotype": []})
    trait_result = pd.DataFrame(
        {"site_id": [], "causal_allele": [], "effect_size": [], "trait_id": []}
    )

    for _ in range(num_rep):
        phenotypes = np.zeros(num_ind)
        beta_list = []
        causal_allele = []

        for variant in ts.variants():
            row = variant.genotypes.astype("float64")
            row = row.reshape((num_ind, 2)).sum(axis=-1)
            # This uses std instead of \sqrt{2p(1-p)} that we are using, but since we
            # are using an infinite-sites model, the standard deviation will
            # approximately be \sqrt{2p(1-p)} when the sample size is large.
            std = np.std(row, ddof=1)
            beta = rng.normal()
            causal_allele.append(variant.alleles[1])
            phenotypes += row * (beta * std**alpha)
            beta_list.append(beta * std**alpha)

        # normalize pheno to have mean 0 and var = h2
        phenotypes -= np.mean(phenotypes)
        phenotypes /= np.std(phenotypes, ddof=1)
        phenotypes *= np.sqrt(h2)

        # sample environmental component with variance 1-h2 and add it to phenotype
        phenotypes += rng.normal(size=num_ind) * np.sqrt(1 - h2)

        # normalize it all
        phenotypes -= np.mean(phenotypes)
        phenotypes /= np.std(phenotypes, ddof=1)

        phenotype_df = pd.DataFrame(
            {"individual_id": np.arange(len(phenotypes)), "phenotype": phenotypes}
        )
        trait_df = pd.DataFrame(
            {
                "site_id": np.arange(len(causal_allele)),
                "causal_allele": causal_allele,
                "effect_size": beta_list,
                "trait_id": np.zeros(len(causal_allele)),
            }
        )

        phenotype_result = pd.concat(
            [phenotype_result, phenotype_df], ignore_index=True
        )
        trait_result = pd.concat([trait_result, trait_df], ignore_index=True)

    trait_result = trait_result.astype({"site_id": int})
    simulation_result = SimulationResult(phenotype=phenotype_result, trait=trait_result)

    return simulation_result


class ExactTest(Test):
    """
    Test class that conducts exact test. Phenotype and trait information are
    simulated by using an external simulator. Afterwards, the trait information is used
    to compute the genetic values by using tstrait, and their values are compared
    to examine if the tstrait package can compute the accurate genetic values.

    The numericalization input is used to modify the numericalization of genotypes.
    The standard numericalization of tstrait is (AA=2, Aa=1, aa=0), where A is the
    causal allele, but when numericalization is True, it will be (AA=1, Aa=0, aa=-1).
    """

    def _simulate_tree_seq(self, random_seed):
        """
        Simulates a tree sequence from an infinite-sites model.
        """
        ts = msprime.sim_ancestry(
            samples=100,
            recombination_rate=1e-8,
            sequence_length=100_000,
            population_size=10_000,
            random_seed=random_seed,
        )
        ts = msprime.sim_mutations(
            ts, rate=1e-8, random_seed=random_seed, discrete_genome=False
        )

        return ts

    def _compute_tstrait(self, ts, trait_df, numericalization=False):
        """
        This method computes genetic values from the trait dataframe.
        There is an option to change the numericalization of genetic values.
        No other post-processing is conducted to the computed genetic values.
        """
        trait_df = trait_df.sort_values(by=["site_id"])
        genetic_df = tstrait.genetic_value(ts, trait_df)
        if numericalization:
            trait_id_array = trait_df["trait_id"].unique()
            for trait_id in trait_id_array:
                effect_size_sum = trait_df[trait_df["trait_id"] == trait_id][
                    "effect_size"
                ].sum()
                genetic_df.loc[
                    genetic_df["trait_id"] == trait_id, ["genetic_value"]
                ] += effect_size_sum
        phenotype_df = tstrait.sim_env(genetic_df, h2=1, random_seed=1)

        return phenotype_df


class ExactTestSimplePHENOTYPES(ExactTest):
    """
    Exact test between simplePHENOTYPES and tstrait.
    """

    def _compare_simplePHENOTYPES(
        self,
        num_causal,
        random_seed,
        add_effect=1,
        num_trait=1,
        add_effect_2=1,
        numericalization=False,
    ):
        ts = self._simulate_tree_seq(random_seed)
        simulation_output = _simulate_simplePHENOTYPES_exact(
            ts=ts,
            num_causal=num_causal,
            random_seed=random_seed,
            add_effect=add_effect,
            add_effect_2=add_effect_2,
            num_trait=num_trait,
        )
        phenotype_df = self._compute_tstrait(
            ts=ts, trait_df=simulation_output.trait, numericalization=numericalization
        )
        grouped = phenotype_df.groupby("trait_id")[["phenotype"]]
        phenotype_df = grouped.transform(lambda x: (x - x.mean()))

        np.testing.assert_array_almost_equal(
            phenotype_df["phenotype"].values,
            simulation_output.phenotype["phenotype"].values,
        )

    def test_exact_simplePHENOTYPES_single_causal_30_add_09(self):
        self._compare_simplePHENOTYPES(
            num_causal=30,
            add_effect=0.9,
            random_seed=100,
            num_trait=1,
            numericalization=False,
        )

    def test_exact_simplePHENOTYPES_single_causal_50_add_09(self):
        self._compare_simplePHENOTYPES(
            num_causal=50,
            add_effect=0.9,
            random_seed=101,
            num_trait=1,
            numericalization=False,
        )

    def test_exact_simplePHENOTYPES_single_causal_30_add_11(self):
        self._compare_simplePHENOTYPES(
            num_causal=30,
            add_effect=1.1,
            random_seed=102,
            num_trait=1,
            numericalization=False,
        )

    def test_exact_simplePHENOTYPES_single_causal_50_add_11(self):
        self._compare_simplePHENOTYPES(
            num_causal=50,
            add_effect=1.1,
            random_seed=103,
            num_trait=1,
            numericalization=False,
        )

    def test_exact_simplePHENOTYPES_single_causal_30_add_09_numer(self):
        self._compare_simplePHENOTYPES(
            num_causal=30,
            add_effect=0.9,
            random_seed=104,
            num_trait=1,
            numericalization=True,
        )

    def test_exact_simplePHENOTYPES_single_causal_30_add_11_numer(self):
        self._compare_simplePHENOTYPES(
            num_causal=30,
            add_effect=1.1,
            random_seed=106,
            num_trait=1,
            numericalization=True,
        )

    def test_exact_simplePHENOTYPES_single_causal_50_add_11_numer(self):
        self._compare_simplePHENOTYPES(
            num_causal=50,
            add_effect=1.1,
            random_seed=107,
            num_trait=1,
            numericalization=True,
        )

    def test_exact_simplePHENOTYPES_pleio_causal_30_add_09_08(self):
        self._compare_simplePHENOTYPES(
            num_causal=30,
            add_effect=0.9,
            random_seed=108,
            num_trait=2,
            add_effect_2=0.8,
            numericalization=False,
        )

    def test_exact_simplePHENOTYPES_pleio_causal_50_add_11_08(self):
        self._compare_simplePHENOTYPES(
            num_causal=50,
            add_effect=1.1,
            random_seed=110,
            num_trait=2,
            add_effect_2=0.8,
            numericalization=False,
        )

    def test_exact_simplePHENOTYPES_pleio_causal_30_add_08_08(self):
        self._compare_simplePHENOTYPES(
            num_causal=30,
            add_effect=0.8,
            random_seed=111,
            num_trait=2,
            add_effect_2=0.8,
            numericalization=False,
        )

    def test_exact_simplePHENOTYPES_pleio_causal_50_add_09_08_numer(self):
        self._compare_simplePHENOTYPES(
            num_causal=50,
            add_effect=0.9,
            random_seed=112,
            num_trait=2,
            add_effect_2=0.8,
            numericalization=True,
        )

    def test_exact_simplePHENOTYPES_pleio_causal_30_add_08_08_numer(self):
        self._compare_simplePHENOTYPES(
            num_causal=30,
            add_effect=0.8,
            random_seed=113,
            num_trait=2,
            add_effect_2=0.8,
            numericalization=True,
        )


class ExactTestAlphaSimR(ExactTest):
    """
    Exact test between AlphaSimR and tstrait.
    """

    def _compare_AlphaSimR(
        self, num_causal, random_seed, corA=1, num_trait=1, numericalization=False
    ):
        ts = self._simulate_tree_seq(random_seed)
        simulation_output = _simulate_AlphaSimR(
            ts=ts,
            num_causal=num_causal,
            h2=1,
            random_seed=random_seed,
            corA=corA,
            num_trait=num_trait,
        )
        phenotype_df = self._compute_tstrait(
            ts=ts, trait_df=simulation_output.trait, numericalization=numericalization
        )
        phenotype_df = tstrait.normalise_phenotypes(phenotype_df, mean=0, var=1, ddof=0)

        np.testing.assert_array_almost_equal(
            phenotype_df["phenotype"].values,
            simulation_output.phenotype["phenotype"].values,
        )

    def test_exact_AlphaSimR_single_causal_30(self):
        self._compare_AlphaSimR(num_causal=30, random_seed=200, numericalization=False)

    def test_exact_AlphaSimR_single_causal_100(self):
        self._compare_AlphaSimR(num_causal=100, random_seed=201, numericalization=False)

    def test_exact_AlphaSimR_single_causal_30_numer(self):
        self._compare_AlphaSimR(num_causal=30, random_seed=202, numericalization=True)

    def test_exact_AlphaSimR_single_causal_100_numer(self):
        self._compare_AlphaSimR(num_causal=100, random_seed=203, numericalization=True)

    def test_exact_AlphaSimR_pleiotropic_causal_30_cor08(self):
        self._compare_AlphaSimR(
            num_causal=30,
            random_seed=204,
            corA=0.8,
            num_trait=2,
            numericalization=False,
        )

    def test_exact_AlphaSimR_pleiotropic_causal_30_cor01(self):
        self._compare_AlphaSimR(
            num_causal=30,
            random_seed=205,
            corA=0.1,
            num_trait=2,
            numericalization=False,
        )

    def test_exact_AlphaSimR_pleiotropic_causal_100_cor01(self):
        self._compare_AlphaSimR(
            num_causal=100,
            random_seed=206,
            corA=0.1,
            num_trait=2,
            numericalization=False,
        )

    def test_exact_AlphaSimR_pleiotropic_causal_100_cor08(self):
        self._compare_AlphaSimR(
            num_causal=100,
            random_seed=207,
            corA=0.8,
            num_trait=2,
            numericalization=False,
        )

    def test_exact_AlphaSimR_pleiotropic_causal_30_cor08_numer(self):
        self._compare_AlphaSimR(
            num_causal=30, random_seed=208, corA=0.8, num_trait=2, numericalization=True
        )

    def test_exact_AlphaSimR_pleiotropic_causal_30_cor01_numer(self):
        self._compare_AlphaSimR(
            num_causal=30, random_seed=209, corA=0.1, num_trait=2, numericalization=True
        )


class ExactTestARGNeedle(ExactTest):
    """
    Exact test between the simulation framework described in the
    ARG-Needle paper and tstrait.
    """

    def _compare_arg_needle(self, alpha, random_seed, numericalization=False):
        ts = self._simulate_tree_seq(random_seed)
        simulation_output = _simulate_arg_needle(
            ts=ts, alpha=alpha, h2=1, random_seed=random_seed
        )
        phenotype_df = self._compute_tstrait(
            ts=ts, trait_df=simulation_output.trait, numericalization=numericalization
        )
        phenotype_df = tstrait.normalise_phenotypes(phenotype_df, mean=0, var=1)

        np.testing.assert_array_almost_equal(
            phenotype_df["phenotype"].values,
            simulation_output.phenotype["phenotype"].values,
        )

    def test_exact_ARG_Needle_alpha_0(self):
        self._compare_arg_needle(alpha=0, random_seed=300, numericalization=False)

    def test_exact_ARG_Needle_alpha_negative_1(self):
        self._compare_arg_needle(alpha=-1, random_seed=301, numericalization=False)

    def test_exact_ARG_Needle_alpha_1(self):
        self._compare_arg_needle(alpha=1, random_seed=302, numericalization=False)

    def test_exact_ARG_Needle_alpha_0_numer(self):
        self._compare_arg_needle(alpha=0, random_seed=303, numericalization=True)

    def test_exact_ARG_Needle_alpha_negative_1_numer(self):
        self._compare_arg_needle(alpha=-1, random_seed=304, numericalization=True)

    def test_exact_ARG_Needle_alpha_1_numer(self):
        self._compare_arg_needle(alpha=1, random_seed=305, numericalization=True)


class ComparisonTest(Test):
    """
    Compare tstrait against currently available simulators and create a QQ-plot. We
    assume an infinite sites model and we also assume that all sites are causal.
    """

    def _compare_phenotype(self, data1, data1_name, data2, data2_name, ind_id):
        """
        The input of this function should be a pandas dataframe with individual_id and
        phenotype columns. The values inside the individual_id column must be a number
        and match with the 2 dataframes that are provided inside this function.
        `ind_id` is a list of individual IDs that are of interest to compare.
        """

        for i in ind_id:
            data1_phenotype = data1.loc[data1["individual_id"] == i]["phenotype"].values
            data2_phenotype = data2.loc[data2["individual_id"] == i]["phenotype"].values
            self._plot_qq_compare(
                data1=data1_phenotype,
                data1_name=f"{data1_name}_ind_{i}",
                data2=data2_phenotype,
                data2_name=f"{data2_name}_ind_{i}",
            )

    def _simulate_tstrait(self, ts, num_rep, h2, random_seed, mean=0, var=1, alpha=0):
        """
        Simulates trait and phenotype information based on a simulated tree
        sequence and return the simulated phenotypes that are normalised.
        We assume that all sites are causal.
        """

        model = tstrait.trait_model(distribution="normal", mean=mean, var=var)
        phenotype_df_list = []
        for i in range(num_rep):
            sim_result = tstrait.sim_phenotype(
                ts,
                num_causal=ts.num_sites,
                alpha=alpha,
                model=model,
                h2=h2,
                random_seed=random_seed + i,
            )
            phenotype_df = tstrait.normalise_phenotypes(sim_result.phenotype)
            phenotype_df_list.append(phenotype_df)

        phenotype_result = pd.concat(phenotype_df_list).reset_index(drop=True)

        return phenotype_result

    def _simulate_out_of_africa(self, random_seed):
        """
        Simulate a tree sequence data from the Out of Africa model and select individuals
        from different populations.
        """
        rng = np.random.default_rng(random_seed)
        species = stdpopsim.get_species("HomSap")
        model = species.get_demographic_model("OutOfAfrica_3G09")
        contig = species.get_contig("chr22", mutation_rate=0, length_multiplier=1e-3)
        samples = {"YRI": 300, "CHB": 300, "CEU": 300}
        engine = stdpopsim.get_engine("msprime")
        ts = engine.simulate(model, contig, samples, seed=random_seed)
        ts = msprime.sim_mutations(
            ts, rate=model.mutation_rate, random_seed=random_seed, discrete_genome=False
        )
        node0 = rng.choice(ts.samples(population=0))
        node1 = rng.choice(ts.samples(population=1))
        node2 = rng.choice(ts.samples(population=2))
        ind_id = np.array(
            [
                ts.node(node0).individual,
                ts.node(node1).individual,
                ts.node(node2).individual,
            ]
        )
        return ts, ind_id


class ComparisonTestSimplePHENOTYPES(ComparisonTest):
    def _qqplot_simplePHENOTYPES(
        self, num_rep, h2, random_seed, mean=0, var=1, alpha=0
    ):
        ts, ind_id = self._simulate_out_of_africa(random_seed=random_seed)
        tstrait_phenotype = self._simulate_tstrait(
            ts=ts,
            num_rep=num_rep,
            h2=h2,
            random_seed=random_seed,
            mean=mean,
            var=var,
            alpha=alpha,
        )
        # - mean because simplePHENOTYPES uses ancestral state as a causal allele
        simplePHENOTYPES_phenotype = _simulate_simplePHENOTYPES_qqplot(
            ts=ts,
            num_causal=ts.num_sites,
            h2=h2,
            random_seed=random_seed,
            num_rep=num_rep,
            mean=-mean,
            var=var,
        )
        self._compare_phenotype(
            data1=tstrait_phenotype,
            data1_name="tstrait",
            data2=simplePHENOTYPES_phenotype,
            data2_name="simplePHENOTYPES",
            ind_id=ind_id,
        )

    def test_compare_simplePHENOTYPES_h2_08_mean_0_var_1(self):
        self._qqplot_simplePHENOTYPES(
            num_rep=500,
            h2=0.8,
            random_seed=1000,
            mean=0,
            var=1,
        )

    def test_compare_simplePHENOTYPES_h2_09_mean_1_var_4(self):
        self._qqplot_simplePHENOTYPES(
            num_rep=500,
            h2=0.9,
            random_seed=1001,
            mean=1,
            var=4,
        )

    def test_compare_simplePHENOTYPES_h2_02_mean_negative_1_var_4(self):
        self._qqplot_simplePHENOTYPES(
            num_rep=500,
            h2=0.2,
            random_seed=1002,
            mean=-1,
            var=4,
        )

    def test_compare_simplePHENOTYPES_h2_01_mean_0_var_4(self):
        self._qqplot_simplePHENOTYPES(
            num_rep=500,
            h2=0.1,
            random_seed=1001,
            mean=0,
            var=4,
        )


class ComparisonTestAlphaSimR(ComparisonTest):
    def _simulate_tstrait_genetic_normalise(self, ts, num_rep, h2, random_seed):
        """
        Simulates trait and phenotype information based on a simulated tree
        sequence and return the simulated phenotypes that are normalised.
        We assume that all sites are causal.

        AlphaSimR simulates effect sizes from a standard normal distribution
        and normalizes the genetic value afterwards.
        """

        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        phenotype_df_list = []
        for i in range(num_rep):
            trait_df = tstrait.sim_trait(
                ts=ts, num_causal=ts.num_sites, model=model, random_seed=random_seed + i
            )
            genetic_df = tstrait.genetic_value(ts, trait_df)
            normalised_df = tstrait.normalise_genetic_value(genetic_df)

            phenotype_df = tstrait.sim_env(
                normalised_df, h2=h2, random_seed=random_seed + i
            )

            phenotype_df_list.append(phenotype_df)

        phenotype_result = pd.concat(phenotype_df_list).reset_index(drop=True)

        return phenotype_result

    def _qqplot_AlphaSimR(self, num_rep, h2, random_seed):
        ts, ind_id = self._simulate_out_of_africa(random_seed=random_seed)
        tstrait_phenotype = self._simulate_tstrait_genetic_normalise(
            ts=ts, num_rep=num_rep, h2=h2, random_seed=random_seed
        )
        alphasimr_phenotype = _simulate_AlphaSimR(
            ts=ts,
            num_causal=ts.num_sites,
            h2=h2,
            random_seed=random_seed,
            num_rep=num_rep,
        )
        self._compare_phenotype(
            data1=tstrait_phenotype,
            data1_name="tstrait",
            data2=alphasimr_phenotype.phenotype,
            data2_name="AlphaSimR",
            ind_id=ind_id,
        )

    def test_compare_AlphaSimR_h2_08(self):
        self._qqplot_AlphaSimR(num_rep=500, random_seed=2000, h2=0.8)

    def test_compare_AlphaSimR_h2_02(self):
        self._qqplot_AlphaSimR(num_rep=500, random_seed=2001, h2=0.2)

    def test_compare_AlphaSimR_h2_05(self):
        self._qqplot_AlphaSimR(num_rep=500, random_seed=2000, h2=0.5)


class ComparisonTestArgNeedle(ComparisonTest):
    def _qqplot_arg_needle(self, num_rep, h2, alpha, random_seed):
        ts, ind_id = self._simulate_out_of_africa(random_seed=random_seed)
        tstrait_phenotype = self._simulate_tstrait(
            ts=ts, num_rep=num_rep, h2=h2, random_seed=random_seed, alpha=alpha
        )
        argneedle_phenotype = _simulate_arg_needle(
            ts=ts, alpha=alpha, h2=h2, random_seed=random_seed, num_rep=num_rep
        )
        self._compare_phenotype(
            data1=tstrait_phenotype,
            data1_name="tstrait",
            data2=argneedle_phenotype.phenotype,
            data2_name="ARG-Needle",
            ind_id=ind_id,
        )

    def test_compare_arg_needle_alpha_0_h2_08(self):
        self._qqplot_arg_needle(num_rep=500, h2=0.8, random_seed=3000, alpha=0)

    def test_compare_arg_needle_alpha_negative_1_h2_08(self):
        self._qqplot_arg_needle(num_rep=500, h2=0.8, random_seed=3001, alpha=-1)

    def test_compare_arg_needle_alpha_positive_1_h2_09(self):
        self._qqplot_arg_needle(num_rep=500, h2=0.9, random_seed=3002, alpha=1)

    def test_compare_arg_needle_alpha_positive_2_h2_01(self):
        self._qqplot_arg_needle(num_rep=500, h2=0.1, random_seed=3003, alpha=2)

    def test_compare_arg_needle_alpha_negative_2_h2_01(self):
        self._qqplot_arg_needle(num_rep=500, h2=0.1, random_seed=3004, alpha=-2)

    def test_compare_arg_needle_alpha_0_h2_02(self):
        self._qqplot_arg_needle(num_rep=500, h2=0.2, random_seed=3005, alpha=0)

    def test_compare_arg_needle_alpha_negative_1_h2_05(self):
        self._qqplot_arg_needle(num_rep=500, h2=0.5, random_seed=3006, alpha=-1)

    def test_compare_arg_needle_alpha_positive_10_h2_09(self):
        self._qqplot_arg_needle(num_rep=500, h2=0.9, random_seed=3007, alpha=10)

    def test_compare_arg_needle_alpha_negative_10_h2_08(self):
        self._qqplot_arg_needle(num_rep=500, h2=0.8, random_seed=3008, alpha=-10)


def model_list(loc, scale):
    df = 10
    shape = 5
    Distribution = namedtuple(
        "Distribution",
        [
            "trait_model",
            "scipy_distribution",
            "loc",
            "scale",
            "distargs",
            "distribution_name",
        ],
    )
    distr = [
        Distribution(
            trait_model=tstrait.trait_model(
                distribution="normal", mean=loc, var=scale**2
            ),
            scipy_distribution=scipy.stats.norm,
            loc=loc,
            scale=scale,
            distargs=(),
            distribution_name=f"normal_mean_{loc}_var_{scale**2}",
        ),
        Distribution(
            trait_model=tstrait.trait_model(distribution="exponential", scale=scale),
            scipy_distribution=scipy.stats.expon,
            loc=0,
            scale=scale,
            distargs=(),
            distribution_name=f"exponential_scale_{scale}",
        ),
        Distribution(
            trait_model=tstrait.trait_model(
                distribution="t", mean=loc, var=scale**2, df=df
            ),
            scipy_distribution=scipy.stats.t,
            loc=loc,
            scale=scale,
            distargs=(df,),
            distribution_name=f"t_mean_{loc}_var_{scale**2}_df_{df}",
        ),
        Distribution(
            trait_model=tstrait.trait_model(
                distribution="gamma", shape=shape, scale=scale
            ),
            scipy_distribution=scipy.stats.gamma,
            loc=0,
            scale=scale,
            distargs=(shape,),
            distribution_name=f"gamma_shape_{shape}_scale_{scale}",
        ),
    ]

    return distr


class EffectSizeDistribution(Test):
    """
    Examine the statistical properties of simulated effect sizes.
    """

    def _simulate_effect_size(self, model):
        sim_result = tstrait.sim_trait(
            ts=sample_ts(), num_causal=1000, model=model, random_seed=100
        )
        return sim_result["effect_size"].values

    def test_univariate(self):
        """
        Test the statistical properties of univariate distribution.
        """
        loc = 4
        scale = 10
        models = model_list(loc=loc, scale=scale)
        stats_type = "simulated_effect_size"
        for model in models:
            effect_size = self._simulate_effect_size(model.trait_model)
            self._plot_qq_dist(
                data=effect_size,
                data_name=stats_type,
                dist=model.scipy_distribution,
                dist_name=model.distribution_name,
                loc=model.loc,
                scale=model.scale,
                distargs=model.distargs,
            )

    def test_random_sign_exponential(self):
        scale = 5
        model1 = tstrait.trait_model(
            distribution="exponential", scale=scale, random_sign=True
        )
        effect_size_1 = self._simulate_effect_size(model=model1)
        effect_size_1 = np.abs(effect_size_1)

        model2 = tstrait.trait_model(
            distribution="exponential", scale=scale, random_sign=False
        )
        effect_size_2 = self._simulate_effect_size(model=model2)

        self._plot_qq_compare(
            data1=effect_size_1,
            data1_name="exponential_random_sign_True",
            data2=effect_size_2,
            data2_name="exponential_random_sign_False",
        )

    def test_random_sign_gamma(self):
        shape = 5
        scale = 3
        model1 = tstrait.trait_model(
            distribution="gamma", shape=shape, scale=scale, random_sign=True
        )
        effect_size_1 = self._simulate_effect_size(model=model1)
        effect_size_1 = np.abs(effect_size_1)

        model2 = tstrait.trait_model(
            distribution="gamma", shape=shape, scale=scale, random_sign=False
        )
        effect_size_2 = self._simulate_effect_size(model=model2)
        effect_size_2 = np.abs(effect_size_2)

        self._plot_qq_compare(
            data1=effect_size_1,
            data1_name="gamma_random_sign_True",
            data2=effect_size_2,
            data2_name="gamma_random_sign_False",
        )

    def test_multivariate(self):
        """
        Test the statistical properties of "multi_normal" distribution.
        """
        np.random.seed(20)
        n = 3
        mean = np.random.randn(n)
        M = np.random.randn(n, n)
        cov = np.dot(M, M.T)
        num_causal = 2000
        model = tstrait.trait_model(distribution="multi_normal", mean=mean, cov=cov)
        sim_result = tstrait.sim_trait(
            ts=sample_ts(), num_causal=num_causal, model=model, random_seed=100
        )

        const = np.random.randn(n)
        data_val = np.matmul(const, cov)
        data_sd = np.sqrt(np.matmul(data_val, const))
        sum_data = np.zeros(num_causal)

        for i in range(n):
            df = sim_result.loc[sim_result.trait_id == i]
            stats_type = f"multivariate_normal_coordinate_{i}"
            self._plot_qq_dist(
                data=df["effect_size"].values,
                data_name=stats_type,
                dist=scipy.stats.norm,
                dist_name=f"normal_mean_{mean[i]}_var_{cov[i,i]}",
                loc=mean[i],
                scale=np.sqrt(cov[i, i]),
            )

            sum_data += df["effect_size"].values * const[i]

        stats_type = "multivariate_normal_sum"
        self._plot_qq_dist(
            data=sum_data,
            data_name=stats_type,
            dist=scipy.stats.norm,
            dist_name=f"normal_mean_{np.matmul(const, mean)}_sd_{data_sd}",
            loc=np.matmul(const, mean),
            scale=data_sd,
        )


class EnvironmentalNoise(Test):
    """
    Examine the statistical properties of simulated environmental noise.
    """

    def _compute_std(self, genetic_value, h2):
        env_std = np.sqrt((1 - h2) / h2 * np.var(genetic_value))
        return env_std

    def test_environmental_univariate(self):
        trait_model = tstrait.trait_model(distribution="normal", mean=2, var=5)
        h2 = 0.3
        sim_result = tstrait.sim_phenotype(
            ts=sample_ts(), num_causal=1000, model=trait_model, h2=h2, random_seed=1
        )
        phenotype_df = sim_result.phenotype
        env_std = self._compute_std(phenotype_df["genetic_value"].values, h2=h2)
        self._plot_qq_dist(
            data=phenotype_df["environmental_noise"].values,
            data_name="environmental_noise_univariate",
            dist=scipy.stats.norm,
            dist_name=f"normal_mean_0_sd_{env_std}",
            loc=0,
            scale=env_std,
        )

    def test_environmental_multivariate(self):
        np.random.seed(10)
        n = 3
        mean = np.random.randn(n)
        M = np.random.randn(n, n)
        cov = np.dot(M, M.T)
        h2 = [0.1, 0.5, 0.9]

        trait_model = tstrait.trait_model(
            distribution="multi_normal", mean=mean, cov=cov
        )
        ts = sample_ts()
        sim_result = tstrait.sim_phenotype(
            ts=ts, num_causal=1000, model=trait_model, h2=h2, random_seed=1
        )
        phenotype_df = sim_result.phenotype

        const = np.random.randn(n)
        sd_array = np.zeros(n)

        sum_data = np.zeros(ts.num_individuals)

        for i in range(n):
            df = phenotype_df.loc[phenotype_df.trait_id == i]
            env_std = self._compute_std(df["genetic_value"].values, h2=h2[i])
            self._plot_qq_dist(
                data=df["environmental_noise"].values,
                data_name=f"environmental_noise_component_{i}",
                dist=scipy.stats.norm,
                dist_name=f"normal_mean_0_sd_{env_std}",
                loc=0,
                scale=env_std,
            )
            sum_data += df["environmental_noise"].values * const[i]
            sd_array[i] = env_std

        sd = np.dot(sd_array**2, const**2)
        sd = np.sqrt(sd)

        self._plot_qq_dist(
            data=sum_data,
            data_name="environmental_noise_multivariate",
            dist=scipy.stats.norm,
            dist_name=f"normal_mean_0_sd_{sd}",
            loc=0,
            scale=sd,
        )

    def test_environmental_multiple_singleh2(self):
        np.random.seed(10)
        n = 3
        mean = np.random.randn(n)
        M = np.random.randn(n, n)
        cov = np.dot(M, M.T)

        h2 = 0.3
        ts = sample_ts()

        trait_model = tstrait.trait_model(
            distribution="multi_normal", mean=mean, cov=cov
        )

        sim_result = tstrait.sim_phenotype(
            ts=ts, num_causal=1000, model=trait_model, h2=h2, random_seed=1
        )
        phenotype_df = sim_result.phenotype

        const = np.random.randn(n)
        sd_array = np.zeros(n)

        sum_data = np.zeros(ts.num_individuals)

        for i in range(n):
            df = phenotype_df.loc[phenotype_df.trait_id == i]
            env_std = self._compute_std(df["genetic_value"].values, h2=h2)
            self._plot_qq_dist(
                data=df["environmental_noise"].values,
                data_name=f"single_h2_environmental_noise_component_{i}",
                dist=scipy.stats.norm,
                dist_name=f"normal_mean_0_sd_{env_std}",
                loc=0,
                scale=env_std,
            )
            sum_data += df["environmental_noise"].values * const[i]
            sd_array[i] = env_std

        sd = np.dot(sd_array**2, const**2)
        sd = np.sqrt(sd)

        self._plot_qq_dist(
            data=sum_data,
            data_name="single_h2_environmental_noise_multivariate",
            dist=scipy.stats.norm,
            dist_name=f"normal_mean_0_sd_{sd}",
            loc=0,
            scale=sd,
        )


###############################################
# Infrastructure for running the tests and CLI
###############################################


@attr.s
class TestInstance:
    """
    A single test instance, that consists of the test class and the test method
    name.
    """

    test_class = attr.ib()
    method_name = attr.ib()

    def run(self, basedir):
        logging.info(f"Running {self}")
        output_dir = pathlib.Path(basedir) / self.test_class / self.method_name
        output_dir.mkdir(parents=True, exist_ok=True)

        instance = getattr(sys.modules[__name__], self.test_class)(output_dir)
        method = getattr(instance, self.method_name)
        method()


@attr.s
class TestSuite:
    """
    Class responsible for registering all known tests.
    """

    tests = attr.ib(init=False, factory=dict)
    classes = attr.ib(init=False, factory=set)

    def register(self, test_class, method_name):
        test_instance = TestInstance(test_class, method_name)
        if method_name in self.tests:
            raise ValueError(f"Test name {method_name} already used.")
        self.tests[method_name] = test_instance
        self.classes.add(test_class)

    def get_tests(self, names=None, test_class=None):
        if names is not None:
            tests = [self.tests[name] for name in names]
        elif test_class is not None:
            tests = [
                test for test in self.tests.values() if test.test_class == test_class
            ]
        else:
            tests = list(self.tests.values())
        return tests


@attr.s
class TestRunner:
    """
    Class responsible for running test instances.
    """

    def __run_sequential(self, tests, basedir, progress):
        for test in tests:
            test.run(basedir)
            progress.update()

    def __run_parallel(self, tests, basedir, num_threads, progress):
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_threads
        ) as executor:
            futures = [executor.submit(test.run, basedir) for test in tests]
            exception = None
            for future in concurrent.futures.as_completed(futures):
                exception = future.exception()
                if exception is not None:
                    # At least tell the user that we've had an exception.
                    # Other stuff will still keep going, though.
                    logging.error("EXCEPTION:%s", exception)
                    break
                progress.update()
            if exception is not None:
                # Try to clear out as much work as we can, but it'll still run a
                # lot of stuff before we finish
                for future in futures:
                    future.cancel()
                raise exception

    def run(self, tests, basedir, num_threads, show_progress):
        progress = tqdm.tqdm(total=len(tests), disable=not show_progress)
        logging.info(f"running {len(tests)} tests using {num_threads} processes")
        if num_threads <= 1:
            self.__run_sequential(tests, basedir, progress)
        else:
            self.__run_parallel(tests, basedir, num_threads, progress)
        progress.close()


def setup_logging(args):
    log_level = "INFO"
    if args.quiet:
        log_level = "WARN"
    if args.debug:
        log_level = "DEBUG"

    daiquiri.setup(level=log_level)
    msprime_logger = daiquiri.getLogger("msprime")
    msprime_logger.setLevel("WARN")
    mpl_logger = daiquiri.getLogger("matplotlib")
    mpl_logger.setLevel("WARN")


def run_tests(suite, args):
    setup_logging(args)
    runner = TestRunner()

    if len(args.tests) > 0:
        tests = suite.get_tests(names=args.tests)
    elif args.test_class is not None:
        tests = suite.get_tests(test_class=args.test_class)
    else:
        tests = suite.get_tests()

    runner.run(tests, args.output_dir, args.num_threads, not args.no_progress)


def make_suite():
    suite = TestSuite()

    for cls_name, cls in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(cls) and issubclass(cls, Test):
            test_class_instance = cls()
            for name, thing in inspect.getmembers(test_class_instance):
                if inspect.ismethod(thing):
                    if name.startswith("test_"):
                        suite.register(cls_name, name)
    return suite


def main():
    suite = make_suite()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-class",
        "-c",
        default=None,
        choices=sorted(suite.classes),
        help="Run all tests for specified test class",
    )
    parser.add_argument(
        "tests",
        nargs="*",
        help="Run specific tests. Use the --list option to see those available",
    )
    parser.add_argument(
        "--output-dir",
        "-d",
        type=str,
        default="tmp__NOBACKUP__",
        help="specify the base output directory",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--quiet", "-q", action="store_true", help="Do not write any output"
    )
    group.add_argument(
        "--debug", "-D", action="store_true", help="Write out debug output"
    )
    parser.add_argument(
        "--no-progress", "-n", action="store_true", help="Do not show progress bar"
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available checks and exit"
    )
    parser.add_argument(
        "--num-threads", "-t", type=int, default=1, help="Specify number of threads"
    )
    args = parser.parse_args()
    if args.list:
        print("All available tests")
        for test in suite.tests.values():
            print(test.test_class, test.method_name, sep="\t")
    else:
        run_tests(suite, args)


if __name__ == "__main__":
    main()
