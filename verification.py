"""
Script to automate verification of tstrait against known statistical
results and benchmark programs such as AlphaSimR and simplePHENOTYPES.

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


class ExactTest(Test):
    """
    Compare tstrait against simplePHENOTYPES, AlphaSimR and the simulation framework
    in ARG-Needle paper. For all benchmarking, we simulate a genetic data of 100
    individuals with 100 kb sequence length and mutations are simulated from an
    infinite-sites model.

    We assume that the following in all quantitative trait simulation:
    - Narrow-sense heritability is 1 to compare the simulated genetic values.
    - Genetic effect sizes are simulated from an external simulator and they are used to
    simulate quantitative traits in tstrait.
    """

    def _simulate_simplePHENOTYPE(
        self, ts, num_causal, add_effect, random_seed, num_trait=1, add_effect_2=1
    ):
        """
        The function to simulate quantitative traits by using simplePHENOTYPES.
        We will specify the number of causal sites and the parameter for the
        geometric series where the effect sizes are determined.
        """

        directory = tempfile.TemporaryDirectory()

        vcf_filename = "vcf_comparison_simplePHENOTYPES"
        with open(f"{directory.name}/{vcf_filename}.vcf", "w") as vcf_file:
            ts.write_vcf(vcf_file)
        cmd = ["Rscript", "data/simulate_simplePHENOTYPES.R"]
        args = [
            str(num_causal),
            str(num_trait),
            str(add_effect),
            str(add_effect_2),
            directory.name,
            vcf_filename,
            str(random_seed),
        ]
        input_cmd = cmd + args
        subprocess.check_output(input_cmd)

        if num_trait == 1:
            phenotype_df = pd.read_csv(
                f"{directory.name}/Simulated_Data_1_Reps_Herit_1.txt", sep="\t"
            )
            del phenotype_df["reps"]
            phenotype_df = phenotype_df.rename(
                columns={"<Trait>": "individual_id", "Pheno": "phenotype"}
            )
        else:
            phenotype_df = pd.read_csv(
                f"{directory.name}/Simulated_Data_1_Reps_Herit_1_1.txt", sep="\t"
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

        num_ind = ts.num_individuals
        # Change the individual ID in simplePHENOTYPES output to be consistent with the
        # tstrait output
        for i in range(num_ind):
            phenotype_df = phenotype_df.replace(f"tsk_{i}", i)

        qtn_df = pd.read_csv(f"{directory.name}/Additive_Selected_QTNs.txt", sep="\t")

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

        directory.cleanup()

        return simulation_result

    def _simulate_AlphaSimR(self, ts, num_causal, random_seed, corA=1, num_trait=1):
        """
        The function to simulate quantitative traits by using AlphaSimR. We will
        specify the number of causal sites, such that the AlphaSimR simulation
        will be conducted randomly. corA is used to specify the correlation
        coefficient of pleiotropic traits.
        """

        directory = tempfile.TemporaryDirectory()
        tree_filename = "tree_comparison_AlphaSimR"
        ts.dump(f"{directory.name}/{tree_filename}.tree")
        phenotype_filename = "phenotype_comparison_AlphaSimR"
        trait_filename = "trait_comparison_AlphaSimR"
        cmd = ["Rscript", "data/simulate_AlphaSimR.R"]
        args = [
            str(num_causal),
            directory.name,
            tree_filename,
            phenotype_filename,
            trait_filename,
            str(corA),
            str(num_trait),
            str(random_seed),
        ]
        input_cmd = cmd + args
        subprocess.check_output(input_cmd)

        phenotype_df = pd.read_csv(f"{directory.name}/{phenotype_filename}.csv")
        trait_df = pd.read_csv(f"{directory.name}/{trait_filename}.csv")

        # Obtain the list of causal allele
        causal_allele = []
        for site_id in trait_df["site_id"]:
            allele = ts.mutation(site_id).derived_state
            causal_allele.append(allele)

        trait_df["causal_allele"] = causal_allele

        simulation_result = SimulationResult(phenotype=phenotype_df, trait=trait_df)

        directory.cleanup()

        return simulation_result

    def _simulate_arg_needle(self, ts, alpha, random_seed):
        """
        Simulate phenotypes by using the quantitative trait simulation framework that is
        adapted from the ARG-Needle paper.
        https://zenodo.org/records/7745746
        """

        rng = np.random.default_rng(random_seed)
        num_ind = ts.num_individuals

        phenotypes = np.zeros(num_ind)
        beta_list = []
        causal_allele = []

        for variant in ts.variants():
            row = variant.genotypes.astype("float64")
            row = row.reshape((num_ind, 2)).sum(axis=-1)
            std = np.std(row, ddof=1)
            beta = rng.normal()
            beta_list.append(beta)
            causal_allele.append(variant.alleles[1])
            phenotypes += row * (beta * std**alpha)

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

        simulation_result = SimulationResult(phenotype=phenotype_df, trait=trait_df)

        return simulation_result

    def _run(self, model, random_seed, **kwargs):
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

        if model == "simplePHENOTYPES":
            simulation_output = self._simulate_simplePHENOTYPE(
                ts=ts,
                num_causal=kwargs["num_causal"],
                add_effect=kwargs["add_effect"],
                random_seed=random_seed,
                num_trait=kwargs["num_trait"],
                add_effect_2=kwargs["add_effect_2"],
            )

        elif model == "AlphaSimR":
            simulation_output = self._simulate_AlphaSimR(
                ts=ts,
                num_causal=kwargs["num_causal"],
                random_seed=random_seed,
                corA=kwargs["corA"],
                num_trait=kwargs["num_trait"],
            )

        elif model == "ARG-Needle":
            simulation_output = self._simulate_arg_needle(
                ts=ts, alpha=kwargs["alpha"], random_seed=random_seed
            )

        trait_df = simulation_output.trait.sort_values(by=["site_id"])
        genetic_df = tstrait.genetic_value(ts, trait_df)
        phenotype_df = tstrait.sim_env(genetic_df, h2=1, random_seed=1)

        if model == "simplePHENOTYPES":
            grouped = phenotype_df.groupby("trait_id")[["phenotype"]]
            phenotype_df = grouped.transform(lambda x: (x - x.mean()))
        elif model == "AlphaSimR":
            phenotype_df = tstrait.normalise_phenotypes(
                phenotype_df, mean=0, var=1, ddof=0
            )
        elif model == "ARG-Needle":
            phenotype_df = tstrait.normalise_phenotypes(phenotype_df, mean=0, var=1)

        tstrait_phenotype = phenotype_df["phenotype"]
        simulated_phenotype = simulation_output.phenotype["phenotype"].values

        np.testing.assert_array_almost_equal(tstrait_phenotype, simulated_phenotype)

    def test_exact_simplePHENOTYPES_single(self):
        self._run(
            model="simplePHENOTYPES",
            num_causal=30,
            add_effect=0.9,
            random_seed=100,
            num_trait=1,
            add_effect_2=1,
        )

    def test_exact_simplePHENOTYPES_pleiotropic(self):
        self._run(
            model="simplePHENOTYPES",
            num_causal=30,
            add_effect=0.9,
            random_seed=101,
            num_trait=2,
            add_effect_2=0.8,
        )

    def test_exact_AlphaSimR_single(self):
        self._run(
            model="AlphaSimR", num_causal=100, random_seed=200, corA=1, num_trait=1
        )

    def test_exact_AlphaSimR_pleiotropic(self):
        self._run(
            model="AlphaSimR", num_causal=100, random_seed=201, corA=0.8, num_trait=2
        )

    def test_exact_ARG_Needle(self):
        self._run(model="ARG-Needle", alpha=0, random_seed=300)


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
