import inspect
import itertools
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import tskit
import tstrait
from tqdm import tqdm


def sim_tree_seq():
    """
    The following tree sequence will be generated:
      10
    ┏━┻┓
    ┃  9
    ┃ ┏┻━┓
    ┃ ┃   8
    ┃ ┃  ┏┻━┓
    ┃ ┃  ┃  7
    ┃ ┃  ┃ ┏┻━┓
    ┃ ┃  ┃ ┃  6
    ┃ ┃  ┃ ┃ ┏┻━┓
    0 1  2 3 4  5

    Individual 0: Node 0 and 1
    Individual 1: Node 2 and 3
    Individual 3: Node 4 and 5

    Site 0 Ancestral State: "A"
        Causal Mutation: Node 8
        Reverse Mutation: Node 4

    Site 1 Ancestral State: "A"
        Causal Mutation: Node 5

    Site 0 Genotype:
        [A, A, T, T, A, T]
        Ancestral state: A
        Causal allele freq: 0.5
        Individual 0: 0 causal
        Individual 1: 2 causal
        Individual 2: 1 causal

    Site 1 Genotype:
        [A, A, A, A, A, T]
        Ancestral state: T
        Causal allele freq: 1/6
        Individual 0: 0 causal
        Individual 1: 0 causal
        Individual 2: 1 causal

    Effect size distribution:

    SD Formula:
        trait_sd / sqrt(2) * [sqrt(2 * freq * (1 - freq))] ^ alpha
        sqrt(2) from 2 causal sites
    """
    ts = tskit.Tree.generate_comb(6, span=2).tree_sequence

    tables = ts.dump_tables()
    for j in range(2):
        tables.sites.add_row(j, "A")

    tables.individuals.add_row()
    tables.individuals.add_row()
    tables.individuals.add_row()
    individuals = tables.nodes.individual
    individuals[0] = 0
    individuals[1] = 0
    individuals[2] = 1
    individuals[3] = 1
    individuals[4] = 2
    individuals[5] = 2
    tables.nodes.individual = individuals
    tables.mutations.add_row(site=0, node=8, derived_state="T")
    tables.mutations.add_row(site=0, node=4, derived_state="A", parent=0)
    tables.mutations.add_row(site=1, node=5, derived_state="T")
    ts = tables.tree_sequence()
    return ts


def sim_tree_seq_freq(alpha):
    const1 = np.sqrt(pow(2 * 0.5 * (1 - 0.5), alpha))
    const2 = np.sqrt(pow(2 * 1 / 6 * (1 - 1 / 6), alpha))

    return const1, const2


class Test:
    def __init__(self, basedir, cl_name):
        self.basedir = basedir
        self.name = cl_name
        self.set_output_dir(cl_name)

    def set_output_dir(self, cl_name):
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

    def plot_qq_compare(self, v1, v2, x_label, y_label, filename, title=""):
        sm.qqplot_2samples(v1, v2, x_label, y_label, line="45")
        plt.title(title)
        f = self._build_filename(filename)
        plt.savefig(f, dpi=72)
        plt.close("all")

    def plot_qq_normal(self, data, loc, scale, filename, title=""):
        sm.qqplot(data, stats.norm, loc=loc, scale=scale, line="45")
        plt.title(title)
        f = self._build_filename(filename)
        plt.savefig(f, dpi=72)
        plt.close("all")

    def plot_qq_exponential(self, data, scale, filename, title=""):
        sm.qqplot(data, stats.expon, scale=scale, line="45")
        plt.title(title)
        f = self._build_filename(filename)
        plt.savefig(f, dpi=72)
        plt.close("all")


class TestNormal(Test):
    def test_normal(self):

        alpha_array = np.array([0, -0.3, 1])
        mean_array = np.array([0, 1])
        var_array = np.array([1, 2])

        ts = sim_tree_seq()

        count = 0

        prod = itertools.product(alpha_array, mean_array, var_array)
        for element in tqdm(prod, total=12):
            alpha = element[0]
            mean = element[1]
            var = element[2]
            sd = np.sqrt(var)
            const0, const1 = sim_tree_seq_freq(alpha)
            model = tstrait.trait_model(distribution="normal", mean=mean, var=var)
            effect_size0 = np.zeros(1000)
            effect_size1 = np.zeros(1000)
            for i in range(1000):
                sim_result = tstrait.sim_trait(
                    ts=ts,
                    num_causal=2,
                    model=model,
                    alpha=alpha,
                    random_seed=i + 1000 * count,
                )
                effect_size0[i] = sim_result["effect_size"][0]
                effect_size1[i] = sim_result["effect_size"][1]
            self.plot_qq_normal(
                data=effect_size0,
                loc=mean * const0 / 2,
                scale=sd * const0 / 2,
                filename=f"effect_size_0_{count}",
                title=f"EffectSize0, Normal, alpha = {alpha}, "
                f"mean = {mean}, var = {var}",
            )
            self.plot_qq_normal(
                data=effect_size1,
                loc=mean * const1 / 2,
                scale=sd * const1 / 2,
                filename=f"effect_size_1_{count}",
                title=f"EffectSize1, Normal, alpha = {alpha}, "
                f"mean = {mean}, var = {var}",
            )
            count += 1


class TestExponential(Test):
    def test_exponential(self):

        alpha_array = np.array([0, -0.3, 1])
        scale_array = np.array([1, 2.5])

        ts = sim_tree_seq()

        count = 0

        prod = itertools.product(alpha_array, scale_array)
        for element in tqdm(prod, total=6):
            alpha = element[0]
            scale = element[1]
            const0, const1 = sim_tree_seq_freq(alpha)
            model = tstrait.trait_model(
                distribution="exponential", scale=scale, negative=False
            )
            effect_size0 = np.zeros(1000)
            effect_size1 = np.zeros(1000)
            for i in range(1000):
                sim_result = tstrait.sim_trait(
                    ts=ts,
                    num_causal=2,
                    model=model,
                    alpha=alpha,
                    random_seed=i + 1000 * count,
                )
                effect_size0[i] = sim_result["effect_size"][0]
                effect_size1[i] = sim_result["effect_size"][1]
            self.plot_qq_exponential(
                data=effect_size0,
                scale=scale / 2 * const0,
                filename=f"effect_size_0_{count}",
                title=f"EffectSize0, Exponential, alpha = {alpha}, "
                f"scale = {scale}, negative = False",
            )
            self.plot_qq_exponential(
                data=effect_size1,
                scale=scale / 2 * const1,
                filename=f"effect_size_1_{count}",
                title=f"EffectSize1, Expnential, alpha = {alpha}, "
                f"scale = {scale}, negative = False",
            )
            count += 1

    def test_exponential_negative(self):

        alpha_array = np.array([0, -0.3, 1])
        scale_array = np.array([1, 2.5])

        ts = sim_tree_seq()

        count = 0

        prod = itertools.product(alpha_array, scale_array)
        for element in tqdm(prod, total=6):
            alpha = element[0]
            scale = element[1]
            const0, const1 = sim_tree_seq_freq(alpha)
            model = tstrait.trait_model(
                distribution="exponential", scale=scale, negative=True
            )
            effect_size0 = np.zeros(1000)
            effect_size1 = np.zeros(1000)
            for i in range(1000):
                sim_result = tstrait.sim_trait(
                    ts=ts,
                    num_causal=2,
                    model=model,
                    alpha=alpha,
                    random_seed=i + 1000 * count,
                )
                effect_size0[i] = sim_result["effect_size"][0]
                effect_size1[i] = sim_result["effect_size"][1]
            rng = np.random.default_rng(count + 1)
            exponential = rng.exponential(scale=scale / 2, size=1000)
            exponential *= rng.choice([-1, 1], size=1000)

            self.plot_qq_compare(
                v1=effect_size0,
                v2=exponential * const0,
                x_label="Simulated Effect Size",
                y_label="Exponential Distribution",
                filename=f"effect_size_0_negative_{count}",
                title=f"EffectSize0, Exponential, alpha = {alpha}, "
                f"scale = {scale}, negative = True",
            )
            self.plot_qq_compare(
                v1=effect_size1,
                v2=exponential * const1,
                x_label="Simulated Effect Size",
                y_label="Exponential Distribution",
                filename=f"effect_size_1_negative_{count}",
                title=f"EffectSize1, Exponential, alpha = {alpha}, "
                f"scale = {scale}, negative = True",
            )

            count += 1


class TestT(Test):
    def test_t(self):

        alpha_array = np.array([0, -0.3, 1])
        mean_array = np.array([0, 1])
        var_array = np.array([1, 2])
        df_array = np.array([10, 13.5])

        ts = sim_tree_seq()

        count = 0

        prod = itertools.product(alpha_array, mean_array, var_array, df_array)
        for element in tqdm(prod, total=24):
            alpha = element[0]
            mean = element[1]
            var = element[2]
            df = element[3]
            sd = np.sqrt(var)
            const0, const1 = sim_tree_seq_freq(alpha)
            model = tstrait.trait_model(distribution="t", mean=mean, var=var, df=df)
            effect_size0 = np.zeros(1000)
            effect_size1 = np.zeros(1000)
            for i in range(1000):
                sim_result = tstrait.sim_trait(
                    ts=ts,
                    num_causal=2,
                    model=model,
                    alpha=alpha,
                    random_seed=i + 1000 * count,
                )
                effect_size0[i] = sim_result["effect_size"][0]
                effect_size1[i] = sim_result["effect_size"][1]
            rng = np.random.default_rng(count + 1)
            simulated_t = rng.standard_t(df, size=1000)
            simulated_t = simulated_t * sd + mean
            simulated_t /= 2

            self.plot_qq_compare(
                v1=effect_size0,
                v2=simulated_t * const0,
                x_label="Simulated Effect Size",
                y_label="T Distribution",
                filename=f"effect_size_0_{count}",
                title=f"EffectSize0, T, alpha = {alpha}, mean = {mean}, "
                f"var = {var}, df = {df}",
            )
            self.plot_qq_compare(
                v1=effect_size1,
                v2=simulated_t * const1,
                x_label="Simulated Effect Size",
                y_label="T Distribution",
                filename=f"effect_size_1_{count}",
                title=f"EffectSize1, T, alpha = {alpha}, mean = {mean}, "
                f"var = {var}, df = {df}",
            )
            count += 1


class TestGamma(Test):
    def test_gamma(self):

        alpha_array = np.array([0, -0.3, 1])
        shape_array = np.array([1, 3.5])
        scale_array = np.array([1, 2.5])

        ts = sim_tree_seq()

        count = 0

        prod = itertools.product(alpha_array, shape_array, scale_array)
        for element in tqdm(prod, total=12):
            alpha = element[0]
            shape = element[1]
            scale = element[2]
            const0, const1 = sim_tree_seq_freq(alpha)
            model = tstrait.trait_model(
                distribution="gamma", shape=shape, scale=scale, negative=False
            )
            effect_size0 = np.zeros(1000)
            effect_size1 = np.zeros(1000)
            for i in range(1000):
                sim_result = tstrait.sim_trait(
                    ts=ts,
                    num_causal=2,
                    model=model,
                    alpha=alpha,
                    random_seed=i + 1000 * count,
                )
                effect_size0[i] = sim_result["effect_size"][0]
                effect_size1[i] = sim_result["effect_size"][1]

            rng = np.random.default_rng(count + 1)
            gamma = rng.gamma(shape=shape, scale=scale, size=1000)
            gamma /= 2

            self.plot_qq_compare(
                v1=effect_size0,
                v2=gamma * const0,
                x_label="Simulated Effect Size",
                y_label="Gamma Distribution",
                filename=f"effect_size_0_{count}",
                title=f"EffectSize0, Gamma, alpha = {alpha}, "
                f"shape = {shape}, scale = {scale}, negative = False",
            )
            self.plot_qq_compare(
                v1=effect_size1,
                v2=gamma * const1,
                x_label="Simulated Effect Size",
                y_label="Gamma Distribution",
                filename=f"effect_size_1_{count}",
                title=f"EffectSize1, Gamma, alpha = {alpha}, "
                f"shape = {shape}, scale = {scale}, negative = False",
            )
            count += 1

    def test_gamma_negative(self):

        alpha_array = np.array([0, -0.3, 1])
        shape_array = np.array([1, 3.5])
        scale_array = np.array([1, 2.5])

        ts = sim_tree_seq()

        count = 0

        prod = itertools.product(alpha_array, shape_array, scale_array)
        for element in tqdm(prod, total=12):
            alpha = element[0]
            shape = element[1]
            scale = element[2]
            const0, const1 = sim_tree_seq_freq(alpha)
            model = tstrait.trait_model(
                distribution="gamma", shape=shape, scale=scale, negative=True
            )
            effect_size0 = np.zeros(1000)
            effect_size1 = np.zeros(1000)
            for i in range(1000):
                sim_result = tstrait.sim_trait(
                    ts=ts,
                    num_causal=2,
                    model=model,
                    alpha=alpha,
                    random_seed=i + 1000 * count,
                )
                effect_size0[i] = sim_result["effect_size"][0]
                effect_size1[i] = sim_result["effect_size"][1]

            rng = np.random.default_rng(count + 1)
            gamma = rng.gamma(shape=shape, scale=scale, size=1000)
            gamma /= 2
            gamma *= rng.choice([-1, 1], size=1000)

            self.plot_qq_compare(
                v1=effect_size0,
                v2=gamma * const0,
                x_label="Simulated Effect Size",
                y_label="Gamma Distribution",
                filename=f"effect_size_0_negative_{count}",
                title=f"EffectSize0, Gamma, alpha = {alpha}, "
                f"shape = {shape}, scale = {scale}, negative = True",
            )
            self.plot_qq_compare(
                v1=effect_size1,
                v2=gamma * const1,
                x_label="Simulated Effect Size",
                y_label="Gamma Distribution",
                filename=f"effect_size_1_negative_{count}",
                title=f"EffectSize1, Gamma, alpha = {alpha}, "
                f"shape = {shape}, scale = {scale}, negative = True",
            )
            count += 1


def run_tests(suite, output_dir):
    print(f"[+] Test suite contains {len(suite)} tests.")
    for cl_name in suite:
        instance = getattr(sys.modules[__name__], cl_name)(output_dir, cl_name)
        instance._run_tests()


if __name__ == "__main__":
    run_tests(
        ["TestNormal", "TestExponential", "TestT", "TestGamma"],
        "_output/stats_tests_output",
    )
