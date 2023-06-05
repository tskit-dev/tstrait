import tskit
import tstrait.simulate_phenotype as simulate_phenotype
import tstrait.trait_model as trait_model
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats

import inspect
import pathlib
import sys

ts = tskit.Tree.generate_comb(6, span=10).tree_sequence

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


class TestGenetic(Test):
    def test_normal(self):
        n = 1000
        rng = np.random.default_rng(1)
        alpha_array = np.array([0, -0.3])
        trait_mean_array = np.array([0, 1])
        trait_sd_array = np.array([1, 2])
        h2_array = np.array([0.3, 0.8])

        count = 0

        for alpha in alpha_array:
            for trait_mean in trait_mean_array:
                for trait_sd in trait_sd_array:
                    for h2 in h2_array:
                        model = trait_model.TraitModelAllele(
                            trait_mean, trait_sd, alpha
                        )
                        genetic0 = np.zeros(1000)
                        genetic1 = np.zeros(1000)
                        genetic2 = np.zeros(1000)

                        environment0 = np.zeros(1000)
                        environment1 = np.zeros(1000)
                        environment2 = np.zeros(1000)

                        for i in range(1000):
                            (
                                phenotype_result,
                                genetic_result,
                            ) = simulate_phenotype.sim_phenotype(
                                ts, num_causal=2, model=model, h2=h2, random_seed=i
                            )
                            genetic0[i] = phenotype_result.genetic_value[0]
                            genetic1[i] = phenotype_result.genetic_value[1]
                            genetic2[i] = phenotype_result.genetic_value[2]

                            environment0[i] = phenotype_result.environment_noise[0]
                            environment1[i] = phenotype_result.environment_noise[1]
                            environment2[i] = phenotype_result.environment_noise[2]

                        assert np.array_equal(genetic0, np.zeros(1000))

                        effect_size_sd0 = (
                            trait_sd
                            / np.sqrt(2)
                            * np.sqrt(pow(2 * 0.5 * (1 - 0.5), alpha))
                        )
                        effect_size_sd1 = (
                            trait_sd
                            / np.sqrt(2)
                            * np.sqrt(pow(2 * 1 / 6 * (1 - 1 / 6), alpha))
                        )

                        ind_sd1 = effect_size_sd0 * 2
                        ind_sd2 = np.sqrt(
                            (effect_size_sd0**2) + (effect_size_sd1**2)
                        )

                        self.plot_qq_normal(
                            data=genetic1,
                            loc=2 * trait_mean,
                            scale=ind_sd1,
                            filename=f"ind_1_genetic_{count}",
                            title=f"Individual 1 Genetic, alpha = {alpha}, trait_mean = {trait_mean}, trait_sd = {trait_sd}, h2 = {h2}",
                        )
                        self.plot_qq_normal(
                            data=genetic2,
                            loc=2 * trait_mean,
                            scale=ind_sd2,
                            filename=f"ind_2_genetic_{count}",
                            title=f"Individual 2 Genetic, alpha = {alpha}, trait_mean = {trait_mean}, trait_sd = {trait_sd}, h2 = {h2}",
                        )

                        genetic_sim = np.zeros(3)
                        env_sim = np.zeros(1000)

                        for i in range(1000):
                            x1 = rng.normal(loc=trait_mean, scale=effect_size_sd0)
                            x2 = rng.normal(loc=trait_mean, scale=effect_size_sd1)
                            genetic_sim[0] = 0
                            genetic_sim[1] = 2 * x1
                            genetic_sim[2] = x1 + x2
                            env_std = np.sqrt((1 - h2) / h2 * np.var(genetic_sim))
                            env_sim[i] = rng.normal(loc=0, scale=env_std)

                        self.plot_qq_compare(
                            v1=environment0,
                            v2=env_sim,
                            x_label="Individual 0 Environment",
                            y_label="Simulated values",
                            filename=f"ind_0_env_{count}",
                            title=f"Individual 0 Env, alpha = {alpha}, trait_mean = {trait_mean}, trait_sd = {trait_sd}, h2 = {h2}",
                        )
                        self.plot_qq_compare(
                            v1=environment1,
                            v2=env_sim,
                            x_label="Individual 1 Environment",
                            y_label="Simulated values",
                            filename=f"ind_1_env_{count}",
                            title=f"Individual 1 Env, alpha = {alpha}, trait_mean = {trait_mean}, trait_sd = {trait_sd}, h2 = {h2}",
                        )
                        self.plot_qq_compare(
                            v1=environment2,
                            v2=env_sim,
                            x_label="Individual 2 Environment",
                            y_label="Simulated values",
                            filename=f"ind_2_env_{count}",
                            title=f"Individual 2 Env, alpha = {alpha}, trait_mean = {trait_mean}, trait_sd = {trait_sd}, h2 = {h2}",
                        )

                        count += 1


def run_tests(suite, output_dir):
    print(f"[+] Test suite contains {len(suite)} tests.")
    for cl_name in suite:
        instance = getattr(sys.modules[__name__], cl_name)(output_dir, cl_name)
        instance._run_tests()


if __name__ == "__main__":
    run_tests(["TestGenetic"], "_output/stats_tests_output")
