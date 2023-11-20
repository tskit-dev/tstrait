import msprime
import numpy as np
import pandas as pd
import pytest
import tstrait
from scipy import stats
from tstrait.base import _check_numeric_array
from tstrait.simulate_effect_size import _TraitSimulator

from .data import (
    allele_freq_one,
    binary_tree,
    non_binary_tree,
    simple_tree_seq,
)  # noreorder


@pytest.fixture(scope="class")
def sample_ts():
    ts = msprime.sim_ancestry(10, sequence_length=100_000, random_seed=1)
    ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
    return ts


@pytest.fixture(scope="class")
def sample_trait_model():
    return tstrait.trait_model(distribution="normal", mean=0, var=1)


def freqdep(alpha, freq):
    return np.sqrt(pow(2 * freq * (1 - freq), alpha))


class TestInput:
    """This test will check that an informative error is raised when the input parameter
    does not have an appropriate type or value.
    """

    def test_input_type(self, sample_ts, sample_trait_model):
        with pytest.raises(
            TypeError, match="ts must be a <class 'tskit.trees.TreeSequence'> instance"
        ):
            tstrait.sim_trait(ts=1, num_causal=1, model=sample_trait_model)
        with pytest.raises(
            TypeError,
            match="model must be a <class 'tstrait.trait_model.TraitModel'> instance",
        ):
            tstrait.sim_trait(ts=sample_ts, num_causal=1, model="model")

    def test_bad_input(self, sample_ts, sample_trait_model):
        ts = msprime.sim_ancestry(2, sequence_length=100_000, random_seed=1)
        with pytest.raises(ValueError, match="No mutation in the tree sequence input"):
            tstrait.sim_trait(ts=ts, num_causal=1, model=sample_trait_model)
        num_causal = sample_ts.num_sites + 1
        with pytest.raises(
            ValueError,
            match="num_causal must be an integer not greater than the number of sites "
            "in ts",
        ):
            tstrait.sim_trait(
                ts=sample_ts, num_causal=num_causal, model=sample_trait_model
            )

    @pytest.mark.parametrize("num_causal", [0, -1])
    def test_num_causal_input(self, sample_ts, sample_trait_model, num_causal):
        with pytest.raises(TypeError, match="num_causal must be an integer"):
            tstrait.sim_trait(ts=sample_ts, num_causal=1.0, model=sample_trait_model)
        with pytest.raises(
            ValueError, match="num_causal must be an integer not less than 1"
        ):
            tstrait.sim_trait(
                ts=sample_ts, num_causal=num_causal, model=sample_trait_model
            )

    def test_alpha_input(self, sample_ts, sample_trait_model):
        with pytest.raises(TypeError, match="alpha must be numeric"):
            tstrait.sim_trait(
                ts=sample_ts, num_causal=1, model=sample_trait_model, alpha="1"
            )


class TestOutputDim:
    """Check that the sim_trait function gives the correct output regardless of the
    tree sequence mutation type or the trait model
    """

    def check_dimensions(self, df, num_causal):
        assert len(df) == num_causal
        assert df.shape[1] == 6
        assert list(df.columns) == [
            "position",
            "site_id",
            "effect_size",
            "causal_allele",
            "allele_freq",
            "trait_id",
        ]
        _check_numeric_array(df["site_id"], "site_id")
        _check_numeric_array(df["effect_size"], "effect_size")
        _check_numeric_array(df["trait_id"], "trait_id")
        _check_numeric_array(df["position"], "position")
        _check_numeric_array(df["allele_freq"], "allele_freq")

    @pytest.mark.parametrize(
        "trait_model",
        [
            tstrait.trait_model(distribution="normal", mean=0, var=1),
            tstrait.trait_model(distribution="exponential", scale=1),
            tstrait.trait_model(distribution="fixed", value=1),
            tstrait.trait_model(distribution="t", mean=0, var=1, df=1),
            tstrait.trait_model(distribution="gamma", shape=1, scale=1),
        ],
    )
    def test_output(self, sample_ts, trait_model):
        num_causal = 10
        sim_result = tstrait.sim_trait(
            ts=sample_ts, num_causal=num_causal, model=trait_model
        )
        self.check_dimensions(sim_result, num_causal)
        np.testing.assert_equal(sim_result["trait_id"], np.zeros(num_causal))
        assert (
            0 <= np.min(sim_result["site_id"])
            and np.max(sim_result["site_id"]) < sample_ts.num_sites
        )

    def test_output_default_num_causal(self, sample_ts):
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        sim_result = tstrait.sim_trait(ts=sample_ts, model=model)
        self.check_dimensions(sim_result, 1)

        sim_result = tstrait.sim_trait(ts=sample_ts, model=model, alpha=-1)
        self.check_dimensions(sim_result, 1)

    @pytest.mark.parametrize("alpha", [1, -1])
    def test_output_alpha(self, alpha, sample_ts, sample_trait_model):
        num_causal = 10
        sim_result = tstrait.sim_trait(
            ts=sample_ts, num_causal=num_causal, model=sample_trait_model, alpha=alpha
        )
        self.check_dimensions(sim_result, num_causal)

    def test_output_binary(self, sample_trait_model):
        num_causal = 5
        ts = msprime.sim_ancestry(10, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1, model="binary")
        sim_result = tstrait.sim_trait(
            ts=ts, num_causal=num_causal, model=sample_trait_model
        )
        self.check_dimensions(sim_result, num_causal)

    def test_output_multivariate(self, sample_ts):
        mean = [0, 1]
        cov = [[4, 1], [1, 4]]
        num_causal = 5
        model = tstrait.trait_model(distribution="multi_normal", mean=mean, cov=cov)
        df = tstrait.sim_trait(
            ts=sample_ts,
            num_causal=num_causal,
            model=model,
            random_seed=1,
        )
        trait_ids = df.trait_id.unique()
        np.testing.assert_equal(trait_ids, np.array([0, 1]))
        assert len(df) == num_causal * 2
        df0 = df[df.trait_id == 0]
        df1 = df[df.trait_id == 1]
        self.check_dimensions(df0, num_causal)
        self.check_dimensions(df1, num_causal)
        assert (
            0 <= np.min(df["site_id"]) and np.max(df["site_id"]) < sample_ts.num_sites
        )
        assert 0 <= np.min(df["trait_id"]) and np.max(df["trait_id"]) <= 1

    def test_output_multivariate_alpha(self, sample_ts):
        mean = [0, 1]
        cov = [[4, 1], [1, 4]]
        num_causal = 5
        model = tstrait.trait_model(distribution="multi_normal", mean=mean, cov=cov)
        df = tstrait.sim_trait(
            ts=sample_ts,
            num_causal=num_causal,
            model=model,
            alpha=-1,
            random_seed=1,
        )
        trait_ids = df.trait_id.unique()
        np.testing.assert_equal(trait_ids, np.array([0, 1]))
        assert len(df) == num_causal * 2
        df0 = df[df.trait_id == 0]
        df1 = df[df.trait_id == 1]
        self.check_dimensions(df0, num_causal)
        self.check_dimensions(df1, num_causal)
        assert (
            0 <= np.min(df["site_id"]) and np.max(df["site_id"]) < sample_ts.num_sites
        )
        assert 0 <= np.min(df["trait_id"]) and np.max(df["trait_id"]) <= 1


def model_list():
    loc = 2
    scale = 5
    df = 10
    shape = 5
    distr = [
        (
            tstrait.trait_model(distribution="normal", mean=loc, var=scale**2),
            "norm",
            (loc, scale),
        ),
        (
            tstrait.trait_model(distribution="exponential", scale=scale),
            "expon",
            (0, scale),
        ),
        (
            tstrait.trait_model(distribution="t", mean=loc, var=scale**2, df=df),
            "t",
            (df, loc, scale),
        ),
        (
            tstrait.trait_model(distribution="gamma", shape=shape, scale=scale),
            "gamma",
            (shape, 0, scale),
        ),
    ]

    return distr


class Test_KSTest:
    """
    Test the distribution of effect sizes by using a Kolmogorov-Smirnov test.
    rvs is array-like object of random variables, df is the output dataframe from
    `tstrait.sim_trait`, dist is a name of a distribution in scipy.stats, and args
    are the args for scipy.stats.dist(*args)
    """

    def check_distribution(self, rvs, dist, args=()):
        D, pval = stats.kstest(rvs, dist, args=args, N=1000)
        if pval < 0.05:
            raise ValueError(f"KS test failed for distribution {dist}")

    @pytest.mark.parametrize("model, distr, args", model_list())
    def test_KStest(self, model, distr, args, sample_ts):
        result = np.array([])
        for i in range(2):
            sim_result = tstrait.sim_trait(
                ts=sample_ts, num_causal=1000, model=model, random_seed=i
            )
            result = np.concatenate((result, sim_result["effect_size"]))
        self.check_distribution(result, distr, args)

    def test_fixed(self, sample_ts):
        """
        Some effect sizes are 0, as there are sites with only ancestral states
        """
        value = 4
        model = tstrait.trait_model(distribution="fixed", value=value)
        sim_result = tstrait.sim_trait(ts=sample_ts, num_causal=1000, model=model)
        data = sim_result.loc[sim_result.effect_size != 0]
        np.testing.assert_allclose(data["effect_size"], np.ones(len(data)) * value)

    def test_multivariate(self, sample_ts):
        """
        Conduct a KS test for individual effect sizes and arbitrary summation of
        effect sizes.
        """
        np.random.seed(20)
        n = 3
        mean = np.random.randn(n)
        M = np.random.randn(n, n)
        cov = np.dot(M, M.T)
        num_causal = 2000
        model = tstrait.trait_model(distribution="multi_normal", mean=mean, cov=cov)
        sim_result = tstrait.sim_trait(
            ts=sample_ts, num_causal=num_causal, model=model, random_seed=100
        )
        sim_result1 = tstrait.sim_trait(
            ts=sample_ts, num_causal=num_causal, model=model, random_seed=201
        )
        sim_result = pd.concat([sim_result, sim_result1])

        sim_result = sim_result.reset_index()
        del sim_result["index"]

        const = np.random.randn(n)
        data_val = np.matmul(const, cov)
        data_sd = np.sqrt(np.matmul(data_val, const))
        sum_data = np.zeros(2 * num_causal)

        for i in range(n):
            df = sim_result.loc[sim_result.trait_id == i]
            self.check_distribution(
                df["effect_size"],
                "norm",
                (mean[i], np.sqrt(cov[i, i])),
            )
            sum_data += df["effect_size"] * const[i]

        self.check_distribution(
            sum_data,
            "norm",
            (np.matmul(const, mean), data_sd),
        )


class TestGenotype:
    """
    Test the `_obtain_allele_count` method and check if can return the correct
    results. Afterwards, we will check the `_sim_beta` method.
    """

    def test_binary_tree(self):
        """
        We will be testing the _TraitSimulator class against the
        binary tree model.
        """
        ts = binary_tree()
        alpha = -1
        value = 2
        fixed_model = tstrait.trait_model(distribution="fixed", value=value)
        tree = ts.first()
        simulator = _TraitSimulator(
            ts, num_causal=ts.num_sites, model=fixed_model, alpha=alpha, random_seed=1
        )

        # We will examine `_obtain_allele_count` method against all possible states
        # at a site to make sure that the code is running correctly
        for state, num in zip(["A", "T", 1], [2, 2, 0]):
            assert simulator._obtain_allele_count(tree, ts.site(0), state) == num
        for state, num in zip(["A", "T", "C"], [2, 1, 1]):
            assert simulator._obtain_allele_count(tree, ts.site(1), state) == num
        for state, num in zip(["T", "G", "C", "A"], [0, 0, 1, 3]):
            assert simulator._obtain_allele_count(tree, ts.site(2), state) == num
        for state, num in zip(["T", "C", "A"], [2, 2, 0]):
            assert simulator._obtain_allele_count(tree, ts.site(3), state) == num

        result = simulator._sim_beta(np.arange(4), ["A", "A", "G", "C"])
        np.testing.assert_array_almost_equal(
            result.beta_array,
            [
                value * freqdep(alpha, 1 / 2),
                value * freqdep(alpha, 1 / 2),
                0,
                value * freqdep(alpha, 1 / 2),
            ],
        )
        np.testing.assert_array_almost_equal(
            result.allele_freq, [1 / 2, 1 / 2, 0, 1 / 2]
        )

    def test_non_binary_tree(self):
        """
        Repeat the above procedure for non-binary tree.
        """
        ts = non_binary_tree()
        value = 2
        alpha = -1
        fixed_model = tstrait.trait_model(distribution="fixed", value=value)
        tree = ts.first()
        simulator = _TraitSimulator(
            ts, num_causal=ts.num_sites, model=fixed_model, alpha=alpha, random_seed=1
        )

        # We will examine `_obtain_allele_count` method against all possible states
        # at a site to make sure that the code is running correctly
        for state, num in zip(["A", "T"], [3, 3]):
            assert simulator._obtain_allele_count(tree, ts.site(0), state) == num
        for state, num in zip(["A", "T", "C"], [3, 1, 2]):
            assert simulator._obtain_allele_count(tree, ts.site(1), state) == num

        result = simulator._sim_beta([0, 1], ["A", "T"])
        np.testing.assert_array_almost_equal(
            result.beta_array,
            [value * freqdep(alpha, 1 / 2), value * freqdep(alpha, 1 / 6)],
        )
        np.testing.assert_array_almost_equal(result.allele_freq, [1 / 2, 1 / 6])

    def test_allele_freq_one(self):
        """
        Examine `sim_trait` function when allele frequency is 1.
        """
        ts = allele_freq_one()
        value = 2
        fixed_model = tstrait.trait_model(distribution="fixed", value=value)
        trait_df = tstrait.sim_trait(ts, num_causal=1, model=fixed_model, alpha=0)

        np.array_equal(trait_df.site_id, [0])
        np.array_equal(trait_df.effect_size, [value])
        np.array_equal(trait_df.trait_id, [0])
        np.array_equal(trait_df.allele_freq, [1])

    def test_allele_freq_one_freq_dep(self):
        """
        Examine the frequency dependence model when allele frequency is 1 under
        the frequency dependent model.
        """
        ts = allele_freq_one()
        fixed_model = tstrait.trait_model(distribution="fixed", value=1)
        trait_df = tstrait.sim_trait(ts, num_causal=1, model=fixed_model, alpha=-1)

        np.array_equal(trait_df.site_id, [0])
        np.array_equal(trait_df.effect_size, [0])
        np.array_equal(trait_df.trait_id, [0])
        np.array_equal(trait_df.allele_freq, [1])


class TestSimTrait:
    def test_tree_seq(self):
        """
        We will be testing the `sim_trait` function against a simple
        tree sequence data, where we only have 1 mutation on each site.
        """
        ts = simple_tree_seq()
        value = 10
        alpha = -1 / 2
        fixed_model = tstrait.trait_model(distribution="fixed", value=value)
        trait_df = tstrait.sim_trait(
            ts, num_causal=ts.num_sites, model=fixed_model, alpha=alpha
        )
        test_df = pd.DataFrame(
            {
                "position": ts.sites_position[np.arange(3)],
                "site_id": np.arange(3),
                "effect_size": [
                    value * freqdep(alpha, 3 / 4),
                    0,
                    value * freqdep(alpha, 1 / 4),
                ],
                "causal_allele": ["T", "A", "C"],
                "allele_freq": [3 / 4, 1, 1 / 4],
                "trait_id": np.zeros(3),
            }
        )

        pd.testing.assert_frame_equal(trait_df, test_df, check_dtype=False)

        assert trait_df.position.dtype == int
