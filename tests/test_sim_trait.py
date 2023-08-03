import msprime
import numpy as np
import pytest
import scipy
import tskit
import tstrait
from tstrait.base import _check_numeric_array
from .data import (
    binary_tree,
    diff_ind_tree,
    non_binary_tree,
    binary_tree_seq,
)  # noreorder


@pytest.fixture(scope="class")
def sample_ts():
    ts = msprime.sim_ancestry(10, sequence_length=100_000, random_seed=1)
    ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
    return ts


@pytest.fixture(scope="class")
def sample_trait_model():
    return tstrait.trait_model(distribution="normal", mean=0, var=1)


@pytest.fixture(scope="class")
def bad_type_param():
    """This object does not have a valid type for sure for all parameters. It will be
    used to check that the error is raised properly when the parameter doesn't match
    any valid type.
    """
    return type("BadType", (), {})()


class TestInput:
    """This test will check that an informative error is raised when the input parameter
    does not have an appropriate type or value.
    """

    def test_input_type(self, sample_ts, sample_trait_model, bad_type_param):
        with pytest.raises(
            TypeError, match="ts must be a <class 'tskit.trees.TreeSequence'> instance"
        ):
            tstrait.sim_trait(ts=bad_type_param, num_causal=1, model=sample_trait_model)
        with pytest.raises(TypeError, match="num_causal must be an integer"):
            tstrait.sim_trait(
                ts=sample_ts, num_causal=bad_type_param, model=sample_trait_model
            )
        with pytest.raises(
            TypeError,
            match="model must be a <class 'tstrait.trait_model.TraitModel'> instance",
        ):
            tstrait.sim_trait(ts=sample_ts, num_causal=1, model=bad_type_param)
        with pytest.raises(TypeError, match="alpha must be numeric"):
            tstrait.sim_trait(
                ts=sample_ts,
                num_causal=1,
                model=sample_trait_model,
                alpha=bad_type_param,
            )
        with pytest.raises(
            ValueError,
            match=f"{bad_type_param!r} cannot be used to construct a new random "
            "generator with numpy.random.default_rng",
        ):
            tstrait.sim_trait(
                ts=sample_ts,
                num_causal=1,
                model=sample_trait_model,
                random_seed=bad_type_param,
            )

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
        with pytest.raises(
            ValueError, match="num_causal must be an integer not less than 0"
        ):
            tstrait.sim_trait(ts=sample_ts, num_causal=-1, model=sample_trait_model)


class TestOutputDim:
    """Check that the sim_trait function gives the correct output regardless of the
    tree sequence mutation type or the trait model
    """

    def check_dimensions(self, df, num_causal):
        assert len(df) == num_causal
        assert df.shape[1] == 4
        assert list(df.columns) == [
            "site_id",
            "causal_state",
            "effect_size",
            "trait_id",
        ]
        _check_numeric_array(df["site_id"], "site_id")
        _check_numeric_array(df["effect_size"], "effect_size")
        _check_numeric_array(df["trait_id"], "trait_id")

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
            alpha=0,
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


class Test_allele_freq:
    @pytest.mark.parametrize("tree_seq", [binary_tree(), diff_ind_tree()])
    def test_binary_tree(self, tree_seq, sample_trait_model):
        simulator = tstrait.TraitSimulator(
            ts=tree_seq,
            num_causal=4,
            model=sample_trait_model,
            alpha=0.3,
            random_seed=1,
        )
        tree = tree_seq.first()
        c0 = simulator._obtain_allele_count(tree, tree_seq.site(0))
        c1 = simulator._obtain_allele_count(tree, tree_seq.site(1))
        c2 = simulator._obtain_allele_count(tree, tree_seq.site(2))
        c3 = simulator._obtain_allele_count(tree, tree_seq.site(3))

        assert c0 == {"T": 2}
        assert c1 == {"C": 1, "T": 1}
        assert c2 == {"C": 1}
        assert c3 == {"C": 2, "T": 2}

    def non_binary_tree(self, sample_trait_model):
        ts = non_binary_tree()
        simulator = tstrait.TraitSimulator(
            ts=ts, num_causal=4, model=sample_trait_model, alpha=0.3, random_seed=1
        )
        tree = ts.first()
        c0 = simulator._obtain_allele_count(tree, ts.site(0))
        c1 = simulator._obtain_allele_count(tree, ts.site(1))

        assert c0 == {"T": 3}
        assert c1 == {"C": 2, "T": 1}


class TestAlleleFreq:
    """Test the allele frequency dependence model by using a fixed trait model.
    We will be using a tree sequence to also test that the simulation model works
    in a tree sequence data.
    """

    def freqdep(self, alpha, freq):
        return np.sqrt(pow(2 * freq * (1 - freq), alpha))

    @pytest.mark.parametrize("alpha", [0, -1])
    def test_fixed_freq_dependence(self, alpha):
        ts = binary_tree_seq()
        num_causal = ts.num_sites
        value = 2
        model = tstrait.trait_model(distribution="fixed", value=value)
        sim_result = tstrait.sim_trait(
            ts=ts, num_causal=num_causal, model=model, alpha=alpha
        )

        np.testing.assert_equal(sim_result["site_id"], np.arange(num_causal))
        np.testing.assert_equal(sim_result["trait_id"], np.zeros(num_causal))
        assert np.all(sim_result["causal_state"] != "A")

        df0 = sim_result.loc[sim_result.site_id == 0]
        df2 = sim_result.loc[sim_result.site_id == 2]

        assert df0["causal_state"].values[0] == "T"
        assert df0["effect_size"].values[0] == value * self.freqdep(alpha, 0.75)
        assert df2["causal_state"].values[0] == "C"
        assert df2["effect_size"].values[0] == value * self.freqdep(alpha, 0.5)

    def test_allele_freq_one(self, sample_trait_model):
        ts = tskit.Tree.generate_comb(6, span=2).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=8, derived_state="T")
        tables.mutations.add_row(site=0, node=8, derived_state="A", parent=0)
        ts = tables.tree_sequence()
        sim_result = tstrait.sim_trait(ts=ts, num_causal=1, model=sample_trait_model)
        np.testing.assert_equal(sim_result["effect_size"], np.array([0]))


def model_list():
    num_causal = 1000
    loc = 2
    scale = 5
    df = 10
    shape = 5
    distr = [
        (
            tstrait.trait_model(distribution="normal", mean=loc, var=scale**2),
            "norm",
            (loc / num_causal, scale / num_causal),
        ),
        (
            tstrait.trait_model(distribution="exponential", scale=scale),
            "expon",
            (0, scale / num_causal),
        ),
        (
            tstrait.trait_model(distribution="t", mean=loc, var=scale**2, df=df),
            "t",
            (df, loc / num_causal, scale / num_causal),
        ),
        (
            tstrait.trait_model(distribution="gamma", shape=shape, scale=scale),
            "gamma",
            (shape, 0, scale / num_causal),
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
        D, pval = scipy.stats.kstest(rvs, dist, args=args, N=1000)
        if pval < 0.05:
            raise ValueError(f"KS test failed for distribution {dist}")

    @pytest.mark.parametrize("model, distr, args", model_list())
    def test_KStest(self, model, distr, args, sample_ts):
        sim_result = tstrait.sim_trait(
            ts=sample_ts, num_causal=1000, model=model, random_seed=2
        )
        self.check_distribution(sim_result["effect_size"], distr, args)

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
        sim_result = tstrait.sim_trait(ts=sample_ts, num_causal=num_causal, model=model)

        const = np.random.randn(n)
        data_val = np.matmul(const, cov)
        data_sd = np.sqrt(np.matmul(data_val, const))
        sum_data = np.zeros(num_causal)

        for i in range(n):
            df = sim_result.loc[sim_result.trait_id == i]
            self.check_distribution(
                df["effect_size"],
                "norm",
                (mean[i] / num_causal, np.sqrt(cov[i, i]) / num_causal),
            )
            sum_data += df["effect_size"] * const[i]

        self.check_distribution(
            sum_data,
            "norm",
            (np.matmul(const, mean) / num_causal, data_sd / num_causal),
        )
