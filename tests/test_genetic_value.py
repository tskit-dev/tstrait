import msprime
import numpy as np
import pandas as pd
import pytest
import tstrait
from tstrait.base import _check_numeric_array
from .data import (
    binary_tree,
    binary_tree_seq,
    diff_ind_tree,
    non_binary_tree,
    triploid_tree,
)  # noreorder
from tstrait.genetic_value import _GeneticValue

# from tstrait.trait import sim_trait


@pytest.fixture(scope="class")
def sample_ts():
    ts = msprime.sim_ancestry(10, sequence_length=100_000, random_seed=1)
    ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
    return ts


@pytest.fixture(scope="class")
def sample_df():
    df = pd.DataFrame(
        {
            "site_id": [0, 1],
            "causal_allele": ["A", "A"],
            "effect_size": [0.1, 0.1],
            "trait_id": [0, 0],
            "allele_freq": [0.2, 0.3],
        }
    )

    return df


@pytest.fixture(scope="class")
def sample_trait_model():
    return tstrait.trait_model(distribution="normal", mean=0, var=1)


class TestInput:
    """This test will check that an informative error is raised when the input parameter
    does not have an appropriate type or value.
    """

    def test_input_type(self, sample_ts, sample_df):
        with pytest.raises(
            TypeError, match="ts must be a <class 'tskit.trees.TreeSequence'> instance"
        ):
            tstrait.genetic_value(ts=1, trait_df=sample_df)
        with pytest.raises(
            TypeError,
            match="trait_df must be a <class 'pandas.core.frame.DataFrame'> instance",
        ):
            tstrait.genetic_value(ts=sample_ts, trait_df=1)

    def test_bad_input(self, sample_ts, sample_df):
        with pytest.raises(
            ValueError, match="columns must be included in trait_df dataframe"
        ):
            df = sample_df.drop(columns=["site_id"])
            tstrait.genetic_value(ts=sample_ts, trait_df=df)

        with pytest.raises(
            ValueError, match="columns must be included in trait_df dataframe"
        ):
            df = sample_df.drop(columns=["effect_size"])
            tstrait.genetic_value(ts=sample_ts, trait_df=df)

        with pytest.raises(ValueError, match="site_id must be non-decreasing"):
            df = sample_df.copy()
            df["site_id"] = [2, 0]
            tstrait.genetic_value(ts=sample_ts, trait_df=df)

    @pytest.mark.parametrize("trait_id", [[2, 3], [0, 2]])
    def test_trait_id(self, sample_ts, sample_df, trait_id):
        with pytest.raises(
            ValueError, match="trait_id must be consecutive and start from 0"
        ):
            df = sample_df.copy()
            df["trait_id"] = trait_id
            tstrait.genetic_value(ts=sample_ts, trait_df=df)

    def test_no_individual(self, sample_df):
        ts = msprime.simulate(100, length=10000, mutation_rate=1e-2)
        df = sample_df.copy()
        with pytest.raises(
            ValueError, match="No individuals in the provided tree sequence dataset"
        ):
            tstrait.genetic_value(ts=ts, trait_df=df)


class TestOutputDim:
    """Check that the genetic_value function gives the output with correct dimensions"""

    def check_dimensions(self, genetic_df, nrow):
        assert len(genetic_df) == nrow
        assert genetic_df.shape[1] == 3
        assert list(genetic_df.columns) == [
            "trait_id",
            "individual_id",
            "genetic_value",
        ]

        _check_numeric_array(genetic_df["individual_id"], "individual_id")
        _check_numeric_array(genetic_df["genetic_value"], "genetic_value")
        _check_numeric_array(genetic_df["trait_id"], "trait_id")

    def test_output_simple(self, sample_ts, sample_df):
        genetic_result = tstrait.genetic_value(ts=sample_ts, trait_df=sample_df)
        self.check_dimensions(genetic_result, sample_ts.num_individuals)

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
    def test_output_sim_trait(self, sample_ts, trait_model):
        num_causal = 10
        trait_df = tstrait.sim_trait(
            ts=sample_ts, num_causal=num_causal, model=trait_model, random_seed=10
        )
        genetic_result = tstrait.genetic_value(ts=sample_ts, trait_df=trait_df)
        self.check_dimensions(genetic_result, sample_ts.num_individuals)
        np.testing.assert_equal(
            genetic_result["trait_id"], np.zeros(sample_ts.num_individuals)
        )
        np.testing.assert_equal(
            genetic_result["individual_id"],
            np.arange(sample_ts.num_individuals),
        )

    def test_output_multivariate(self, sample_ts):
        num_trait = 3
        mean = np.ones(num_trait)
        cov = np.eye(num_trait)
        num_causal = 10
        model = tstrait.trait_model(distribution="multi_normal", mean=mean, cov=cov)
        trait_df = tstrait.sim_trait(
            ts=sample_ts, num_causal=num_causal, model=model, random_seed=5
        )
        genetic_result = tstrait.genetic_value(ts=sample_ts, trait_df=trait_df)
        self.check_dimensions(genetic_result, sample_ts.num_individuals * num_trait)
        np.testing.assert_equal(
            genetic_result["individual_id"],
            np.tile(np.arange(sample_ts.num_individuals), 3),
        )
        np.testing.assert_equal(
            genetic_result["trait_id"],
            np.repeat(np.arange(3), sample_ts.num_individuals),
        )

    def test_additional_row(self, sample_ts, sample_df):
        """Check that adding unexpected rows to trait_df won't cause any errors"""
        df = sample_df.copy()
        df["height"] = [170, 180]
        tstrait.genetic_value(ts=sample_ts, trait_df=df)


class TestGenotype:
    """Test the `_individual_genetic_values` and `_obtain_allele_count` method and
    check if they can accurately detect the individual genotype. Afterwards, we will
    check the output of `genetic_value`.
    """

    def test_binary_tree(self):
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 2, 3],
                "effect_size": [1, 10, 100],
                "trait_id": [0, 0, 0],
                "causal_allele": ["T", "C", "T"],
            }
        )

        ts = binary_tree()
        tree = ts.first()
        genetic = _GeneticValue(ts, trait_df)
        g0 = genetic._individual_genetic_values(tree, ts.site(0), "T", 1)
        g1 = genetic._individual_genetic_values(tree, ts.site(1), "T", 2)
        g2 = genetic._individual_genetic_values(tree, ts.site(2), "C", 3)
        g3 = genetic._individual_genetic_values(tree, ts.site(3), "C", 4)

        np.testing.assert_equal(g0, np.array([1, 0, 2]) * 1)
        np.testing.assert_equal(g1, np.array([1, 1, 0]) * 2)
        np.testing.assert_equal(g2, np.array([0, 1, 0]) * 3)
        np.testing.assert_equal(g3, np.array([1, 2, 0]) * 4)

        genetic_result = tstrait.genetic_value(ts=ts, trait_df=trait_df)

        genetic_df = pd.DataFrame(
            {
                "trait_id": [0, 0, 0],
                "individual_id": [0, 1, 2],
                "genetic_value": [101, 10, 202],
            }
        )

        pd.testing.assert_frame_equal(genetic_df, genetic_result, check_dtype=False)

    def test_diff_ind_tree(self):
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 2],
                "effect_size": [1, 10],
                "trait_id": [0, 0],
                "causal_allele": ["T", "C"],
            }
        )

        ts = diff_ind_tree()
        tree = ts.first()
        genetic = _GeneticValue(ts, trait_df)
        g0 = genetic._individual_genetic_values(tree, ts.site(0), "T", 1)
        g1 = genetic._individual_genetic_values(tree, ts.site(1), "T", 2)
        g2 = genetic._individual_genetic_values(tree, ts.site(2), "C", 3)
        g3 = genetic._individual_genetic_values(tree, ts.site(3), "C", 4)

        np.testing.assert_equal(g0, np.array([1, 1, 1]) * 1)
        np.testing.assert_equal(g1, np.array([1, 0, 1]) * 2)
        np.testing.assert_equal(g2, np.array([0, 1, 0]) * 3)
        np.testing.assert_equal(g3, np.array([1, 1, 1]) * 4)

        genetic_result = tstrait.genetic_value(ts=ts, trait_df=trait_df)
        genetic_df = pd.DataFrame(
            {
                "trait_id": [0, 0, 0],
                "individual_id": [0, 1, 2],
                "genetic_value": [1, 11, 1],
            }
        )

        pd.testing.assert_frame_equal(genetic_df, genetic_result, check_dtype=False)

    def test_non_binary_tree(self):
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 1],
                "effect_size": [1, 10],
                "trait_id": [0, 0],
                "causal_allele": ["A", "T"],
            }
        )

        ts = non_binary_tree()
        tree = ts.first()
        genetic = _GeneticValue(ts, trait_df)
        g0 = genetic._individual_genetic_values(tree, ts.site(0), "T", 1)
        g1 = genetic._individual_genetic_values(tree, ts.site(1), "C", 2)

        np.testing.assert_equal(g0, np.array([0, 1, 2]) * 1)
        np.testing.assert_equal(g1, np.array([0, 1, 1]) * 2)

        genetic_result = tstrait.genetic_value(ts=ts, trait_df=trait_df)

        genetic_df = pd.DataFrame(
            {
                "trait_id": [0, 0, 0],
                "individual_id": [0, 1, 2],
                "genetic_value": [2, 1, 10],
            }
        )

        pd.testing.assert_frame_equal(genetic_df, genetic_result, check_dtype=False)

    def test_triploid(self, sample_df):
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 1],
                "effect_size": [1, 10],
                "trait_id": [0, 0],
                "causal_allele": ["T", "C"],
            }
        )

        ts = triploid_tree()
        tree = ts.first()
        genetic = _GeneticValue(ts, sample_df)
        g0 = genetic._individual_genetic_values(tree, ts.site(0), "T", 1)
        g1 = genetic._individual_genetic_values(tree, ts.site(1), "C", 2)

        np.testing.assert_equal(g0, np.array([1, 2]) * 1)
        np.testing.assert_equal(g1, np.array([1, 1]) * 2)

        genetic_result = tstrait.genetic_value(ts=ts, trait_df=trait_df)
        genetic_df = pd.DataFrame(
            {
                "trait_id": [0, 0],
                "individual_id": [0, 1],
                "genetic_value": [11, 12],
            }
        )
        pd.testing.assert_frame_equal(genetic_df, genetic_result, check_dtype=False)

    def test_allele_freq_one(self):
        ts = binary_tree()
        tables = ts.dump_tables()
        tables.sites.add_row(4, "A")
        tables.mutations.add_row(site=4, node=0, derived_state="T")
        tables.mutations.add_row(site=4, node=0, derived_state="A", parent=9)
        ts = tables.tree_sequence()
        trait_df = pd.DataFrame(
            {
                "site_id": [4],
                "effect_size": [1],
                "trait_id": [0],
                "causal_allele": ["A"],
            }
        )
        genetic_result = tstrait.genetic_value(ts=ts, trait_df=trait_df)
        genetic_df = pd.DataFrame(
            {
                "trait_id": np.zeros(3),
                "individual_id": np.arange(3),
                "genetic_value": np.ones(3) * 2,
            }
        )
        pd.testing.assert_frame_equal(genetic_df, genetic_result, check_dtype=False)


class TestTreeSeq:
    """Test the `genetic_value` function by using a tree sequence data."""

    def test_tree_seq(self):
        ts = binary_tree_seq()
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 1, 2],
                "effect_size": [1, 10, 100],
                "causal_allele": ["T", "G", "T"],
                "trait_id": [0, 0, 0],
            }
        )

        genetic_result = tstrait.genetic_value(ts=ts, trait_df=trait_df)

        genetic_df = pd.DataFrame(
            {
                "trait_id": [0, 0],
                "individual_id": [0, 1],
                "genetic_value": [1, 22],
            }
        )

        pd.testing.assert_frame_equal(genetic_df, genetic_result, check_dtype=False)

    def test_tree_seq_multiple_trait(self):
        ts = binary_tree_seq()
        trait_df = pd.DataFrame(
            {
                "trait_id": [0, 1, 0, 1],
                "site_id": [0, 0, 2, 2],
                "effect_size": [1, 2, 10, 20],
                "causal_allele": ["T", "T", "C", "C"],
            }
        )

        genetic_result = tstrait.genetic_value(ts=ts, trait_df=trait_df)

        genetic_df = pd.DataFrame(
            {
                "trait_id": [0, 0, 1, 1],
                "individual_id": [0, 1, 0, 1],
                "genetic_value": [11, 12, 22, 24],
            }
        )

        pd.testing.assert_frame_equal(genetic_df, genetic_result, check_dtype=False)


class TestNormaliseGenetic:
    def test_output(self, sample_ts):
        mean = 2
        var = 4
        model = tstrait.trait_model(distribution="normal", mean=2, var=6)
        trait_df = tstrait.sim_trait(
            ts=sample_ts, num_causal=20, model=model, random_seed=500
        )
        genetic_df = tstrait.genetic_value(sample_ts, trait_df)

        normalised_df = tstrait.normalise_genetic_value(genetic_df, mean=mean, var=var)
        genetic_array = normalised_df["genetic_value"].values
        np.testing.assert_almost_equal(np.mean(genetic_array), mean, decimal=2)
        np.testing.assert_almost_equal(np.var(genetic_array, ddof=1), var, decimal=2)
        pd.testing.assert_series_equal(
            normalised_df["trait_id"], genetic_df["trait_id"]
        )
        pd.testing.assert_series_equal(
            normalised_df["individual_id"], genetic_df["individual_id"]
        )

        num_ind = sample_ts.num_individuals
        assert len(normalised_df) == num_ind
        assert normalised_df.shape[1] == 3
        assert list(normalised_df.columns) == [
            "individual_id",
            "trait_id",
            "genetic_value",
        ]

    def test_default(self, sample_ts):
        mean = 0
        var = 1
        model = tstrait.trait_model(distribution="normal", mean=2, var=6)
        trait_df = tstrait.sim_trait(
            ts=sample_ts, num_causal=20, model=model, random_seed=1000
        )
        genetic_df = tstrait.genetic_value(sample_ts, trait_df)
        normalised_df = tstrait.normalise_genetic_value(genetic_df)
        genetic_array = normalised_df["genetic_value"].values
        np.testing.assert_almost_equal(np.mean(genetic_array), mean, decimal=2)
        np.testing.assert_almost_equal(np.var(genetic_array, ddof=1), var, decimal=2)
        pd.testing.assert_series_equal(
            normalised_df["trait_id"], genetic_df["trait_id"]
        )
        pd.testing.assert_series_equal(
            normalised_df["individual_id"], genetic_df["individual_id"]
        )

        num_ind = sample_ts.num_individuals
        assert len(normalised_df) == num_ind
        assert normalised_df.shape[1] == 3
        assert list(normalised_df.columns) == [
            "individual_id",
            "trait_id",
            "genetic_value",
        ]

    def test_column(self, sample_ts):
        model = tstrait.trait_model(distribution="normal", mean=2, var=6)
        trait_df = tstrait.sim_trait(
            ts=sample_ts, num_causal=20, model=model, random_seed=1000
        )
        genetic_df = tstrait.genetic_value(sample_ts, trait_df)
        with pytest.raises(
            ValueError, match="columns must be included in genetic_df dataframe"
        ):
            tstrait.normalise_genetic_value(genetic_df[["trait_id", "individual_id"]])

        with pytest.raises(
            ValueError, match="columns must be included in genetic_df dataframe"
        ):
            tstrait.normalise_genetic_value(genetic_df[["trait_id", "genetic_value"]])

        with pytest.raises(
            ValueError, match="columns must be included in genetic_df dataframe"
        ):
            tstrait.normalise_genetic_value(
                genetic_df[["genetic_value", "individual_id"]]
            )

    @pytest.mark.parametrize("var", [0, -1])
    def test_negative_var(self, sample_ts, var):
        model = tstrait.trait_model(distribution="normal", mean=2, var=6)
        trait_df = tstrait.sim_trait(
            ts=sample_ts, num_causal=20, model=model, random_seed=1000
        )
        genetic_df = tstrait.genetic_value(sample_ts, trait_df)

        with pytest.raises(ValueError, match="Variance must be greater than 0."):
            tstrait.normalise_genetic_value(genetic_df, var=var)

    @pytest.mark.parametrize("ddof", [0, 1])
    def test_ddof(self, sample_ts, ddof):
        model = tstrait.trait_model(distribution="normal", mean=2, var=6)
        trait_df = tstrait.sim_trait(
            ts=sample_ts, num_causal=20, model=model, random_seed=1000
        )
        genetic_df = tstrait.genetic_value(sample_ts, trait_df)
        normalised_df = tstrait.normalise_genetic_value(
            genetic_df, mean=0, var=1, ddof=ddof
        )
        normalised_genetic_array = normalised_df["genetic_value"].values

        genetic_array = genetic_df["genetic_value"].values
        genetic_array = (genetic_array - np.mean(genetic_array)) / np.std(
            genetic_array, ddof=ddof
        )

        np.testing.assert_array_almost_equal(normalised_genetic_array, genetic_array)

    def test_pleiotropy(self, sample_ts):
        mean = 0
        var = 1
        model = tstrait.trait_model(
            distribution="multi_normal", mean=np.zeros(2), cov=np.identity(2)
        )
        trait_df = tstrait.sim_trait(
            ts=sample_ts, num_causal=20, model=model, random_seed=1000
        )
        genetic_df = tstrait.genetic_value(sample_ts, trait_df)
        normalised_df = tstrait.normalise_genetic_value(genetic_df, mean=mean, var=var)
        grouped = normalised_df.groupby(["trait_id"])[["genetic_value"]]
        mean_array = grouped.mean().values.T[0]
        var_array = grouped.var().values.T[0]
        np.testing.assert_almost_equal(mean_array, np.zeros(2), decimal=2)
        np.testing.assert_almost_equal(var_array, np.ones(2), decimal=2)
