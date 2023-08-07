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
            "causal_state": ["A", "A"],
            "effect_size": [0.1, 0.1],
            "trait_id": [0, 0],
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


class TestOutputDim:
    """Check that the genetic_value function gives the output with correct dimensions"""

    def check_dimensions(self, df, nrow):
        assert len(df) == nrow
        assert df.shape[1] == 3
        assert list(df.columns) == [
            "trait_id",
            "individual_id",
            "genetic_value",
        ]
        _check_numeric_array(df["individual_id"], "individual_id")
        _check_numeric_array(df["genetic_value"], "genetic_value")
        _check_numeric_array(df["trait_id"], "trait_id")

    def test_output_simple(self, sample_ts, sample_df):
        genetic_df = tstrait.genetic_value(ts=sample_ts, trait_df=sample_df)
        self.check_dimensions(genetic_df, sample_ts.num_individuals)

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
            ts=sample_ts, num_causal=num_causal, model=trait_model
        )
        genetic_df = tstrait.genetic_value(ts=sample_ts, trait_df=trait_df)
        self.check_dimensions(genetic_df, sample_ts.num_individuals)
        np.testing.assert_equal(
            genetic_df["trait_id"], np.zeros(sample_ts.num_individuals)
        )
        np.testing.assert_equal(
            genetic_df["individual_id"], np.arange(sample_ts.num_individuals)
        )

    def test_output_multivariate(self, sample_ts):
        num_trait = 3
        mean = np.ones(num_trait)
        cov = np.eye(num_trait)
        num_causal = 10
        model = tstrait.trait_model(distribution="multi_normal", mean=mean, cov=cov)
        trait_df = tstrait.sim_trait(ts=sample_ts, num_causal=num_causal, model=model)
        genetic_df = tstrait.genetic_value(ts=sample_ts, trait_df=trait_df)
        self.check_dimensions(genetic_df, sample_ts.num_individuals * num_trait)
        ind_id = np.arange(sample_ts.num_individuals)
        trait_id = np.ones(sample_ts.num_individuals)
        np.testing.assert_equal(
            genetic_df["individual_id"], np.concatenate((ind_id, ind_id, ind_id))
        )
        np.testing.assert_equal(
            genetic_df["trait_id"],
            np.concatenate((trait_id * 0, trait_id * 1, trait_id * 2)),
        )

    def test_additional_row(self, sample_ts, sample_df):
        """Check that adding unexpected rows to trait_df won't cause any errors"""
        df = sample_df.copy()
        df["height"] = [170, 180]
        tstrait.genetic_value(ts=sample_ts, trait_df=df)


class TestGenotype:
    """Test the `_individual_genotype` method and check if it can accurately detect
    the individual genotype. Afterwards, we will check the output of `sim_trait`.
    """

    def test_binary_tree(self):
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 1, 2, 3],
                "causal_state": ["T", "T", "C", "C"],
                "effect_size": [1, 10, 100, 1000],
                "trait_id": [0, 0, 0, 0],
            }
        )

        ts = binary_tree()
        tree = ts.first()
        genetic = tstrait.GeneticValue(ts, trait_df)
        g0 = genetic._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g1 = genetic._individual_genotype(tree, ts.site(1), "T", ts.num_nodes)
        g2 = genetic._individual_genotype(tree, ts.site(2), "C", ts.num_nodes)
        g3 = genetic._individual_genotype(tree, ts.site(3), "C", ts.num_nodes)

        np.testing.assert_equal(g0, np.array([1, 0, 2]))
        np.testing.assert_equal(g1, np.array([1, 1, 0]))
        np.testing.assert_equal(g2, np.array([0, 1, 0]))
        np.testing.assert_equal(g3, np.array([1, 2, 0]))

        genetic_df = tstrait.genetic_value(ts=ts, trait_df=trait_df)

        result_df = pd.DataFrame(
            {
                "trait_id": [0, 0, 0],
                "individual_id": [0, 1, 2],
                "genetic_value": [1011, 2110, 2],
            }
        )

        pd.testing.assert_frame_equal(genetic_df, result_df, check_dtype=False)

    def test_diff_ind_tree(self):
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 1, 2, 3],
                "causal_state": ["T", "T", "C", "C"],
                "effect_size": [1, 10, 100, 1000],
                "trait_id": [0, 0, 0, 0],
            }
        )

        ts = diff_ind_tree()
        tree = ts.first()
        genetic = tstrait.GeneticValue(ts, trait_df)
        g0 = genetic._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g1 = genetic._individual_genotype(tree, ts.site(1), "T", ts.num_nodes)
        g2 = genetic._individual_genotype(tree, ts.site(2), "C", ts.num_nodes)
        g3 = genetic._individual_genotype(tree, ts.site(3), "C", ts.num_nodes)

        np.testing.assert_equal(g0, np.array([1, 1, 1]))
        np.testing.assert_equal(g1, np.array([1, 0, 1]))
        np.testing.assert_equal(g2, np.array([0, 1, 0]))
        np.testing.assert_equal(g3, np.array([1, 1, 1]))

        genetic_df = tstrait.genetic_value(ts=ts, trait_df=trait_df)

        result_df = pd.DataFrame(
            {
                "trait_id": [0, 0, 0],
                "individual_id": [0, 1, 2],
                "genetic_value": [1011, 1101, 1011],
            }
        )

        pd.testing.assert_frame_equal(genetic_df, result_df, check_dtype=False)

    def test_non_binary_tree(self):
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 1],
                "causal_state": ["T", "C"],
                "effect_size": [1, 10],
                "trait_id": [0, 0],
            }
        )

        ts = non_binary_tree()
        tree = ts.first()
        genetic = tstrait.GeneticValue(ts, trait_df)
        g0 = genetic._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g1 = genetic._individual_genotype(tree, ts.site(1), "C", ts.num_nodes)

        np.testing.assert_equal(g0, np.array([0, 1, 2]))
        np.testing.assert_equal(g1, np.array([0, 1, 1]))

        genetic_df = tstrait.genetic_value(ts=ts, trait_df=trait_df)

        result_df = pd.DataFrame(
            {
                "trait_id": [0, 0, 0],
                "individual_id": [0, 1, 2],
                "genetic_value": [0, 11, 12],
            }
        )

        pd.testing.assert_frame_equal(genetic_df, result_df, check_dtype=False)

    def test_triploid(self, sample_df):
        trait_df = pd.DataFrame(
            {
                "trait_id": [0, 0],
                "site_id": [0, 1],
                "causal_state": ["T", "C"],
                "effect_size": [1, 10],
            }
        )

        ts = triploid_tree()
        tree = ts.first()
        genetic = tstrait.GeneticValue(ts, sample_df)
        g0 = genetic._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g1 = genetic._individual_genotype(tree, ts.site(1), "C", ts.num_nodes)

        np.testing.assert_equal(g0, np.array([1, 2]))
        np.testing.assert_equal(g1, np.array([1, 1]))

        genetic_df = tstrait.genetic_value(ts=ts, trait_df=trait_df)

        result_df = pd.DataFrame(
            {
                "trait_id": [0, 0],
                "individual_id": [0, 1],
                "genetic_value": [11, 12],
            }
        )

        pd.testing.assert_frame_equal(genetic_df, result_df, check_dtype=False)


class TestTreeSeq:
    """Test the `genetic_value` function by using a tree sequence data."""

    def test_tree_seq(self):
        ts = binary_tree_seq()
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 1, 2],
                "causal_state": ["T", "G", "C"],
                "effect_size": [1, 10, 100],
                "trait_id": [0, 0, 0],
            }
        )

        genetic_df = tstrait.genetic_value(ts=ts, trait_df=trait_df)

        result_df = pd.DataFrame(
            {
                "trait_id": [0, 0],
                "individual_id": [0, 1],
                "genetic_value": [101, 122],
            }
        )

        pd.testing.assert_frame_equal(genetic_df, result_df, check_dtype=False)

    def test_tree_seq_multiple_trait(self):
        ts = binary_tree_seq()
        trait_df = pd.DataFrame(
            {
                "trait_id": np.tile([0, 1], 3),
                "site_id": np.repeat([0, 1, 2], 2),
                "causal_state": np.repeat(["T", "G", "C"], 2),
                "effect_size": [1, 2, 10, 20, 100, 200],
            }
        )

        genetic_df = tstrait.genetic_value(ts=ts, trait_df=trait_df)

        result_df = pd.DataFrame(
            {
                "trait_id": [0, 0, 1, 1],
                "individual_id": [0, 1, 0, 1],
                "genetic_value": [101, 122, 202, 244],
            }
        )

        pd.testing.assert_frame_equal(genetic_df, result_df, check_dtype=False)
