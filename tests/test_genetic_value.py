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


def freqdep(alpha, freq):
    return np.sqrt(pow(2 * freq * (1 - freq), alpha))


class TestInput:
    """This test will check that an informative error is raised when the input parameter
    does not have an appropriate type or value.
    """

    def test_input_type(self, sample_ts, sample_df):
        with pytest.raises(
            TypeError, match="ts must be a <class 'tskit.trees.TreeSequence'> instance"
        ):
            tstrait.sim_genetic(ts=1, trait_df=sample_df, random_seed=1)
        with pytest.raises(
            TypeError,
            match="trait_df must be a <class 'pandas.core.frame.DataFrame'> instance",
        ):
            tstrait.sim_genetic(ts=sample_ts, trait_df=1, random_seed=1)

    def test_bad_input(self, sample_ts, sample_df):
        with pytest.raises(
            ValueError, match="columns must be included in trait_df dataframe"
        ):
            df = sample_df.drop(columns=["site_id"])
            tstrait.sim_genetic(ts=sample_ts, trait_df=df, random_seed=1)

        with pytest.raises(ValueError, match="site_id must be non-decreasing"):
            df = sample_df.copy()
            df["site_id"] = [2, 0]
            tstrait.sim_genetic(ts=sample_ts, trait_df=df, random_seed=1)

    @pytest.mark.parametrize("trait_id", [[2, 3], [0, 2]])
    def test_trait_id(self, sample_ts, sample_df, trait_id):
        with pytest.raises(
            ValueError, match="trait_id must be consecutive and start from 0"
        ):
            df = sample_df.copy()
            df["trait_id"] = trait_id
            tstrait.sim_genetic(ts=sample_ts, trait_df=df, random_seed=1)


class TestOutputDim:
    """Check that the genetic_value function gives the output with correct dimensions"""

    def check_dimensions(self, result, nrow, input_df):
        genetic_df = result.genetic
        effect_size_df = result.effect_size

        np.testing.assert_array_equal(effect_size_df["site_id"], input_df["site_id"])
        np.testing.assert_array_equal(effect_size_df["trait_id"], input_df["trait_id"])

        assert list(effect_size_df.columns) == [
            "site_id",
            "effect_size",
            "trait_id",
            "causal_state",
            "allele_frequency",
        ]

        assert len(input_df) == len(effect_size_df)

        _check_numeric_array(effect_size_df["effect_size"], "effect_size")

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
        genetic_result = tstrait.sim_genetic(
            ts=sample_ts, trait_df=sample_df, random_seed=5
        )
        self.check_dimensions(genetic_result, sample_ts.num_individuals, sample_df)

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
        genetic_result = tstrait.sim_genetic(
            ts=sample_ts, trait_df=trait_df, random_seed=100
        )
        self.check_dimensions(genetic_result, sample_ts.num_individuals, trait_df)
        np.testing.assert_equal(
            genetic_result.genetic["trait_id"], np.zeros(sample_ts.num_individuals)
        )
        np.testing.assert_equal(
            genetic_result.genetic["individual_id"],
            np.arange(sample_ts.num_individuals),
        )

    def test_output_multivariate(self, sample_ts):
        num_trait = 3
        mean = np.ones(num_trait)
        cov = np.eye(num_trait)
        num_causal = 10
        model = tstrait.trait_model(distribution="multi_normal", mean=mean, cov=cov)
        trait_df = tstrait.sim_trait(ts=sample_ts, num_causal=num_causal, model=model)
        genetic_result = tstrait.sim_genetic(
            ts=sample_ts, trait_df=trait_df, random_seed=6
        )
        self.check_dimensions(
            genetic_result, sample_ts.num_individuals * num_trait, trait_df
        )
        np.testing.assert_equal(
            genetic_result.genetic["individual_id"],
            np.tile(np.arange(sample_ts.num_individuals), 3),
        )
        np.testing.assert_equal(
            genetic_result.genetic["trait_id"],
            np.repeat(np.arange(3), sample_ts.num_individuals),
        )

    def test_additional_row(self, sample_ts, sample_df):
        """Check that adding unexpected rows to trait_df won't cause any errors"""
        df = sample_df.copy()
        df["height"] = [170, 180]
        tstrait.sim_genetic(ts=sample_ts, trait_df=df, random_seed=1000)


class TestGenotype:
    """Test the `_individual_genotype` and `_obtain_allele_count` method and check they
    it can accurately detect the individual genotype. Afterwards, we will check the
    output of `sim_trait`.
    """

    def test_binary_tree(self):
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 2],
                "effect_size": [1, 10],
                "trait_id": [0, 0],
            }
        )

        ts = binary_tree()
        tree = ts.first()
        genetic = tstrait.genetic_value._GeneticValue(
            ts, trait_df, alpha=-1, random_seed=1
        )
        g0 = genetic._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g1 = genetic._individual_genotype(tree, ts.site(1), "T", ts.num_nodes)
        g2 = genetic._individual_genotype(tree, ts.site(2), "C", ts.num_nodes)
        g3 = genetic._individual_genotype(tree, ts.site(3), "C", ts.num_nodes)

        np.testing.assert_equal(g0, np.array([1, 0, 2]))
        np.testing.assert_equal(g1, np.array([1, 1, 0]))
        np.testing.assert_equal(g2, np.array([0, 1, 0]))
        np.testing.assert_equal(g3, np.array([1, 2, 0]))

        c0 = genetic._obtain_allele_count(tree, ts.site(0))
        c1 = genetic._obtain_allele_count(tree, ts.site(1))
        c2 = genetic._obtain_allele_count(tree, ts.site(2))
        c3 = genetic._obtain_allele_count(tree, ts.site(3))

        assert c0 == {"T": 2}
        assert c1 == {"C": 1, "T": 1}
        assert c2 == {"C": 1}
        assert c3 == {"C": 2, "T": 2}

        genetic_result = tstrait.sim_genetic(
            ts=ts, trait_df=trait_df, alpha=-1, random_seed=2
        )
        first = 1 * freqdep(-1, 1 / 2)
        second = 10 * freqdep(-1, 1 / 4)
        effect_size_df = pd.DataFrame(
            {
                "site_id": [0, 2],
                "effect_size": [first, second],
                "trait_id": [0, 0],
                "causal_state": ["T", "C"],
                "allele_frequency": [1 / 2, 1 / 4],
            }
        )
        genetic_df = pd.DataFrame(
            {
                "trait_id": [0, 0, 0],
                "individual_id": [0, 1, 2],
                "genetic_value": [first, second, first * 2],
            }
        )

        pd.testing.assert_frame_equal(
            effect_size_df, genetic_result.effect_size, check_dtype=False
        )
        pd.testing.assert_frame_equal(
            genetic_df, genetic_result.genetic, check_dtype=False
        )

    def test_diff_ind_tree(self):
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 2],
                "effect_size": [1, 10],
                "trait_id": [0, 0],
            }
        )

        ts = diff_ind_tree()
        tree = ts.first()
        genetic = tstrait.genetic_value._GeneticValue(
            ts, trait_df, alpha=0, random_seed=1
        )
        g0 = genetic._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g1 = genetic._individual_genotype(tree, ts.site(1), "T", ts.num_nodes)
        g2 = genetic._individual_genotype(tree, ts.site(2), "C", ts.num_nodes)
        g3 = genetic._individual_genotype(tree, ts.site(3), "C", ts.num_nodes)

        np.testing.assert_equal(g0, np.array([1, 1, 1]))
        np.testing.assert_equal(g1, np.array([1, 0, 1]))
        np.testing.assert_equal(g2, np.array([0, 1, 0]))
        np.testing.assert_equal(g3, np.array([1, 1, 1]))

        c0 = genetic._obtain_allele_count(tree, ts.site(0))
        c1 = genetic._obtain_allele_count(tree, ts.site(1))
        c2 = genetic._obtain_allele_count(tree, ts.site(2))
        c3 = genetic._obtain_allele_count(tree, ts.site(3))

        assert c0 == {"T": 2}
        assert c1 == {"C": 1, "T": 1}
        assert c2 == {"C": 1}
        assert c3 == {"C": 2, "T": 2}

        genetic_result = tstrait.sim_genetic(
            ts=ts, trait_df=trait_df, alpha=-0.5, random_seed=2
        )
        first = 1 * freqdep(-0.5, 1 / 2)
        second = 10 * freqdep(-0.5, 1 / 4)
        effect_size_df = pd.DataFrame(
            {
                "site_id": [0, 2],
                "effect_size": [first, second],
                "trait_id": [0, 0],
                "causal_state": ["T", "C"],
                "allele_frequency": [1 / 2, 1 / 4],
            }
        )
        genetic_df = pd.DataFrame(
            {
                "trait_id": [0, 0, 0],
                "individual_id": [0, 1, 2],
                "genetic_value": [first, first + second, first],
            }
        )

        pd.testing.assert_frame_equal(
            effect_size_df, genetic_result.effect_size, check_dtype=False
        )
        pd.testing.assert_frame_equal(
            genetic_df, genetic_result.genetic, check_dtype=False
        )

    def test_non_binary_tree(self):
        trait_df = pd.DataFrame(
            {
                "site_id": [0],
                "effect_size": [1],
                "trait_id": [0],
            }
        )

        ts = non_binary_tree()
        tree = ts.first()
        genetic = tstrait.genetic_value._GeneticValue(
            ts, trait_df, alpha=-1, random_seed=0
        )
        g0 = genetic._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g1 = genetic._individual_genotype(tree, ts.site(1), "C", ts.num_nodes)

        np.testing.assert_equal(g0, np.array([0, 1, 2]))
        np.testing.assert_equal(g1, np.array([0, 1, 1]))

        c0 = genetic._obtain_allele_count(tree, ts.site(0))
        c1 = genetic._obtain_allele_count(tree, ts.site(1))

        assert c0 == {"T": 3}
        assert c1 == {"C": 2, "T": 1}

        genetic_result = tstrait.sim_genetic(
            ts=ts, trait_df=trait_df, alpha=-1, random_seed=2
        )
        effect_size_df = pd.DataFrame(
            {
                "site_id": [0],
                "effect_size": [freqdep(-1, 1 / 2)],
                "trait_id": [0],
                "causal_state": ["T"],
                "allele_frequency": [1 / 2],
            }
        )
        genetic_df = pd.DataFrame(
            {
                "trait_id": [0, 0, 0],
                "individual_id": [0, 1, 2],
                "genetic_value": [0, freqdep(-1, 1 / 2), 2 * freqdep(-1, 1 / 2)],
            }
        )

        pd.testing.assert_frame_equal(
            effect_size_df, genetic_result.effect_size, check_dtype=False
        )
        pd.testing.assert_frame_equal(
            genetic_df, genetic_result.genetic, check_dtype=False
        )

    def test_triploid(self, sample_df):
        trait_df = pd.DataFrame(
            {
                "site_id": [0],
                "effect_size": [1],
                "trait_id": [0],
            }
        )

        ts = triploid_tree()
        tree = ts.first()
        genetic = tstrait.genetic_value._GeneticValue(
            ts, sample_df, alpha=-1, random_seed=1
        )
        g0 = genetic._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g1 = genetic._individual_genotype(tree, ts.site(1), "C", ts.num_nodes)

        np.testing.assert_equal(g0, np.array([1, 2]))
        np.testing.assert_equal(g1, np.array([1, 1]))

        c0 = genetic._obtain_allele_count(tree, ts.site(0))
        c1 = genetic._obtain_allele_count(tree, ts.site(1))

        assert c0 == {"T": 3}
        assert c1 == {"C": 2, "T": 1}

        genetic_result = tstrait.sim_genetic(
            ts=ts, trait_df=trait_df, alpha=-1, random_seed=2
        )
        effect_size_df = pd.DataFrame(
            {
                "site_id": [0],
                "effect_size": [freqdep(-1, 1 / 2)],
                "trait_id": [0],
                "causal_state": ["T"],
                "allele_frequency": [1 / 2],
            }
        )
        genetic_df = pd.DataFrame(
            {
                "trait_id": [0, 0],
                "individual_id": [0, 1],
                "genetic_value": [freqdep(-1, 1 / 2), 2 * freqdep(-1, 1 / 2)],
            }
        )

        pd.testing.assert_frame_equal(
            effect_size_df, genetic_result.effect_size, check_dtype=False
        )
        pd.testing.assert_frame_equal(
            genetic_df, genetic_result.genetic, check_dtype=False
        )

    def test_allele_freq_one(self, sample_trait_model):
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
            }
        )
        genetic_result = tstrait.sim_genetic(
            ts=ts, trait_df=trait_df, alpha=-1, random_seed=2
        )
        effect_size_df = pd.DataFrame(
            {
                "site_id": [4],
                "effect_size": [0],
                "trait_id": [0],
                "causal_state": ["A"],
                "allele_frequency": [1],
            }
        )
        genetic_df = pd.DataFrame(
            {
                "trait_id": np.zeros(3),
                "individual_id": np.arange(3),
                "genetic_value": np.zeros(3),
            }
        )

        pd.testing.assert_frame_equal(
            effect_size_df, genetic_result.effect_size, check_dtype=False
        )
        pd.testing.assert_frame_equal(
            genetic_df, genetic_result.genetic, check_dtype=False
        )


class TestTreeSeq:
    """Test the `genetic_value` function by using a tree sequence data."""

    def test_tree_seq(self):
        ts = binary_tree_seq()
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 2],
                "effect_size": [1, 10],
                "trait_id": [0, 0],
            }
        )

        genetic_result = tstrait.sim_genetic(
            ts=ts, trait_df=trait_df, alpha=-1, random_seed=2
        )

        first = freqdep(-1, 3 / 4)
        second = 10 * freqdep(-1, 1 / 2)
        effect_size_df = pd.DataFrame(
            {
                "site_id": [0, 2],
                "effect_size": [first, second],
                "trait_id": [0, 0],
                "causal_state": ["T", "C"],
                "allele_frequency": [3 / 4, 1 / 2],
            }
        )
        genetic_df = pd.DataFrame(
            {
                "trait_id": [0, 0],
                "individual_id": [0, 1],
                "genetic_value": [first + second, 2 * first + second],
            }
        )

        pd.testing.assert_frame_equal(
            effect_size_df, genetic_result.effect_size, check_dtype=False
        )
        pd.testing.assert_frame_equal(
            genetic_df, genetic_result.genetic, check_dtype=False
        )

    def test_tree_seq_multiple_trait(self):
        ts = binary_tree_seq()
        trait_df = pd.DataFrame(
            {
                "trait_id": np.tile([0, 1], 2),
                "site_id": np.repeat([0, 2], 2),
                "effect_size": [1, 2, 10, 20],
            }
        )

        genetic_result = tstrait.sim_genetic(
            ts=ts, trait_df=trait_df, alpha=-0.5, random_seed=2
        )

        first = freqdep(-0.5, 3 / 4)
        second = 10 * freqdep(-0.5, 1 / 2)
        effect_size_df = pd.DataFrame(
            {
                "site_id": np.repeat([0, 2], 2),
                "effect_size": [first, first * 2, second, second * 2],
                "trait_id": np.tile([0, 1], 2),
                "causal_state": np.repeat(["T", "C"], 2),
                "allele_frequency": np.repeat([3 / 4, 1 / 2], 2),
            }
        )
        genetic_df = pd.DataFrame(
            {
                "trait_id": np.repeat([0, 1], 2),
                "individual_id": np.tile([0, 1], 2),
                "genetic_value": [
                    first + second,
                    2 * first + second,
                    2 * (first + second),
                    2 * (2 * first + second),
                ],
            }
        )

        pd.testing.assert_frame_equal(
            effect_size_df, genetic_result.effect_size, check_dtype=False
        )
        pd.testing.assert_frame_equal(
            genetic_df, genetic_result.genetic, check_dtype=False
        )
