import msprime
import numpy as np
import pandas as pd
import pytest
import tstrait


@pytest.fixture(scope="class")
def sample_ts():
    ts = msprime.sim_ancestry(10, sequence_length=100_000, random_seed=1)
    ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
    return ts


@pytest.fixture(scope="class")
def sample_trait_model():
    return tstrait.trait_model(distribution="normal", mean=0, var=1)


class TestInput:
    """This test will check that an informative error is raised when the input parameter
    does not have an appropriate type or value.
    """

    def test_input_type(self, sample_ts, sample_trait_model):
        with pytest.raises(
            TypeError, match="ts must be a <class 'tskit.trees.TreeSequence'> instance"
        ):
            tstrait.sim_phenotype(ts=1, num_causal=1, model=sample_trait_model, h2=0.3)
        with pytest.raises(
            TypeError,
            match="model must be a <class 'tstrait.trait_model.TraitModel'> instance",
        ):
            tstrait.sim_phenotype(ts=sample_ts, num_causal=1, model=1, h2=0.3)

    def test_bad_input(self, sample_ts, sample_trait_model):
        ts = msprime.sim_ancestry(2, sequence_length=100_000, random_seed=1)
        with pytest.raises(ValueError, match="No mutation in the tree sequence input"):
            tstrait.sim_phenotype(ts=ts, num_causal=1, model=sample_trait_model, h2=0.3)
        num_causal = sample_ts.num_sites + 1
        with pytest.raises(
            ValueError,
            match="num_causal must be an integer not greater than the number of sites "
            "in ts",
        ):
            tstrait.sim_phenotype(
                ts=sample_ts, num_causal=num_causal, model=sample_trait_model, h2=0.3
            )

    def test_bad_input_h2(self, sample_ts, sample_trait_model):
        with pytest.raises(
            ValueError, match="Narrow-sense heritability must be 0 < h2 <= 1"
        ):
            tstrait.sim_phenotype(
                ts=sample_ts, num_causal=5, model=sample_trait_model, h2=-0.1
            )
        with pytest.raises(
            ValueError, match="Narrow-sense heritability must be 0 < h2 <= 1"
        ):
            trait_model = tstrait.trait_model(
                distribution="multi_normal", mean=np.zeros(2), cov=np.eye(2)
            )
            tstrait.sim_phenotype(
                ts=sample_ts, num_causal=5, model=trait_model, h2=[0.1, 0]
            )

    @pytest.mark.parametrize("h2", [0.1, [0.2, 0.3, 0.4]])
    def test_h2_dim(self, sample_ts, h2):
        with pytest.raises(
            ValueError, match="Length of h2 must match the number of traits"
        ):
            trait_model = tstrait.trait_model(
                distribution="multi_normal", mean=np.zeros(2), cov=np.eye(2)
            )
            tstrait.sim_phenotype(ts=sample_ts, num_causal=5, model=trait_model, h2=h2)


class TestModel:
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
        h2 = 0.3
        alpha = -0.3
        result = tstrait.sim_phenotype(
            ts=sample_ts,
            num_causal=10,
            model=trait_model,
            h2=h2,
            alpha=alpha,
            random_seed=1,
        )
        trait_df = tstrait.sim_trait(
            ts=sample_ts,
            num_causal=num_causal,
            model=trait_model,
            random_seed=1,
        )
        genetic_result = tstrait.sim_genetic(
            ts=sample_ts, trait_df=trait_df, alpha=alpha, random_seed=1
        )
        phenotype_df = tstrait.sim_env(
            genetic_df=genetic_result.genetic, h2=h2, random_seed=1
        )

        pd.testing.assert_frame_equal(result.effect_size, genetic_result.effect_size)
        pd.testing.assert_frame_equal(result.phenotype, phenotype_df)

    def test_multivariate(self, sample_ts):
        num_causal = 10
        h2 = [0.1, 0.8]
        alpha = -0.3
        trait_model = tstrait.trait_model(
            distribution="multi_normal", mean=[1, 2], cov=[[1, 0.3], [0.3, 1]]
        )
        result = tstrait.sim_phenotype(
            ts=sample_ts,
            num_causal=10,
            model=trait_model,
            h2=h2,
            alpha=alpha,
            random_seed=1,
        )
        trait_df = tstrait.sim_trait(
            ts=sample_ts,
            num_causal=num_causal,
            model=trait_model,
            random_seed=1,
        )
        genetic_result = tstrait.sim_genetic(
            ts=sample_ts, trait_df=trait_df, alpha=alpha, random_seed=1
        )
        phenotype_df = tstrait.sim_env(
            genetic_df=genetic_result.genetic, h2=h2, random_seed=1
        )

        pd.testing.assert_frame_equal(result.effect_size, genetic_result.effect_size)
        pd.testing.assert_frame_equal(result.phenotype, phenotype_df)
