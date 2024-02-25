import msprime
import numpy as np
import pandas as pd
import pytest
import tstrait


@pytest.fixture(scope="class")
def sample_ts():
    ts = msprime.sim_ancestry(1000, sequence_length=100_000, random_seed=1)
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
        with pytest.raises(
            ValueError, match="Cannot specify both num_causal and causal_sites"
        ):
            tstrait.sim_phenotype(
                ts=sample_ts,
                num_causal=2,
                model=sample_trait_model,
                causal_sites=[2, 4],
                h2=0.3,
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

    @pytest.mark.parametrize("h2", [[0.1], [0.2, 0.3, 0.4]])
    def test_h2_dim(self, sample_ts, h2):
        with pytest.raises(
            ValueError, match="Length of h2 must match the number of traits"
        ):
            trait_model = tstrait.trait_model(
                distribution="multi_normal", mean=np.zeros(2), cov=np.eye(2)
            )
            tstrait.sim_phenotype(ts=sample_ts, num_causal=5, model=trait_model, h2=h2)

    def test_causal_sites_bad_input(self, sample_ts, sample_trait_model):
        with pytest.raises(
            ValueError, match="There must not be repeated values in causal_sites"
        ):
            tstrait.sim_phenotype(
                ts=sample_ts, causal_sites=[1, 5, 1], model=sample_trait_model, h2=-0.1
            )


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
        random_seed = 1
        result = tstrait.sim_phenotype(
            ts=sample_ts,
            num_causal=num_causal,
            model=trait_model,
            h2=h2,
            alpha=alpha,
            random_seed=random_seed,
        )
        trait_df = tstrait.sim_trait(
            ts=sample_ts,
            num_causal=num_causal,
            model=trait_model,
            alpha=alpha,
            random_seed=random_seed,
        )
        genetic_df = tstrait.genetic_value(ts=sample_ts, trait_df=trait_df)
        phenotype_df = tstrait.sim_env(
            genetic_df=genetic_df, h2=h2, random_seed=random_seed
        )

        pd.testing.assert_frame_equal(result.trait, trait_df)
        pd.testing.assert_frame_equal(result.phenotype, phenotype_df)

    def test_multivariate(self, sample_ts):
        num_causal = 10
        h2 = [0.1, 0.8]
        alpha = -0.3
        trait_model = tstrait.trait_model(
            distribution="multi_normal", mean=[1, 2], cov=[[1, 0.3], [0.3, 1]]
        )
        random_seed = 10
        result = tstrait.sim_phenotype(
            ts=sample_ts,
            num_causal=num_causal,
            model=trait_model,
            h2=h2,
            alpha=alpha,
            random_seed=random_seed,
        )
        trait_df = tstrait.sim_trait(
            ts=sample_ts,
            num_causal=num_causal,
            model=trait_model,
            alpha=alpha,
            random_seed=random_seed,
        )
        genetic_df = tstrait.genetic_value(ts=sample_ts, trait_df=trait_df)
        phenotype_df = tstrait.sim_env(
            genetic_df=genetic_df, h2=h2, random_seed=random_seed
        )

        pd.testing.assert_frame_equal(result.trait, trait_df)
        pd.testing.assert_frame_equal(result.phenotype, phenotype_df)

    @pytest.mark.parametrize("causal_sites", [[0], [4, 2], [1, 2, 3]])
    def test_causal_sites(self, sample_ts, sample_trait_model, causal_sites):
        h2 = 0.3
        alpha = -0.3
        random_seed = 1
        result = tstrait.sim_phenotype(
            ts=sample_ts,
            causal_sites=causal_sites,
            model=sample_trait_model,
            h2=h2,
            alpha=alpha,
            random_seed=random_seed,
        )
        trait_df = tstrait.sim_trait(
            ts=sample_ts,
            causal_sites=causal_sites,
            model=sample_trait_model,
            alpha=alpha,
            random_seed=random_seed,
        )
        genetic_df = tstrait.genetic_value(ts=sample_ts, trait_df=trait_df)
        phenotype_df = tstrait.sim_env(
            genetic_df=genetic_df, h2=h2, random_seed=random_seed
        )

        pd.testing.assert_frame_equal(result.trait, trait_df)
        pd.testing.assert_frame_equal(result.phenotype, phenotype_df)

    @pytest.mark.parametrize("causal_sites", [[0], [4, 2]])
    def test_causal_sites_multivariate(self, sample_ts, causal_sites):
        h2 = [0.1, 0.8]
        alpha = -0.3
        trait_model = tstrait.trait_model(
            distribution="multi_normal", mean=[1, 2], cov=[[1, 0.3], [0.3, 1]]
        )
        random_seed = 10
        result = tstrait.sim_phenotype(
            ts=sample_ts,
            causal_sites=causal_sites,
            model=trait_model,
            h2=h2,
            alpha=alpha,
            random_seed=random_seed,
        )
        trait_df = tstrait.sim_trait(
            ts=sample_ts,
            causal_sites=causal_sites,
            model=trait_model,
            alpha=alpha,
            random_seed=random_seed,
        )
        genetic_df = tstrait.genetic_value(ts=sample_ts, trait_df=trait_df)
        phenotype_df = tstrait.sim_env(
            genetic_df=genetic_df, h2=h2, random_seed=random_seed
        )

        pd.testing.assert_frame_equal(result.trait, trait_df)
        pd.testing.assert_frame_equal(result.phenotype, phenotype_df)


class TestNormalise:
    def test_output(self, sample_ts):
        mean = 2
        var = 4
        model = tstrait.trait_model(distribution="normal", mean=2, var=6)
        sim_result = tstrait.sim_phenotype(
            ts=sample_ts, num_causal=100, model=model, h2=0.3, random_seed=1
        )
        phenotype_df = sim_result.phenotype
        normalised_df = tstrait.normalise_phenotypes(phenotype_df, mean=mean, var=var)
        phenotype_array = normalised_df["phenotype"].values
        np.testing.assert_almost_equal(np.mean(phenotype_array), mean, decimal=2)
        np.testing.assert_almost_equal(np.var(phenotype_array, ddof=1), var, decimal=2)
        pd.testing.assert_series_equal(
            normalised_df["trait_id"], phenotype_df["trait_id"]
        )
        pd.testing.assert_series_equal(
            normalised_df["individual_id"], phenotype_df["individual_id"]
        )

        num_ind = sample_ts.num_individuals
        assert len(normalised_df) == num_ind
        assert normalised_df.shape[1] == 3
        assert list(normalised_df.columns) == [
            "individual_id",
            "trait_id",
            "phenotype",
        ]

    def test_default(self, sample_ts):
        mean = 0
        var = 1
        model = tstrait.trait_model(distribution="normal", mean=2, var=6)
        sim_result = tstrait.sim_phenotype(
            ts=sample_ts, num_causal=100, model=model, h2=0.3, random_seed=1
        )
        phenotype_df = sim_result.phenotype
        normalised_df = tstrait.normalise_phenotypes(phenotype_df)
        phenotype_array = normalised_df["phenotype"].values
        np.testing.assert_almost_equal(np.mean(phenotype_array), mean, decimal=2)
        np.testing.assert_almost_equal(np.var(phenotype_array, ddof=1), var, decimal=2)
        pd.testing.assert_series_equal(
            normalised_df["trait_id"], phenotype_df["trait_id"]
        )
        pd.testing.assert_series_equal(
            normalised_df["individual_id"], phenotype_df["individual_id"]
        )

        num_ind = sample_ts.num_individuals
        assert len(normalised_df) == num_ind
        assert normalised_df.shape[1] == 3
        assert list(normalised_df.columns) == [
            "individual_id",
            "trait_id",
            "phenotype",
        ]

    def test_column(self, sample_ts):
        model = tstrait.trait_model(distribution="normal", mean=2, var=6)
        sim_result = tstrait.sim_phenotype(
            ts=sample_ts, num_causal=100, model=model, h2=0.3, random_seed=1
        )
        phenotype_df = sim_result.phenotype
        with pytest.raises(
            ValueError, match="columns must be included in phenotype_df dataframe"
        ):
            tstrait.normalise_phenotypes(phenotype_df[["trait_id", "individual_id"]])

        with pytest.raises(
            ValueError, match="columns must be included in phenotype_df dataframe"
        ):
            tstrait.normalise_phenotypes(phenotype_df[["trait_id", "phenotype"]])

        with pytest.raises(
            ValueError, match="columns must be included in phenotype_df dataframe"
        ):
            tstrait.normalise_phenotypes(phenotype_df[["phenotype", "individual_id"]])

    @pytest.mark.parametrize("var", [0, -1])
    def test_negative_var(self, sample_ts, var):
        model = tstrait.trait_model(distribution="normal", mean=2, var=6)
        sim_result = tstrait.sim_phenotype(
            ts=sample_ts, num_causal=100, model=model, h2=0.3, random_seed=1
        )
        phenotype_df = sim_result.phenotype

        with pytest.raises(ValueError, match="Variance must be greater than 0."):
            tstrait.normalise_phenotypes(phenotype_df, var=var)

    @pytest.mark.parametrize("ddof", [0, 1])
    def test_ddof(self, sample_ts, ddof):
        model = tstrait.trait_model(distribution="normal", mean=2, var=6)
        sim_result = tstrait.sim_phenotype(
            ts=sample_ts, num_causal=100, model=model, h2=0.3, random_seed=1
        )
        phenotype_df = sim_result.phenotype
        normalised_df = tstrait.normalise_phenotypes(
            phenotype_df, mean=0, var=1, ddof=ddof
        )
        normalised_phenotype_array = normalised_df["phenotype"].values

        phenotype_array = phenotype_df["phenotype"].values
        phenotype_array = (phenotype_array - np.mean(phenotype_array)) / np.std(
            phenotype_array, ddof=ddof
        )

        np.testing.assert_array_almost_equal(
            normalised_phenotype_array, phenotype_array
        )

    def test_pleiotropy(self, sample_ts):
        mean = 0
        var = 1
        model = tstrait.trait_model(
            distribution="multi_normal", mean=np.zeros(2), cov=np.identity(2)
        )
        sim_result = tstrait.sim_phenotype(
            ts=sample_ts, num_causal=100, model=model, h2=0.3, random_seed=1
        )
        phenotype_df = sim_result.phenotype
        normalised_df = tstrait.normalise_phenotypes(phenotype_df, mean=mean, var=var)
        grouped = normalised_df.groupby(["trait_id"])[["phenotype"]]
        mean_array = grouped.mean().values.T[0]
        var_array = grouped.var().values.T[0]
        np.testing.assert_almost_equal(mean_array, np.zeros(2), decimal=2)
        np.testing.assert_almost_equal(var_array, np.ones(2), decimal=2)
