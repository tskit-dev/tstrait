import msprime
import numpy as np
import pandas as pd
import pytest
import tstrait
from scipy import stats
from tstrait.base import _check_numeric_array


@pytest.fixture(scope="class")
def sample_ts():
    ts = msprime.sim_ancestry(10, sequence_length=100_000, random_seed=1)
    ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
    return ts


@pytest.fixture(scope="class")
def sample_df():
    df = pd.DataFrame(
        {
            "individual_id": [0, 1],
            "genetic_value": [1, 2],
            "trait_id": [0, 0],
        }
    )

    return df


@pytest.fixture(scope="class")
def sample_two_trait_df():
    df = pd.DataFrame(
        {
            "individual_id": [0, 1, 0, 1],
            "genetic_value": [1, 2, 2, 4],
            "trait_id": [0, 0, 1, 1],
        }
    )

    return df


class TestInput:
    """This test will check that an informative error is raised when the input parameter
    does not have an appropriate type or value.
    """

    def test_input_type(self):
        with pytest.raises(
            TypeError,
            match="genetic_df must be a <class 'pandas.core.frame.DataFrame'>"
            " instance",
        ):
            tstrait.sim_env(genetic_df=1, h2=0.3)

    def test_bad_input_df(self, sample_df):
        df = sample_df.copy()
        df = df.drop(columns=["trait_id"])

        with pytest.raises(
            ValueError, match="columns must be included in genetic_df dataframe"
        ):
            tstrait.sim_env(genetic_df=df, h2=[0.3, 0.3])

        df["trait_id"] = [2, 3]
        with pytest.raises(
            ValueError, match="trait_id must be consecutive and start from 0"
        ):
            tstrait.sim_env(genetic_df=df, h2=[0.3, 0.3])

        df["trait_id"] = [0, 2]
        with pytest.raises(
            ValueError, match="trait_id must be consecutive and start from 0"
        ):
            tstrait.sim_env(genetic_df=df, h2=[0.3, 0.3])

    def test_bad_input_h2(self, sample_df, sample_two_trait_df):
        with pytest.raises(
            ValueError, match="Length of h2 must match the number of traits"
        ):
            tstrait.sim_env(genetic_df=sample_df, h2=[0.3, 0.1])

        with pytest.raises(
            ValueError, match="Length of h2 must match the number of traits"
        ):
            tstrait.sim_env(genetic_df=sample_two_trait_df, h2=0.3)

        with pytest.raises(
            ValueError, match="Narrow-sense heritability must be 0 < h2 <= 1"
        ):
            tstrait.sim_env(genetic_df=sample_df, h2=0)

        with pytest.raises(
            ValueError, match="Narrow-sense heritability must be 0 < h2 <= 1"
        ):
            tstrait.sim_env(genetic_df=sample_two_trait_df, h2=[-1, 0.3])


class TestOutputDim:
    """Check that the genetic_value function gives the output with correct dimensions"""

    def check_dimensions(self, df, nrow):
        assert len(df) == nrow
        assert df.shape[1] == 5
        assert list(df.columns) == [
            "trait_id",
            "individual_id",
            "genetic_value",
            "environmental_noise",
            "phenotype",
        ]
        _check_numeric_array(df["individual_id"], "individual_id")
        _check_numeric_array(df["genetic_value"], "genetic_value")
        _check_numeric_array(df["environmental_noise"], "environmental_noise")
        _check_numeric_array(df["phenotype"], "phenotype")
        _check_numeric_array(df["trait_id"], "trait_id")

        pd.testing.assert_series_equal(
            df["phenotype"],
            df["genetic_value"] + df["environmental_noise"],
            check_names=False,
        )

    def test_output_simple(self, sample_df, sample_two_trait_df):
        single_df = tstrait.sim_env(genetic_df=sample_df, h2=0.3)
        self.check_dimensions(single_df, len(sample_df))

        two_trait_df = tstrait.sim_env(genetic_df=sample_two_trait_df, h2=[0.3, 0.1])
        self.check_dimensions(two_trait_df, len(sample_two_trait_df))

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
    def test_output_sim_env(self, sample_ts, trait_model):
        num_causal = 10
        trait_df = tstrait.sim_trait(
            ts=sample_ts, num_causal=num_causal, model=trait_model
        )
        genetic_result = tstrait.sim_genetic(
            ts=sample_ts, trait_df=trait_df, random_seed=10
        )
        phenotype_df = tstrait.sim_env(genetic_df=genetic_result.genetic, h2=0.3)

        self.check_dimensions(phenotype_df, sample_ts.num_individuals)

        np.testing.assert_equal(
            phenotype_df["trait_id"], np.zeros(sample_ts.num_individuals)
        )
        np.testing.assert_equal(
            phenotype_df["individual_id"], np.arange(sample_ts.num_individuals)
        )

    def test_output_multivariate(self, sample_ts):
        num_trait = 3
        mean = np.ones(num_trait)
        cov = np.eye(num_trait)
        num_causal = 10
        model = tstrait.trait_model(distribution="multi_normal", mean=mean, cov=cov)
        trait_df = tstrait.sim_trait(ts=sample_ts, num_causal=num_causal, model=model)
        genetic_result = tstrait.sim_genetic(
            ts=sample_ts, trait_df=trait_df, random_seed=10
        )
        phenotype_df = tstrait.sim_env(
            genetic_df=genetic_result.genetic, h2=np.ones(3) * 0.3
        )

        self.check_dimensions(phenotype_df, sample_ts.num_individuals * num_trait)

        ind_id = np.arange(sample_ts.num_individuals)
        trait_id = np.ones(sample_ts.num_individuals)
        np.testing.assert_equal(
            phenotype_df["individual_id"], np.concatenate((ind_id, ind_id, ind_id))
        )
        np.testing.assert_equal(
            phenotype_df["trait_id"],
            np.concatenate((trait_id * 0, trait_id * 1, trait_id * 2)),
        )

    def test_additional_row(self, sample_df):
        """Check that adding unexpected rows to genetic_df won't cause any errors"""
        df = sample_df
        df["height"] = [170, 180]
        tstrait.sim_env(genetic_df=df, h2=0.3)

    def test_h2_one(self, sample_df, sample_two_trait_df):
        single_df = tstrait.sim_env(genetic_df=sample_df, h2=1)
        self.check_dimensions(single_df, len(sample_df))
        np.testing.assert_equal(
            single_df["environmental_noise"], np.zeros(len(single_df))
        )

        two_trait_df = tstrait.sim_env(genetic_df=sample_two_trait_df, h2=[1, 1])
        self.check_dimensions(two_trait_df, len(sample_two_trait_df))
        np.testing.assert_equal(
            two_trait_df["environmental_noise"], np.zeros(len(two_trait_df))
        )


class Test_KSTest:
    """Test the distribution of environmental noise by using a Kolmogorov-Smirnov test"""

    def check_env_dist(self, df, h2):
        """The genetic_value dataframe must come from a single trait_id, and h2 must be
        a number.
        """
        env_std = np.sqrt((1 - h2) / h2 * np.var(df["genetic_value"]))
        D, pval = stats.kstest(
            df["environmental_noise"], "norm", args=(0, env_std), N=1000
        )

        if pval < 0.05:
            raise ValueError("KS test failed")

    def check_distribution(self, rvs, dist, args=()):
        D, pval = stats.kstest(rvs, dist, args=args, N=1000)
        if pval < 0.05:
            raise ValueError(f"KS test failed for distribution {dist}, p-val {pval}")

    def test_env_dist(self):
        num_ind = 10_000
        num_causal = 1_000
        h2 = 0.3
        ts = msprime.sim_ancestry(num_ind, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)

        model = tstrait.trait_model(distribution="normal", mean=0, var=1)

        trait_df = tstrait.sim_trait(
            ts=ts, num_causal=num_causal, model=model, random_seed=10
        )
        genetic_result = tstrait.sim_genetic(ts=ts, trait_df=trait_df, random_seed=10)
        phenotype_df = tstrait.sim_env(
            genetic_df=genetic_result.genetic, h2=h2, random_seed=20
        )
        phenotype_df1 = tstrait.sim_env(
            genetic_df=genetic_result.genetic, h2=h2, random_seed=30
        )
        phenotype_df = pd.concat([phenotype_df, phenotype_df1])

        phenotype_df = phenotype_df.reset_index()
        del phenotype_df["index"]

        self.check_env_dist(phenotype_df, h2)

    def test_env_multiple(self):
        num_ind = 10_000
        num_causal = 1_000

        np.random.seed(10)
        n = 3
        mean = np.random.randn(n)
        M = np.random.randn(n, n)
        cov = np.dot(M, M.T)

        h2 = [0.1, 0.5, 0.9]
        ts = msprime.sim_ancestry(num_ind, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)

        model_multiple = tstrait.trait_model(
            distribution="multi_normal", mean=mean, cov=cov
        )

        trait_df = tstrait.sim_trait(
            ts=ts, num_causal=num_causal, model=model_multiple, random_seed=11
        )
        genetic_result = tstrait.sim_genetic(ts=ts, trait_df=trait_df, random_seed=1)
        phenotype_df = tstrait.sim_env(
            genetic_df=genetic_result.genetic, h2=h2, random_seed=10
        )
        phenotype_df1 = tstrait.sim_env(
            genetic_df=genetic_result.genetic, h2=h2, random_seed=20
        )

        phenotype_df = pd.concat([phenotype_df, phenotype_df1])

        phenotype_df = phenotype_df.reset_index()
        del phenotype_df["index"]

        const = np.random.randn(n)
        sd_array = np.zeros(3)

        sum_data = np.zeros(2 * num_ind)

        for i in range(n):
            df = phenotype_df.loc[phenotype_df.trait_id == i]
            var = np.var(df["genetic_value"])
            sd = np.sqrt((1 - h2[i]) / h2[i] * var)
            self.check_distribution(
                df["environmental_noise"],
                "norm",
                (0, sd),
            )
            sum_data += df["environmental_noise"] * const[i]
            sd_array[i] = sd

        sd = np.dot(sd_array, const)

        self.check_distribution(
            sum_data,
            "norm",
            (0, sd),
        )
