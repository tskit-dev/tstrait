import numpy as np
import pytest
import tstrait
from scipy import stats


def simulate(model, num_causal):
    return np.array(
        [
            model._sim_effect_size(num_causal, np.random.default_rng(i))
            for i in range(num_causal * 2)
        ]
    )


class TestTraitModel:
    """
    Test TraitModel superclass
    """

    tstrait.TraitModel.__abstractmethods__ = set()
    model = tstrait.TraitModel("sample")
    assert model._sim_effect_size() is None


class TestInput:
    """This test will check that an informative error is raised when the input parameter
    does not have an appropriate type or value.
    """

    def test_input_distribution(self):
        with pytest.raises(
            TypeError, match="distribution must be a <class 'str'> instance"
        ):
            tstrait.trait_model(distribution=1)
        with pytest.raises(ValueError, match="Distribution 'bad' unknown..."):
            tstrait.trait_model(distribution="bad")

    def test_exponential(self):
        with pytest.raises(
            TypeError, match="negative must be a <class 'bool'> instance"
        ):
            tstrait.trait_model(distribution="exponential", scale=1, negative=1)

    def test_fixed(self):
        with pytest.raises(TypeError, match="value must be numeric"):
            tstrait.trait_model(distribution="fixed", value="1")

    def test_gamma(self):
        with pytest.raises(
            TypeError, match="negative must be a <class 'bool'> instance"
        ):
            tstrait.trait_model(distribution="gamma", shape=1, scale=1, negative=1)


class TestMultivariate:
    def test_output_dim(self):
        model = tstrait.trait_model(
            distribution="multi_normal", mean=np.zeros(4), cov=np.eye(4)
        )
        beta = model._sim_effect_size(10, np.random.default_rng(1))
        assert beta.shape == (10, 4)
        assert model.num_trait == 4

    def test_large_sample(self):
        np.random.seed(12)
        n = 4
        num_causal = 10000
        mean = np.random.randn(n)
        M = np.random.randn(n, n)
        cov = np.dot(M, M.T)
        model = tstrait.trait_model(distribution="multi_normal", mean=mean, cov=cov)
        effect_size = model._sim_effect_size(num_causal, np.random.default_rng(1))
        np.testing.assert_allclose(
            np.cov(effect_size.T) * (num_causal**2), cov, rtol=2e-1
        )
        np.testing.assert_allclose(effect_size.mean(0) * num_causal, mean, rtol=1e-1)


@pytest.mark.parametrize("num_causal", [1000])
class TestKSTest:
    def check_distribution(self, rvs, distr, args=()):
        """
        Test the distribution of effect sizes by using a Kolmogorov-Smirnov test.
        rvs is array-like object of random variables, dist is a name of a distribution
        in scipy.stats, and args are the args for scipy.stats.dist(*args)
        """
        D, pval = stats.kstest(rvs, distr, args=args, N=1000)
        if pval < 0.05:
            raise ValueError(f"KS test failed for distribution {distr}")

    def test_normal(self, num_causal):
        loc = 2
        scale = 5
        model = tstrait.trait_model(distribution="normal", mean=loc, var=scale**2)
        effect_size = model._sim_effect_size(num_causal, np.random.default_rng(1))
        self.check_distribution(
            effect_size, "norm", (loc / num_causal, scale / num_causal)
        )

    def test_exponential(self, num_causal):
        scale = 2
        model = tstrait.trait_model(distribution="exponential", scale=scale)
        effect_size = model._sim_effect_size(num_causal, np.random.default_rng(1))
        self.check_distribution(effect_size, "expon", (0, scale / num_causal))
        assert np.min(effect_size) > 0

    def test_exponential_negative(self, num_causal):
        scale = 3
        model = tstrait.trait_model(
            distribution="exponential", scale=scale, negative=True
        )
        effect_size = model._sim_effect_size(num_causal, np.random.default_rng(1))
        assert np.min(effect_size) < 0
        effect_size = np.abs(effect_size)
        self.check_distribution(effect_size, "expon", (0, scale / num_causal))

    def test_t(self, num_causal):
        loc = 2
        scale = 3
        df = 10
        model = tstrait.trait_model(distribution="t", mean=loc, var=scale**2, df=df)
        effect_size = model._sim_effect_size(num_causal, np.random.default_rng(1))
        self.check_distribution(
            effect_size, "t", (df, loc / num_causal, scale / num_causal)
        )

    def test_gamma(self, num_causal):
        shape = 3
        scale = 2
        model = tstrait.trait_model(distribution="gamma", shape=shape, scale=scale)
        effect_size = model._sim_effect_size(num_causal, np.random.default_rng(1))
        self.check_distribution(effect_size, "gamma", (shape, 0, scale / num_causal))
        assert np.min(effect_size) > 0

    def test_gamma_negative(self, num_causal):
        shape = 6
        scale = 2
        model = tstrait.trait_model(
            distribution="gamma", shape=shape, scale=scale, negative=True
        )
        effect_size = model._sim_effect_size(num_causal, np.random.default_rng(1))
        assert np.min(effect_size) < 0
        effect_size = np.abs(effect_size)
        self.check_distribution(effect_size, "gamma", (shape, 0, scale / num_causal))

    def test_multivariate(self, num_causal):
        np.random.seed(20)
        n = 3
        mean = np.random.randn(n)
        M = np.random.randn(n, n)
        cov = np.dot(M, M.T)
        model = tstrait.trait_model(distribution="multi_normal", mean=mean, cov=cov)
        effect_size = model._sim_effect_size(num_causal, np.random.default_rng(1))
        for i in range(n):
            self.check_distribution(
                effect_size[:, i],
                "norm",
                (mean[i] / num_causal, np.sqrt(cov[i, i]) / num_causal),
            )
        const = np.random.randn(n)
        data = np.matmul(effect_size, const)
        data_val = np.matmul(const, cov)
        data_sd = np.sqrt(np.matmul(data_val, const))
        self.check_distribution(
            data, "norm", (np.matmul(const, mean) / num_causal, data_sd / num_causal)
        )
