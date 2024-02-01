import numpy as np
import pytest
import tstrait


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
            TypeError, match="random_sign must be a <class 'bool'> instance"
        ):
            tstrait.trait_model(distribution="exponential", scale=1, random_sign=1)

    def test_fixed(self):
        with pytest.raises(TypeError, match="value must be numeric"):
            tstrait.trait_model(distribution="fixed", value="1")

        with pytest.raises(
            TypeError, match="random_sign must be a <class 'bool'> instance"
        ):
            tstrait.trait_model(distribution="fixed", value=1, random_sign=1)

    def test_gamma(self):
        with pytest.raises(
            TypeError, match="random_sign must be a <class 'bool'> instance"
        ):
            tstrait.trait_model(distribution="gamma", shape=1, scale=1, random_sign=1)


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
        np.testing.assert_allclose(np.cov(effect_size.T), cov, rtol=2e-1)
        np.testing.assert_allclose(effect_size.mean(0), mean, rtol=1e-1)


class TestFixed:
    def test_value(self):
        num_causal = 100
        value = 2
        model = tstrait.trait_model(distribution="fixed", value=value)
        effect_size = model._sim_effect_size(num_causal, np.random.default_rng(1))
        np.testing.assert_array_equal(np.unique(effect_size), np.array([value]))

    def test_negative_value(self):
        num_causal = 100
        value = 2
        model = tstrait.trait_model(distribution="fixed", value=value, random_sign=True)
        effect_size = model._sim_effect_size(num_causal, np.random.default_rng(2))
        np.testing.assert_array_equal(np.unique(effect_size), np.array([-value, value]))


@pytest.mark.parametrize("num_causal", [1000])
class TestRandomSign:
    """
    Test that the random_sign input generates both positive and negative values.
    """

    def test_exponential_random_sign(self, num_causal):
        scale = 3
        model = tstrait.trait_model(
            distribution="exponential", scale=scale, random_sign=True
        )
        effect_size = model._sim_effect_size(num_causal, np.random.default_rng(1))
        assert np.min(effect_size) < 0
        assert np.max(effect_size) > 0

        model = tstrait.trait_model(
            distribution="exponential", scale=scale, random_sign=False
        )
        effect_size = model._sim_effect_size(num_causal, np.random.default_rng(1))
        assert np.min(effect_size) > 0

    def test_gamma_random_sign(self, num_causal):
        shape = 6
        scale = 2
        model = tstrait.trait_model(
            distribution="gamma", shape=shape, scale=scale, random_sign=True
        )
        effect_size = model._sim_effect_size(num_causal, np.random.default_rng(1))
        assert np.min(effect_size) < 0
        assert np.max(effect_size) > 0

        model = tstrait.trait_model(
            distribution="gamma", shape=shape, scale=scale, random_sign=False
        )
        effect_size = model._sim_effect_size(num_causal, np.random.default_rng(1))
        assert np.min(effect_size) > 0
