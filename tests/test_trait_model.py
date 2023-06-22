import numpy as np
import pytest
import tstrait.trait_model as trait_model


class Test_TraitModelAdditive:
    @pytest.mark.parametrize("trait_mean", [0, 1.1, -1, np.array([1])[0]])
    @pytest.mark.parametrize("trait_sd", [0.1, np.array([1])[0]])
    @pytest.mark.parametrize("num_causal", [5, np.array([1])[0]])
    @pytest.mark.parametrize("random_seed", [1, None])
    def test_pass_condition(self, trait_mean, trait_sd, num_causal, random_seed):
        model = trait_model.TraitModelAdditive(trait_mean, trait_sd)
        assert model.name == "additive"

        rng = np.random.default_rng(random_seed)
        beta = model.sim_effect_size(num_causal, 0.5, rng)

        assert isinstance(beta, float)

    @pytest.mark.parametrize("trait_mean", ["a", "1", None, [1, 1]])
    def test_mean_type(self, trait_mean):
        with pytest.raises(TypeError, match="Mean value of traits should be a number"):
            trait_model.TraitModelAdditive(trait_mean, 1)

    @pytest.mark.parametrize("trait_sd", [-0.1, -1])
    def test_negative_sd(self, trait_sd):
        with pytest.raises(
            ValueError,
            match="Standard deviation of traits should be a non-negative number",
        ):
            trait_model.TraitModelAdditive(0, trait_sd)

    @pytest.mark.parametrize("trait_sd", ["a", "1", None])
    def test_sd_type(self, trait_sd):
        with pytest.raises(
            TypeError, match="Standard deviation of traits should be a number"
        ):
            trait_model.TraitModelAdditive(0, trait_sd)

    @pytest.mark.parametrize("trait_mean", [0, 1, 1.1, -1.1])
    @pytest.mark.parametrize("num_causal", [1, 5])
    @pytest.mark.parametrize("random_seed", [1, 2])
    def test_zero_sd(self, trait_mean, num_causal, random_seed):
        model = trait_model.TraitModelAdditive(trait_mean, 0)
        assert model.name == "additive"

        rng = np.random.default_rng(random_seed)
        beta = model.sim_effect_size(num_causal, 0.5, rng)

        assert beta == trait_mean

    @pytest.mark.parametrize("num_causal", [0.1, -1.0])
    def test_num_causal_value(self, num_causal):
        model = trait_model.TraitModelAdditive(0, 1)
        rng = np.random.default_rng(1)
        with pytest.raises(
            ValueError, match="Number of causal sites should be a positive integer"
        ):
            model.sim_effect_size(num_causal, 0.3, rng)

    @pytest.mark.parametrize("num_causal", ["a", None])
    def test_num_causal_type(self, num_causal):
        model = trait_model.TraitModelAdditive(0, 1)
        rng = np.random.default_rng(1)
        with pytest.raises(
            TypeError, match="Number of causal sites should be a number"
        ):
            model.sim_effect_size(num_causal, 0.3, rng)

    @pytest.mark.parametrize("rng", [1, 1.1, "a", None])
    def test_rng(self, rng):
        model = trait_model.TraitModelAdditive(0, 1)

        with pytest.raises(TypeError, match="rng should be a numpy random generator"):
            model.sim_effect_size(5, 0.5, rng)


class Test_TraitModelAlleleFrequency:
    @pytest.mark.parametrize("trait_mean", [1, 1.1, -1, np.array([0])[0]])
    @pytest.mark.parametrize("trait_sd", [0.1, 1, np.array([1])[0]])
    @pytest.mark.parametrize("num_causal", [1, 5])
    @pytest.mark.parametrize("random_seed", [1, None])
    @pytest.mark.parametrize("allele_freq", [0.1, np.array([0.5])[0]])
    @pytest.mark.parametrize("alpha", [1, -1, -1.1, np.array([0])[0]])
    def test_pass_condition(
        self, trait_mean, trait_sd, num_causal, allele_freq, alpha, random_seed
    ):
        model = trait_model.TraitModelAlleleFrequency(trait_mean, trait_sd, alpha)
        assert model.name == "allele"

        rng = np.random.default_rng(random_seed)
        beta = model.sim_effect_size(num_causal, allele_freq, rng)

        assert isinstance(beta, float)

    @pytest.mark.parametrize("trait_sd", [-0.1, -1])
    def test_negative_sd(self, trait_sd):
        with pytest.raises(
            ValueError,
            match="Standard deviation of traits should be a non-negative number",
        ):
            trait_model.TraitModelAlleleFrequency(0, trait_sd, -1)

    @pytest.mark.parametrize("trait_mean", [0, 1, 1.1, -1, -1.1])
    @pytest.mark.parametrize("num_causal", [1, 5])
    @pytest.mark.parametrize("random_seed", [1, 2])
    @pytest.mark.parametrize("allele_freq", [0.1, 0.99])
    @pytest.mark.parametrize("alpha", [0, 1, 1.1, -1, -1.1])
    def test_zero_sd(self, trait_mean, num_causal, allele_freq, alpha, random_seed):
        model = trait_model.TraitModelAlleleFrequency(trait_mean, 0, alpha)
        assert model.name == "allele"

        rng = np.random.default_rng(random_seed)
        beta = model.sim_effect_size(num_causal, allele_freq, rng)
        assert np.isclose(
            beta, trait_mean * np.sqrt(pow(2 * allele_freq * (1 - allele_freq), alpha))
        )

    @pytest.mark.parametrize("num_causal", [0.1, -1.0])
    def test_num_causal_value(self, num_causal):
        model = trait_model.TraitModelAlleleFrequency(0, 1, -1)
        rng = np.random.default_rng(1)
        with pytest.raises(
            ValueError, match="Number of causal sites should be a positive integer"
        ):
            model.sim_effect_size(num_causal, 0.3, rng)

    @pytest.mark.parametrize("num_causal", ["a", "1", None])
    def test_num_causal_type(self, num_causal):
        model = trait_model.TraitModelAlleleFrequency(0, 1, -1)
        rng = np.random.default_rng(1)
        with pytest.raises(
            TypeError, match="Number of causal sites should be a number"
        ):
            model.sim_effect_size(num_causal, 0.3, rng)

    @pytest.mark.parametrize("num_causal", [0, -1])
    def test_num_causal_negative(self, num_causal):
        model = trait_model.TraitModelAlleleFrequency(0, 1, -1)
        rng = np.random.default_rng(3)
        with pytest.raises(
            ValueError, match="Number of causal sites should be a positive integer"
        ):
            model.sim_effect_size(num_causal, 0.3, rng)

    @pytest.mark.parametrize("rng", [1, 1.1, "a"])
    def test_rng(self, rng):
        model = trait_model.TraitModelAlleleFrequency(0, 1, -1)

        with pytest.raises(TypeError, match="rng should be a numpy random generator"):
            model.sim_effect_size(5, 0.5, rng)

    @pytest.mark.parametrize("allele_freq", [0.0, 1.0, -0.1, 1.01])
    def test_allele_freq_value_value(self, allele_freq):
        model = trait_model.TraitModelAlleleFrequency(0, 1, -1)
        rng = np.random.default_rng(1)

        with pytest.raises(
            ValueError, match="Allele frequency should be 0 < Allele frequency < 1"
        ):
            model.sim_effect_size(5, allele_freq, rng)

    @pytest.mark.parametrize("allele_freq", ["a", [1, 1]])
    def test_allele_freq_value_type(self, allele_freq):
        model = trait_model.TraitModelAlleleFrequency(0, 1, -1)
        rng = np.random.default_rng(1)

        with pytest.raises(TypeError, match="Allele frequency should be a number"):
            model.sim_effect_size(5, allele_freq, rng)

    @pytest.mark.parametrize("alpha", [[1, 1], "a", {"alpha": 1}, None])
    def test_alpha_type(self, alpha):
        with pytest.raises(TypeError, match="Alpha should be a number"):
            trait_model.TraitModelAlleleFrequency(0, 1, alpha)

    @pytest.mark.parametrize("trait_mean", [0, 1, 1.1, -1, -1.1])
    @pytest.mark.parametrize("trait_sd", [0.1, 1])
    @pytest.mark.parametrize("num_causal", [1, 5])
    @pytest.mark.parametrize("random_seed", [1, 2])
    @pytest.mark.parametrize("allele_freq", [0.1, 0.99])
    def test_alpha_zero(
        self, trait_mean, trait_sd, num_causal, allele_freq, random_seed
    ):
        model1 = trait_model.TraitModelAlleleFrequency(trait_mean, trait_sd, 0)
        model2 = trait_model.TraitModelAdditive(trait_mean, trait_sd)
        assert model1.name == "allele"
        assert model2.name == "additive"

        rng = np.random.default_rng(random_seed)
        beta1 = model1.sim_effect_size(num_causal, allele_freq, rng)
        rng = np.random.default_rng(random_seed)
        beta2 = model2.sim_effect_size(num_causal, allele_freq, rng)

        assert np.isclose(beta1, beta2)
