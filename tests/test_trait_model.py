import numpy as np
import pytest
import tstrait


class Test_TraitModelNormal:
    @pytest.mark.parametrize("mean", [0, -1, np.array([1])[0]])
    @pytest.mark.parametrize("var", [0.1, np.array([1])[0]])
    @pytest.mark.parametrize("num_causal", [5, np.array([1])[0]])
    @pytest.mark.parametrize("random_seed", [1, None])
    def test_pass_condition(self, mean, var, num_causal, random_seed):
        model = tstrait.trait_model(distribution="normal", mean=mean, var=var)
        assert model.name == "normal"
        assert isinstance(model, tstrait.TraitModel)
        assert model.num_trait == 1

        rng = np.random.default_rng(random_seed)
        beta = model.sim_effect_size(num_causal=num_causal, rng=rng)

        assert isinstance(beta, float)

    @pytest.mark.parametrize("mean", ["a", [0, 1]])
    def test_mean_error(self, mean):
        with pytest.raises(TypeError, match="Mean must be a number"):
            tstrait.trait_model(distribution="normal", mean=mean, var=1)

    @pytest.mark.parametrize("var", ["a", [0, 1]])
    def test_var_error(self, var):
        with pytest.raises(TypeError, match="Variance must be a number"):
            tstrait.trait_model(distribution="normal", mean=0, var=var)

    @pytest.mark.parametrize("var", [0, -1])
    def test_var_error_negative(self, var):
        with pytest.raises(ValueError, match="Variance must be greater than 0"):
            tstrait.trait_model(distribution="normal", mean=0, var=var)

    @pytest.mark.parametrize("num_causal", ["A", None, [1, 2]])
    def test_num_causal_value_type(self, num_causal):
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        rng = np.random.default_rng(1)
        with pytest.raises(
            TypeError, match="Number of causal sites must be an integer"
        ):
            model.sim_effect_size(num_causal=num_causal, rng=rng)

    @pytest.mark.parametrize("num_causal", [1.5, -1, 0])
    def test_num_causal_value_value(self, num_causal):
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        rng = np.random.default_rng(1)
        with pytest.raises(
            ValueError, match="Number of causal sites must be a positive integer"
        ):
            model.sim_effect_size(num_causal=num_causal, rng=rng)

    @pytest.mark.parametrize("rng", [1, None, "A"])
    def test_rng_type(self, rng):
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        with pytest.raises(TypeError, match="rng must be a numpy random generator"):
            model.sim_effect_size(num_causal=1, rng=rng)


class Test_TraitModelExponential:
    @pytest.mark.parametrize("scale", [1.1, np.array([1])[0]])
    @pytest.mark.parametrize("negative", [True, False])
    @pytest.mark.parametrize("num_causal", [5, np.array([1])[0]])
    @pytest.mark.parametrize("random_seed", [1, None])
    def test_pass_condition(self, scale, negative, num_causal, random_seed):
        model = tstrait.trait_model(
            distribution="exponential", scale=scale, negative=negative
        )
        assert model.name == "exponential"
        assert isinstance(model, tstrait.TraitModel)
        assert model.num_trait == 1

        rng = np.random.default_rng(random_seed)
        beta = model.sim_effect_size(num_causal=num_causal, rng=rng)

        assert isinstance(beta, float)

    @pytest.mark.parametrize("scale", ["a", [0, 1]])
    def test_scale_error(self, scale):
        with pytest.raises(TypeError, match="Scale must be a number"):
            tstrait.trait_model(distribution="exponential", scale=scale)

    @pytest.mark.parametrize("scale", [-1, 0])
    def test_scale_value_error(self, scale):
        with pytest.raises(ValueError, match="Scale must be greater than 0"):
            tstrait.trait_model(distribution="exponential", scale=scale)

    @pytest.mark.parametrize("negative", [0, "True"])
    def test_negative_error(self, negative):
        with pytest.raises(TypeError, match="Negative must be a boolean"):
            tstrait.trait_model(distribution="exponential", scale=1, negative=negative)

    @pytest.mark.parametrize("num_causal", ["A", None, [1, 2]])
    def test_num_causal_value_type(self, num_causal):
        model = tstrait.trait_model(distribution="exponential", scale=1)
        rng = np.random.default_rng(1)
        with pytest.raises(
            TypeError, match="Number of causal sites must be an integer"
        ):
            model.sim_effect_size(num_causal=num_causal, rng=rng)

    @pytest.mark.parametrize("num_causal", [1.5, -1, 0])
    def test_num_causal_value_value(self, num_causal):
        model = tstrait.trait_model(distribution="exponential", scale=1)
        rng = np.random.default_rng(1)
        with pytest.raises(
            ValueError, match="Number of causal sites must be a positive integer"
        ):
            model.sim_effect_size(num_causal=num_causal, rng=rng)

    @pytest.mark.parametrize("rng", [1, None, "A"])
    def test_rng_type(self, rng):
        model = tstrait.trait_model(distribution="exponential", scale=1)
        with pytest.raises(TypeError, match="rng must be a numpy random generator"):
            model.sim_effect_size(num_causal=1, rng=rng)


class Test_TraitModelFixed:
    @pytest.mark.parametrize("value", [1.1, np.array([1.0])[0]])
    @pytest.mark.parametrize("num_causal", [5, np.array([1])[0]])
    @pytest.mark.parametrize("random_seed", [1, None])
    def test_pass_condition(self, value, num_causal, random_seed):
        model = tstrait.trait_model(distribution="fixed", value=value)
        assert model.name == "fixed"
        assert isinstance(model, tstrait.TraitModel)
        assert model.num_trait == 1

        rng = np.random.default_rng(random_seed)
        beta = model.sim_effect_size(num_causal=num_causal, rng=rng)

        assert isinstance(beta, float)

    @pytest.mark.parametrize("value", ["a", [0, 1]])
    def test_value_error(self, value):
        with pytest.raises(TypeError, match="Value must be a number"):
            tstrait.trait_model(distribution="fixed", value=value)

    @pytest.mark.parametrize("num_causal", ["A", None, [1, 2]])
    def test_num_causal_value_type(self, num_causal):
        model = tstrait.trait_model(distribution="fixed", value=1)
        rng = np.random.default_rng(1)
        with pytest.raises(
            TypeError, match="Number of causal sites must be an integer"
        ):
            model.sim_effect_size(num_causal=num_causal, rng=rng)

    @pytest.mark.parametrize("num_causal", [1.5, -1, 0])
    def test_num_causal_value_value(self, num_causal):
        model = tstrait.trait_model(distribution="fixed", value=1)
        rng = np.random.default_rng(1)
        with pytest.raises(
            ValueError, match="Number of causal sites must be a positive integer"
        ):
            model.sim_effect_size(num_causal=num_causal, rng=rng)

    @pytest.mark.parametrize("rng", [1, None, "A"])
    def test_rng_type(self, rng):
        model = tstrait.trait_model(distribution="fixed", value=1)
        with pytest.raises(TypeError, match="rng must be a numpy random generator"):
            model.sim_effect_size(num_causal=1, rng=rng)


class Test_TraitModelT:
    @pytest.mark.parametrize("mean", [0, -1, np.array([1])[0]])
    @pytest.mark.parametrize("var", [0.1, np.array([1])[0]])
    @pytest.mark.parametrize("df", [3, np.array([0.1])[0]])
    @pytest.mark.parametrize("num_causal", [5, np.array([1])[0]])
    @pytest.mark.parametrize("random_seed", [1, None])
    def test_pass_condition(self, mean, var, df, num_causal, random_seed):
        model = tstrait.trait_model(distribution="t", mean=mean, var=var, df=df)
        assert model.name == "t"
        assert isinstance(model, tstrait.TraitModel)
        assert model.num_trait == 1

        rng = np.random.default_rng(random_seed)
        beta = model.sim_effect_size(num_causal=num_causal, rng=rng)

        assert isinstance(beta, float)

    @pytest.mark.parametrize("mean", ["a", [0, 1]])
    def test_mean_error(self, mean):
        with pytest.raises(TypeError, match="Mean must be a number"):
            tstrait.trait_model(distribution="t", mean=mean, var=1, df=1)

    @pytest.mark.parametrize("var", ["a", [0, 1]])
    def test_var_error(self, var):
        with pytest.raises(TypeError, match="Variance must be a number"):
            tstrait.trait_model(distribution="t", mean=0, var=var, df=1)

    @pytest.mark.parametrize("var", [0, -1])
    def test_var_error_negative(self, var):
        with pytest.raises(ValueError, match="Variance must be greater than 0"):
            tstrait.trait_model(distribution="t", mean=0, var=var, df=1)

    @pytest.mark.parametrize("df", ["a", [0, 1]])
    def test_df_error(self, df):
        with pytest.raises(TypeError, match="Degrees of freedom must be a number"):
            tstrait.trait_model(distribution="t", mean=0, var=1, df=df)

    @pytest.mark.parametrize("df", [0, -1])
    def test_df_val_error_negative(self, df):
        with pytest.raises(
            ValueError, match="Degrees of freedom must be larger than 0"
        ):
            tstrait.trait_model(distribution="t", mean=0, var=1, df=df)

    @pytest.mark.parametrize("num_causal", ["A", None, [1, 2]])
    def test_num_causal_value_type(self, num_causal):
        model = tstrait.trait_model(distribution="t", mean=0, var=1, df=20)
        rng = np.random.default_rng(1)
        with pytest.raises(
            TypeError, match="Number of causal sites must be an integer"
        ):
            model.sim_effect_size(num_causal=num_causal, rng=rng)

    @pytest.mark.parametrize("num_causal", [1.5, -1, 0])
    def test_num_causal_value_value(self, num_causal):
        model = tstrait.trait_model(distribution="t", mean=0, var=1, df=20)
        rng = np.random.default_rng(1)
        with pytest.raises(
            ValueError, match="Number of causal sites must be a positive integer"
        ):
            model.sim_effect_size(num_causal=num_causal, rng=rng)

    @pytest.mark.parametrize("rng", [1, None, "A"])
    def test_rng_type(self, rng):
        model = tstrait.trait_model(distribution="t", mean=0, var=1, df=20)
        with pytest.raises(TypeError, match="rng must be a numpy random generator"):
            model.sim_effect_size(num_causal=1, rng=rng)


class Test_TraitModelGamma:
    @pytest.mark.parametrize("shape", [1.1, np.array([1])[0]])
    @pytest.mark.parametrize("scale", [1.1, np.array([1])[0]])
    @pytest.mark.parametrize("negative", [True, False])
    @pytest.mark.parametrize("num_causal", [5, np.array([1])[0]])
    @pytest.mark.parametrize("random_seed", [1, None])
    def test_pass_condition(self, shape, scale, negative, num_causal, random_seed):
        model = tstrait.trait_model(
            distribution="gamma", shape=shape, scale=scale, negative=negative
        )
        assert model.name == "gamma"
        assert model.num_trait == 1
        assert isinstance(model, tstrait.TraitModel)

        rng = np.random.default_rng(random_seed)
        beta = model.sim_effect_size(num_causal=num_causal, rng=rng)

        assert isinstance(beta, float)

    @pytest.mark.parametrize("shape", ["a", [0, 1]])
    def test_shape_error(self, shape):
        with pytest.raises(TypeError, match="Shape must be a number"):
            tstrait.trait_model(distribution="gamma", shape=shape, scale=1)

    @pytest.mark.parametrize("shape", [-1, 0])
    def test_shape_val_error(self, shape):
        with pytest.raises(ValueError, match="Shape must be greater than 0"):
            tstrait.trait_model(distribution="gamma", shape=shape, scale=1)

    @pytest.mark.parametrize("scale", ["a", [0, 1]])
    def test_scale_error(self, scale):
        with pytest.raises(TypeError, match="Scale must be a number"):
            tstrait.trait_model(distribution="gamma", shape=1, scale=scale)

    @pytest.mark.parametrize("scale", [-1, 0])
    def test_scale_val_error(self, scale):
        with pytest.raises(ValueError, match="Scale must be greater than 0"):
            tstrait.trait_model(distribution="gamma", shape=1, scale=scale)

    @pytest.mark.parametrize("negative", [0, "True"])
    def test_negative_error(self, negative):
        with pytest.raises(TypeError, match="Negative must be a boolean"):
            tstrait.trait_model(
                distribution="gamma", shape=1, scale=1, negative=negative
            )

    @pytest.mark.parametrize("num_causal", ["A", None, [1, 2]])
    def test_num_causal_value_type(self, num_causal):
        model = tstrait.trait_model(distribution="gamma", shape=1, scale=1)
        rng = np.random.default_rng(1)
        with pytest.raises(
            TypeError, match="Number of causal sites must be an integer"
        ):
            model.sim_effect_size(num_causal=num_causal, rng=rng)

    @pytest.mark.parametrize("num_causal", [1.5, -1, 0])
    def test_num_causal_value_value(self, num_causal):
        model = tstrait.trait_model(distribution="gamma", shape=1, scale=1)
        rng = np.random.default_rng(1)
        with pytest.raises(
            ValueError, match="Number of causal sites must be a positive integer"
        ):
            model.sim_effect_size(num_causal=num_causal, rng=rng)

    @pytest.mark.parametrize("rng", [1, None, "A"])
    def test_rng_type(self, rng):
        model = tstrait.trait_model(distribution="gamma", shape=1, scale=1)
        with pytest.raises(TypeError, match="rng must be a numpy random generator"):
            model.sim_effect_size(num_causal=1, rng=rng)


class Test_TraitModelMultivariateNormal:
    def test_dimension(self):
        mean = [0, 1]
        var = [2, 3]
        cor = [[1, 0], [0, 1]]
        model = tstrait.trait_model(
            distribution="multi_normal", mean=mean, var=var, cor=cor
        )
        rng = np.random.default_rng(1)
        beta = model.sim_effect_size(num_causal=1, rng=rng)
        assert len(beta) == 2
        assert model.name == "multi_normal"
        assert isinstance(model, tstrait.TraitModel)
        assert model.num_trait == 2

    def test_dimension_numpy(self):
        mean = np.array([0, 0, 0])
        var = np.array([1, 1, 1])
        cor = np.array([[1, 0.3, 0.4], [0.3, 1, 0], [0.4, 0, 1]])
        model = tstrait.trait_model(
            distribution="multi_normal", mean=mean, var=var, cor=cor
        )
        rng = np.random.default_rng(1)
        beta = model.sim_effect_size(num_causal=1, rng=rng)
        assert len(beta) == 3
        assert model.name == "multi_normal"
        assert isinstance(model, tstrait.TraitModel)
        assert model.num_trait == 3

    @pytest.mark.parametrize("mean", [0, "A", {"A": 1}])
    def test_mean_input(self, mean):
        var = np.array([2, 3])
        cor = np.array([[1, 0], [0, 1]])
        with pytest.raises(TypeError, match="Mean must be a list or a numpy array"):
            tstrait.trait_model(
                distribution="multi_normal", mean=mean, var=var, cor=cor
            )

    @pytest.mark.parametrize("var", [0, "A", {"A": 1}])
    def test_var_input(self, var):
        mean = np.array([2, 3])
        cor = np.array([[1, 0], [0, 1]])
        with pytest.raises(TypeError, match="Variance must be a list or a numpy array"):
            tstrait.trait_model(
                distribution="multi_normal", mean=mean, var=var, cor=cor
            )

    @pytest.mark.parametrize("cor", [0, "A", {"A": 1}])
    def test_cor_input(self, cor):
        mean = np.array([2, 3])
        var = np.array([1, 1])
        with pytest.raises(
            TypeError, match="Correlation matrix must be a list or a numpy array"
        ):
            tstrait.trait_model(
                distribution="multi_normal", mean=mean, var=var, cor=cor
            )

    def test_var_zero(self):
        mean = np.array([1, 1, 1])
        var = np.array([2, 0, 1])
        cor = np.array([[1, 0.3, 0.4], [0.3, 1, 0], [0.4, 0, 1]])
        with pytest.raises(ValueError, match="Variance must be greater than 0"):
            tstrait.trait_model(
                distribution="multi_normal", mean=mean, var=var, cor=cor
            )

    def test_var_dim(self):
        mean = np.array([0, 0])
        var = np.array([1, 2, 3])
        cor = np.array([[1, 0], [0, 1]])
        with pytest.raises(
            ValueError,
            match="The length of the trait variance vector does not match the length "
            "of the trait mean vector",
        ):
            tstrait.trait_model(
                distribution="multi_normal", mean=mean, var=var, cor=cor
            )

    def test_cor_dim(self):
        mean = np.array([1, 1, 1])
        var = np.array([2, 1, 1])
        cor = np.array([[1, 0.3, 0.4], [0.3, 1, 0]])
        with pytest.raises(
            ValueError,
            match="The dimension of the correlation matrix does not match the dimension "
            "of the trait mean and variance vector",
        ):
            tstrait.trait_model(
                distribution="multi_normal", mean=mean, var=var, cor=cor
            )

    def test_cov_matrix(self):
        mean = np.array([0, 0])
        var = np.array([1, 9])
        cor = np.array([[1, 0.5], [0.5, 1]])
        model = tstrait.trait_model(
            distribution="multi_normal", mean=mean, var=var, cor=cor
        )
        cov = model.cov_matrix()
        assert np.array_equal(cov, np.array([[1, 1.5], [1.5, 9]]))

    def test_cov_matrix_three(self):
        mean = np.array([0, 0, 0])
        var = np.array([1, 9, 16])
        cor = np.array([[1, 0.1, 0.2], [0.1, 1, 0.3], [0.2, 0.3, 1]])
        model = tstrait.trait_model(
            distribution="multi_normal", mean=mean, var=var, cor=cor
        )
        cov = model.cov_matrix()
        assert (
            np.sum(
                np.isclose(
                    cov, np.array([[1, 0.3, 0.8], [0.3, 9, 3.6], [0.8, 3.6, 16]])
                )
            )
            == 9
        )


class Test_distribution_input:
    @pytest.mark.parametrize("dist", [0, 1, ["a", "b"]])
    def test_string_error(self, dist):
        with pytest.raises(TypeError, match="Distribution must be a string"):
            tstrait.trait_model(distribution=dist)

    def test_string_error_1(self):
        with pytest.raises(ValueError):
            tstrait.trait_model(distribution="normall", mean=0, var=1)
