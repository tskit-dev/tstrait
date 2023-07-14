import numbers

import numpy as np


class TraitModel:
    """Superclass of the trait model. See the :ref:`sec_trait_model` section for
    more details on the available models and examples.

    :param model_name: Name of the trait model.
    :type model_name: str
    """

    def __init__(self, name):
        self.name = name

    def _check_parameter(self, num_causal, rng):
        if not isinstance(num_causal, numbers.Number):
            raise TypeError("Number of causal sites must be an integer")
        if int(num_causal) != num_causal or num_causal <= 0:
            raise ValueError("Number of causal sites must be a positive integer")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy random generator")


class TraitModelNormal(TraitModel):
    """Normal distribution trait model class, where the effect sizes are simulated
    from a normal distribution.

    :param mean: Mean value of the simulated effect sizes.
    :type mean: float
    :param var: Variance of the simulated effect sizes.
    :type var: float
    """

    def __init__(self, mean, var):
        if not isinstance(mean, numbers.Number):
            raise TypeError("Mean must be a number")
        if not isinstance(var, numbers.Number):
            raise TypeError("Variance must be a number")
        if var <= 0:
            raise ValueError("Variance must be greater than 0")
        super().__init__("normal")
        self.mean = mean
        self.var = var

    def sim_effect_size(self, num_causal, rng):
        """
        This method simulates an effect size from a normal distribution.

        :param num_causal: Number of causal sites
        :type num_causal: int
        :param rng: Random generator that will be used to simulate effect size.
        :type rng: numpy.random.Generator
        :return: Simulated effect size of a causal mutation.
        :rtype: float
        """
        self._check_parameter(num_causal, rng)
        beta = rng.normal(
            loc=self.mean / num_causal, scale=np.sqrt(self.var) / num_causal
        )
        return beta


class TraitModelExponential(TraitModel):
    """Exponential distribution trait model class, where the effect sizes
    are simulated from an exponential distribution.

    :param scale: The scale parameter of the exponential distribution, and
        it must be greater than zero.
    :type scale: float
    :param negative: Determines if a negative value can be simulated from the
        trait model. If it is set to be True, 1 or -1 will be multipled to
        the simulated effect size.
    :type negative: bool
    """

    def __init__(self, scale, negative=False):
        if not isinstance(scale, numbers.Number):
            raise TypeError("Scale must be a number")
        if scale <= 0:
            raise ValueError("Scale must be greater than 0")
        if not isinstance(negative, bool):
            raise TypeError("Negative must be a boolean")
        super().__init__("exponential")
        self.scale = scale
        self.negative = negative

    def sim_effect_size(self, num_causal, rng):
        """
        This method simulates an effect size from an exponential distribution.

        :param num_causal: Number of causal sites
        :type num_causal: int
        :param rng: Random generator that will be used to simulate effect size.
        :type rng: numpy.random.Generator
        :return: Simulated effect size of a causal mutation.
        :rtype: float
        """
        self._check_parameter(num_causal, rng)
        beta = rng.exponential(scale=self.scale / num_causal)
        if self.negative:
            beta *= rng.choice([-1, 1])
        return beta


class TraitModelFixed(TraitModel):
    """Fixed trait model class, where the effect size is a fixed quantity.

    :param value: Effect size of causal mutation.
    :type value: float
    """

    def __init__(self, value):
        if not isinstance(value, numbers.Number):
            raise TypeError("Value must be a number")
        self.value = value
        super().__init__("fixed")

    def sim_effect_size(self, num_causal, rng):
        """
        This method returns an effect size from a fixed trait model.

        :param num_causal: Number of causal sites
        :type num_causal: int
        :param rng: Random generator that will be used to simulate effect size.
        :type rng: numpy.random.Generator
        :return: Simulated effect size of a causal mutation.
        :rtype: float
        """
        self._check_parameter(num_causal, rng)
        beta = self.value
        return beta


class TraitModelT(TraitModel):
    """T distribution trait model class, where the effect sizes are simulated from
    a t distribution.

    :param mean: Mean value of the simulated effect sizes.
    :type mean: float
    :param var: Variance of the simulated effect sizes.
    :type var: float
    :param df: Degrees of freedom, and it must be greater than 0.
    :type df: float
    """

    def __init__(self, mean, var, df):
        self.mean = mean
        self.var = var
        self.df = df
        super().__init__("t")
        if not isinstance(mean, numbers.Number):
            raise TypeError("Mean must be a number")
        if not isinstance(var, numbers.Number):
            raise TypeError("Variance must be a number")
        if var <= 0:
            raise ValueError("Variance must be greater than 0")
        if not isinstance(df, numbers.Number):
            raise TypeError("Degrees of freedom must be a number")
        if df <= 0:
            raise ValueError("Degrees of freedom must be larger than 0")
        self.df = df

    def sim_effect_size(self, num_causal, rng):
        """
        This method returns an effect size from a t distribution.

        :param num_causal: Number of causal sites
        :type num_causal: int
        :param rng: Random generator that will be used to simulate effect size.
        :type rng: numpy.random.Generator
        :return: Simulated effect size of a causal mutation.
        :rtype: float
        """
        self._check_parameter(num_causal, rng)
        beta = rng.standard_t(self.df)
        beta = (beta * np.sqrt(self.var) + self.mean) / num_causal
        return beta


class TraitModelGamma(TraitModel):
    """Gamma distribution trait model class, where the effect sizes are
    simulated from a gamma distribution.

    :param shape: The shape parameter of the gamma distribution, and it must be
        greater than zero.
    :type shape: float
    :param scale: The scale parameter of the gamma distribution, and it must be
        greater than zero.
    :type scale: float
    :param negative: Determines if a negative value can be simulated from the
        trait model. If it is set to be True, 1 or -1 will be multipled to
        the simulated effect size.
    :type negative: bool
    """

    def __init__(self, shape, scale, negative=False):
        if not isinstance(shape, numbers.Number):
            raise TypeError("Shape must be a number")
        if shape <= 0:
            raise ValueError("Shape must be greater than 0")
        if not isinstance(scale, numbers.Number):
            raise TypeError("Scale must be a number")
        if scale <= 0:
            raise ValueError("Scale must be greater than 0")
        if not isinstance(negative, bool):
            raise TypeError("Negative must be a boolean")
        super().__init__("gamma")
        self.shape = shape
        self.scale = scale
        self.negative = negative

    def sim_effect_size(self, num_causal, rng):
        """
        This method returns an effect size from a gamma distribution.

        :param num_causal: Number of causal sites
        :type num_causal: int
        :param rng: Random generator that will be used to simulate effect size.
        :type rng: numpy.random.Generator
        :return: Simulated effect size of a causal mutation.
        :rtype: float
        """
        self._check_parameter(num_causal, rng)
        beta = rng.gamma(self.shape, self.scale) / num_causal
        if self.negative:
            beta *= rng.choice([-1, 1])
        return beta


MODEL_MAP = {
    "normal": TraitModelNormal,
    "exponential": TraitModelExponential,
    "fixed": TraitModelFixed,
    "t": TraitModelT,
    "gamma": TraitModelGamma,
}


def trait_model(distribution, **kwargs):
    """Returns a trait model corresponding to the specified model. The arguments
    corresponding to the specified distribution must be inputted as arguments into
    this function.

    :param distribution: A string describing the trait model.
    :type distribution: str
    :return: The corresponding trait model.
    :rtype: TraitModel
    """
    if not isinstance(distribution, str):
        raise TypeError("Distribution must be a string")
    lower_model = distribution.lower()
    if lower_model not in MODEL_MAP:
        raise ValueError(
            "Distribution '{}' unknown. Choose from {}".format(
                distribution, sorted(MODEL_MAP.keys())
            )
        )
    model_instance = MODEL_MAP[lower_model](**kwargs)

    return model_instance
