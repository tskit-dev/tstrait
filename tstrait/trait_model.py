from abc import ABCMeta
from abc import abstractmethod

import numpy as np

from .base import (
    _check_int,
    _check_val,
    _check_symmetric,
    _check_instance,
    _check_numeric_array,
    _check_same_length,
)  # noreorder


class TraitModel(metaclass=ABCMeta):
    """Superclass of the trait model. See the :ref:`sec_trait_model` section for
    more details on the available models and examples.

    :param model_name: Name of the trait model.
    :type model_name: str
    """

    @abstractmethod
    def __init__(self, name):
        self.name = name
        self.num_trait = 1

    def _check_parameter(self, num_causal, rng):
        num_causal = _check_int(num_causal, "num_causal", minimum=1)
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy random generator")
        return num_causal

    @abstractmethod
    def sim_effect_size(self):
        pass


class TraitModelNormal(TraitModel):
    """Normal distribution trait model class, where the effect sizes are simulated
    from a normal distribution.

    :param mean: Mean value of the simulated effect sizes.
    :type mean: float
    :param var: Variance of the simulated effect sizes.
    :type var: float
    """

    def __init__(self, mean, var):
        self.mean = _check_val(mean, "mean")
        self.var = _check_val(var, "var", minimum=0)
        super().__init__("normal")

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
        num_causal = self._check_parameter(num_causal, rng)
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
        self.scale = _check_val(scale, "scale", minimum=0)
        self.negative = _check_instance(negative, "negative", bool)
        super().__init__("exponential")

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
        num_causal = self._check_parameter(num_causal, rng)
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
        self.value = _check_val(value, "value")
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
        num_causal = self._check_parameter(num_causal, rng)
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
        self.mean = _check_val(mean, "mean")
        self.var = _check_val(var, "var", minimum=0)
        self.df = _check_val(df, "df", minimum=0)
        super().__init__("t")

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
        num_causal = self._check_parameter(num_causal, rng)
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
        self.shape = _check_val(shape, "shape", minimum=0)
        self.scale = _check_val(scale, "scale", minimum=0)
        self.negative = _check_instance(negative, "negative", bool)
        super().__init__("gamma")

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
        num_causal = self._check_parameter(num_causal, rng)
        beta = rng.gamma(self.shape, self.scale) / num_causal
        if self.negative:
            beta *= rng.choice([-1, 1])
        return beta


class TraitModelMultivariateNormal(TraitModel):
    """Trait model class of multiple traits.

    We assume that the phenotypes are pleiotropic, meaning that the genes are
    influencing more than one trait.

    :param mean: Mean vector of the simulated effect sizes. The length of the vector
        should match the number of traits.
    :type mean: list or numpy.ndarray
    :param cov: Covariance matrix of simulated effect sizes.
    :type cov: list or numpy.ndarray
    """

    def __init__(self, mean, cov):
        mean = _check_numeric_array(mean, "mean")
        cov = _check_symmetric(cov, "cov")
        _check_same_length(mean, cov[0, :], "mean", "cov")
        self.num_trait = len(mean)
        self.mean = mean
        self.cov = cov

    def sim_effect_size(self, num_causal, rng):
        """
        This method returns an effect size from a multivariate normal distribution.

        :param num_causal: Number of causal sites
        :type num_causal: int
        :param rng: Random generator that will be used to simulate effect size.
        :type rng: numpy.random.Generator
        :return: Simulated effect size of a causal mutation.
        :rtype: float
        """
        num_causal = self._check_parameter(num_causal, rng)
        beta = rng.multivariate_normal(mean=self.mean, cov=self.cov)
        beta /= num_causal
        return beta


MODEL_MAP = {
    "normal": TraitModelNormal,
    "exponential": TraitModelExponential,
    "fixed": TraitModelFixed,
    "t": TraitModelT,
    "gamma": TraitModelGamma,
    "multi_normal": TraitModelMultivariateNormal,
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
    distribution = _check_instance(distribution, "distribution", str)
    lower_model = distribution.lower()
    if lower_model not in MODEL_MAP:
        raise ValueError(
            "Distribution '{}' unknown. Choose from {}".format(
                distribution, sorted(MODEL_MAP.keys())
            )
        )
    model_instance = MODEL_MAP[lower_model](**kwargs)

    return model_instance
