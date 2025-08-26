from abc import ABCMeta
from abc import abstractmethod

import numpy as np

from .base import (
    _check_int,
    _check_val,
    _check_instance,
)  # noreorder


class TraitModel(metaclass=ABCMeta):
    """
    Superclass of the trait model.

    :ivar name: Name of the trait model.
    :vartype name: str
    :ivar num_trait: Number of traits to be simulated.
    :vartype num_trait: int

    .. seealso::
        :func:`trait_model` Construct a trait model.

        :class:`TraitModelNormal` Return a normal distribution trait model.

        :class:`TraitModelT` Return a Student's t-distribution trait model.

        :class:`TraitModelFixed` Return a fixed value trait model.

        :class:`TraitModelExponential` Return an exponential distribution trait model.

        :class:`TraitModelGamma` Return a gamma distribution trait model.

        :class:`TraitModelMultivariateNormal` Return a multivariate normal
        distribution trait model.

    .. note::
        This is the base class for all trait models in tstrait. All trait models
        should set all parameters in their ``__init__`` as arguments.
    """

    @abstractmethod
    def __init__(self, name):
        self.name = name
        self.num_trait = 1

    def _check_parameter(self, num_causal, rng):
        num_causal = _check_int(num_causal, "num_causal", minimum=1)
        _check_instance(rng, "rng", np.random.Generator)
        return num_causal

    @abstractmethod
    def _sim_effect_size(self):
        pass


class TraitModelNormal(TraitModel):
    """
    Normal distribution trait model.

    :param mean: Mean of the simulated effect size.
    :type mean: float
    :param var: Variance of the simulated effect size. Must be non-negative.
    :type var: float
    :returns: Normal distribution trait model.
    :rtype: TraitModel

    .. seealso::
        :func:`trait_model` Construct a trait model.

        :py:meth:`numpy.random.Generator.normal` Details on the input parameters
        and distribution.

    .. note::
        This is a trait model built on top of
        :py:meth:`numpy.random.Generator.normal`, so please see its documentation
        for the details of the normal distribution simulation.

    .. rubric:: Examples

    Please see the docstring example of :func:`trait_model` for constructing a
    normal distribution trait model.
    """

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
        super().__init__("normal")

    def _sim_effect_size(self, num_causal, rng):
        """
        This method simulates an effect size from a normal distribution.

        :param num_causal: Number of causal sites
        :type num_causal: int
        :param rng: Random generator that will be used to simulate effect size.
        :type rng: numpy.random.Generator
        :returns: Simulated effect size of a causal mutation.
        :rtype: float or array-like
        """
        num_causal = self._check_parameter(num_causal, rng)
        beta = rng.normal(
            loc=self.mean,
            scale=np.sqrt(self.var),
            size=num_causal,
        )
        return beta


class TraitModelExponential(TraitModel):
    """Exponential distribution trait model.

    :param scale: Scale of the exponential distribution. Must be non-negative.
    :type scale: float
    :param random_sign: If True, :math:`1` or :math:`-1` will be randomly
        multiplied to the simulated effect sizes, such that we can simulate effect
        sizes with randomly chosen signs. If False, only positive values are being
        simulated as part of the property of the exponential distribution.
    :type random_sign: bool
    :returns: Exponential distribution trait model.
    :rtype: TraitModel

    .. seealso::
        :func:`trait_model` Construct a trait model.

        :py:meth:`numpy.random.Generator.exponential` Details on the input
        parameters and distribution.

    .. note::
        This is a trait model built on top of
        :py:meth:`numpy.random.Generator.exponential`, so please see its
        documentation for the details of the exponential distribution simulation.

    .. rubric:: Examples

    Please see the docstring example of :func:`trait_model` for constructing an
    exponential distribution trait model.
    """

    def __init__(self, scale, random_sign=False):
        self.scale = scale
        self.random_sign = _check_instance(random_sign, "random_sign", bool)
        super().__init__("exponential")

    def _sim_effect_size(self, num_causal, rng):
        """
        This method simulates an effect size from an exponential distribution.

        :param num_causal: Number of causal sites
        :type num_causal: int
        :param rng: Random generator that will be used to simulate effect size.
        :type rng: numpy.random.Generator
        :returns: Simulated effect size of a causal mutation.
        :rtype: float or array-like
        """
        num_causal = self._check_parameter(num_causal, rng)
        beta = rng.exponential(scale=self.scale, size=num_causal)
        if self.random_sign:
            beta = np.multiply(rng.choice([-1, 1], size=num_causal), beta)
        return beta


class TraitModelFixed(TraitModel):
    """
    Fixed value trait model.

    :param value: Value of the simulated effect size.
    :type value: float
    :param random_sign: If True, :math:`1` or :math:`-1` will be randomly
        multiplied to the simulated effect sizes, such that we can simulate constant
        value effect sizes with randomly chosen signs.
    :type random_sign: bool
    :returns: Fixed value trait model.
    :rtype: TraitModel

    .. seealso::
        :func:`trait_model` Construct a trait model.

    .. note::
        This is a trait model that gives the fixed value that is specified in
        `value` if `random_sign` is False. If it is true, this simulates effect
        sizes with randomly chosen signs.

    .. rubric:: Examples

    Please see the docstring example of :func:`trait_model` for constructing a
    fixed value trait model.
    """

    def __init__(self, value, random_sign=False):
        self.value = _check_val(value, "value")
        self.random_sign = _check_instance(random_sign, "random_sign", bool)
        super().__init__("fixed")

    def _sim_effect_size(self, num_causal, rng):
        """
        This method returns an effect size from a fixed trait model.

        :param num_causal: Number of causal sites
        :type num_causal: int
        :param rng: Random generator that will be used to simulate effect size.
        :type rng: numpy.random.Generator
        :returns: Simulated effect size of a causal mutation.
        :rtype: float or array-like
        """
        num_causal = self._check_parameter(num_causal, rng)
        beta = np.repeat(self.value, num_causal)
        if self.random_sign:
            beta = np.multiply(rng.choice([-1, 1], size=num_causal), beta)
        return beta


class TraitModelT(TraitModel):
    """
    Student's t distribution trait model.

    :param mean: Mean of the simulated effect size.
    :type mean: float
    :param var: Variance of the simulated effect size. Must be > 0.
    :type var: float
    :param df: Degrees of freedom. Must be > 0.
    :type df: float
    :returns: Student's t distribution trait model.
    :rtype: TraitModel

    .. seealso::
        :func:`trait_model` Construct a trait model.

        :py:meth:`numpy.random.Generator.standard_t` Details on the input
        parameters and distribution.

    .. note::
        This is a trait model built on top of
        :py:meth:`numpy.random.Generator.standard_t`, so please see its
        documentation for the details of the normal distribution simulation.

    .. rubric:: Examples

    Please see the docstring example of :func:`trait_model` for constructing a student's
    t distribution trait model.
    """

    def __init__(self, mean, var, df):
        self.mean = mean
        self.var = var
        self.df = df
        super().__init__("t")

    def _sim_effect_size(self, num_causal, rng):
        """
        This method returns an effect size from a t distribution.

        :param num_causal: Number of causal sites
        :type num_causal: int
        :param rng: Random generator that will be used to simulate effect size.
        :type rng: numpy.random.Generator
        :returns: Simulated effect size of a causal mutation.
        :rtype: float or array-like
        """
        num_causal = self._check_parameter(num_causal, rng)
        beta = rng.standard_t(self.df, size=num_causal)
        beta = beta * np.sqrt(self.var) + self.mean
        return beta


class TraitModelGamma(TraitModel):
    """
    Gamma distribution trait model.

    :param shape: Shape of the gamma distribution. Must be non-negative.
    :type shape: float
    :param scale: Scale of the gamma distribution. Must be non-negative.
    :type scale: float
    :param random_sign: If True, :math:`1` or :math:`-1` will be randomly
        multiplied to the simulated effect sizes, such that we can simulate effect
        sizes with randomly chosen signs. If False, only positive values are being
        simulated as part of the property of the gamma distribution.
    :type random_sign: bool
    :returns: Gamma distribution trait model.
    :rtype: TraitModel

    .. seealso::
        :func:`trait_model` Construct a trait model.

        :py:meth:`numpy.random.Generator.gamma` Details on the input parameters
        and distribution.

    .. note::
        This is a trait model built on top of
        :py:meth:`numpy.random.Generator.gamma`, so please see its documentation
        for the details of the gamma distribution simulation.

    .. rubric:: Examples

    Please see the docstring example of :func:`trait_model` for constructing an
    gamma distribution trait model.
    """

    def __init__(self, shape, scale, random_sign=False):
        self.shape = shape
        self.scale = scale
        self.random_sign = _check_instance(random_sign, "random_sign", bool)
        super().__init__("gamma")

    def _sim_effect_size(self, num_causal, rng):
        """
        This method returns an effect size from a gamma distribution.

        :param num_causal: Number of causal sites
        :type num_causal: int
        :param rng: Random generator that will be used to simulate effect size.
        :type rng: numpy.random.Generator
        :returns: Simulated effect size of a causal mutation.
        :rtype: float or array-like
        """
        num_causal = self._check_parameter(num_causal, rng)
        beta = rng.gamma(self.shape, self.scale, size=num_causal)
        if self.random_sign:
            beta = np.multiply(rng.choice([-1, 1], size=num_causal), beta)
        return beta


class TraitModelMultivariateNormal(TraitModel):
    """
    Multivariate normal distribution trait model.

    :param mean: Mean vector.
    :type mean: 1-D array_like, of length N
    :param cov: Covariance matrix. Must be symmetric and positive-semidefinite.
    :type cov: 2-D array_like, of shape (N, N)
    :returns: Multivariate normal distribution trait model.
    :rtype: TraitModel

    .. seealso::
        :func:`trait_model` Construct a trait model.

        :py:meth:`numpy.random.Generator.multivariate_normal` Details on the
        input parameters and distribution.

    .. note::
        Multivariate normal distribution simulation is used in multi-trait simulation,
        which is described in :ref:`multi_trait`.

        This is a trait model built on top of
        :py:meth:`numpy.random.Generator.multivariate_normal`, so please see its
        documentation for the details of the multivariate normal distribution
        simulation.

        The number of dimensions of mean vector and covariance matrix should match,
        and the length of the mean vector specifies the number of traits that will
        be simulated by using this model.

    .. rubric:: Examples

    Please see the docstring example of :func:`trait_model` for constructing a
    multivariate normal distribution trait model.
    """

    def __init__(self, mean, cov):
        super().__init__("multi_normal")
        self.num_trait = len(mean)
        self.mean = mean
        self.cov = cov

    def _sim_effect_size(self, num_causal, rng):
        """
        This method returns an effect size from a multivariate normal distribution.

        :param num_causal: Number of causal sites
        :type num_causal: int
        :param rng: Random generator that will be used to simulate effect size.
        :type rng: numpy.random.Generator
        :returns: Simulated effect size of a causal mutation.
        :rtype: float or array-like
        """
        num_causal = self._check_parameter(num_causal, rng)
        beta = rng.multivariate_normal(mean=self.mean, cov=self.cov, size=num_causal)
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
    """
    Return a trait model corresponding to the specified model.

    :param distribution: String describing the trait model. The list of
        supported distributions are:

        * "normal": Normal distribution
        * "t": Student's t distribution
        * "fixed": Fixed value
        * "exponential": Exponential distribution
        * "gamma": Gamma distribution
        * "multi_normal": Multivariate normal distribution
    :type distribution: str
    :param kwargs: These parameters will be used to specify the trait model.
    :returns: Trait model that specifies the distribution of effect size
        simulation.
    :rtype: TraitModel

    .. seealso::
        :class:`TraitModelNormal` Return a normal distribution trait model.

        :class:`TraitModelT` Return a Student's t-distribution trait model.

        :class:`TraitModelFixed` Return a fixed value trait model.

        :class:`TraitModelExponential` Return an exponential distribution trait model.

        :class:`TraitModelGamma` Return a gamma distribution trait model.

        :class:`TraitModelMultivariateNormal` Return a multivariate normal
        distribution trait model.

    .. note::
        Please reference :ref:`effect_size` for details on the effect size
        simulation. Multivariate normal distribution trait model is used in
        multi-trait simulation, which is described in :ref:`multi_trait`.

    .. rubric:: Examples

    >>> import tstrait

    Constructing a normal distribution trait model with mean :math:`0` and
    variance :math:`1`.

    >>> import tstrait
    >>> model = tstrait.trait_model(distribution="normal", mean=0, var=1)
    >>> model.name
    'normal'

    Constructing a student's t-distribution trait model with mean :math:`0`,
    variance :math:`1` and degrees of freedom :math:`1`.

    >>> model = tstrait.trait_model(distribution="t", mean=0, var=1, df=1)
    >>> model.name
    't'

    Constructing a fixed value trait model with value :math:`1`.

    >>> model = tstrait.trait_model(distribution="fixed", value=1)
    >>> model.name
    'fixed'

    Constructing an exponential distribution trait model with scale
    :math:`1`.

    >>> model = tstrait.trait_model(distribution="exponential", scale=1)
    >>> model.name
    'exponential'

    Constructing an exponential distribution trait model with scale
    :math:`1`, and enable simulation of negative values.

    >>> model = tstrait.trait_model(distribution="exponential", scale=1, \
                                    random_sign=True)

    Constructing a gamma distribution trait model with shape :math:`1`
    and scale :math:`2`.

    >>> model = tstrait.trait_model(distribution="gamma", shape=1, scale=2)
    >>> model.name
    'gamma'

    Constructing a gamma distribution trait model with shape :math:`1`,
    scale :math:`2`, and allow simulation of negative values.

    >>> model = tstrait.trait_model(distribution="gamma", shape=1, scale=2, \
                                    random_sign=True)
    >>> model.name
    'gamma'

    Constructing a multivariate normal distribution trait model with
    mean vector :math:`[0, 0]` and covariance matrix being an
    identity matrix.

    >>> import numpy as np
    >>> model = tstrait.trait_model(distribution="multi_normal", \
                                    mean=np.zeros(2), cov=np.eye(2))
    >>> model.name
    'multi_normal'
    >>> model.num_trait
    2
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
