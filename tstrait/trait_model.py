import numbers

import numpy as np

class TraitModel:
    """Superclass of the trait model. See the :ref:`sec_trait_model` section for
    more details on the available models and examples.

    :param model_name: Name of the trait model.
    :type model_name: str
    :param trait_mean: Mean value of the simulated effect sizes.
    :type trait_mean: float
    :param trait_var: Variance of the simulated effect sizes.
    :type trait_var: float
    """

    def __init__(self, model_name, trait_mean, trait_var, alpha):
        if not isinstance(trait_mean, numbers.Number):
            raise TypeError("Trait mean should be a number")
        if not isinstance(trait_var, numbers.Number):
            raise TypeError("Trait variance should be a number")
        if trait_var < 0:
            raise ValueError("Trait variance should be a non-negative number")
        if not isinstance(alpha, numbers.Number):
            raise TypeError("Alpha should be a number")
        self._model_name = model_name
        self.trait_mean = trait_mean
        self.trait_var = trait_var
        self.alpha = alpha
    
    def sim_effect_size(self, num_causal, allele_freq, rng):
        if not isinstance(num_causal, numbers.Number):
            raise TypeError("Number of causal sites should be a number")
        if not isinstance(allele_freq, numbers.Number):
            raise TypeError("Allele frequency should be a number")
        if allele_freq >= 1 or allele_freq <= 0:
            raise ValueError("Allele frequency should be 0 < Allele frequency < 1")    
        if int(num_causal) != num_causal or num_causal <= 0:        
            raise ValueError("Number of causal sites should be a positive integer")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng should be a numpy random generator")

    @property
    def name(self):
        """
        Name of the trait model.
        """
        return self._model_name

class TraitModelNormal(TraitModel):
    """Normal distribution trait model class, where the distribution of effect size
    is simulated from a normal distribution.

    The `alpha` parameter modifies the relative emphasis placed on rarer variants to
    simulate the effect sizes of causal mutations. See the
    :ref:`sec_trait_model_normal` section for more details on the model.

    :param trait_mean: Mean value of the simulated effect sizes.
    :type trait_mean: float
    :param trait_var: Variance of the simulated effect sizes.
    :type trait_var: float
    :param alpha: Parameter that determines the relative weight on rarer variants.
        A negative `alpha` value can increase the magnitude of effect sizes coming
        from rarer variants. The frequency dependent architecture can be ignored by
        setting `alpha` to be zero.
    :type alpha: float
    """

    def __init__(self, kwargs   ):
        trait_mean = kwargs["trait_mean"]
        trait_var = kwargs["trait_var"]
        alpha = kwargs["alpha"]
        super().__init__("normal", trait_mean, trait_var, alpha)

    def sim_effect_size(self, num_causal, allele_freq, rng):
        """
        This method initially simulates an effect size from
        :func:`TraitModel.sim_effect_size`. Afterwards, it will be multiplied by a
        constant that depends on `allele_freq` and `alpha` input of
        :class:`TraitModelAlleleFrequency`. Negative `alpha` value can increase the
        magnitude of effect sizes coming from rarer variants. The effects of allele
        frequency on simulating effect size can be ignored by setting `alpha` to be
        zero.

        :param num_causal: Number of causal sites
        :type num_causal: int
        :param allele_freq: Allele frequency of causal mutation
        :type allele_freq: float
        :param rng: Random generator that will be used to simulate effect size
        :type rng: numpy.random.Generator
        :return: Simulated effect size of a causal mutation
        :rtype: float
        """
        super().sim_effect_size(num_causal, allele_freq, rng)
        beta = rng.normal(
            loc=self.trait_mean / num_causal,
            scale=np.sqrt(self.trait_var / num_causal))
        beta *= np.sqrt(pow(2 * allele_freq * (1 - allele_freq), self.alpha))
        return beta

class TraitModelExponential(TraitModel):
    def __init__(self, kwargs):
        trait_scale = kwargs["trait_scale"]
        alpha = kwargs["alpha"]
        trait_mean = trait_scale
        trait_var = trait_scale ** 2
        super().__init__("exponential", trait_mean, trait_var, alpha)
        self.trait_scale = trait_scale

    def sim_effect_size(self, num_causal, allele_freq, rng):
        super().sim_effect_size(num_causal, allele_freq, rng)
        beta = rng.exponential(scale=self.trait_scale)
        beta /= num_causal
        beta *= np.sqrt(pow(2 * allele_freq * (1 - allele_freq), self.alpha))
        return beta

class TraitModelFixed(TraitModel):
    def __init__(self, kwargs):
        trait_value = kwargs["trait_value"]
        alpha = kwargs["alpha"]
        trait_mean = trait_value
        trait_var = 0
        super().__init__("exponential", trait_mean, trait_var, alpha)
        self.alpha = alpha
    
    def sim_effect_size(self, num_causal, allele_freq, rng):
        super().sim_effect_size(num_causal, allele_freq, rng)
        beta = self.trait_mean / num_causal
        beta *= np.sqrt(pow(2 * allele_freq * (1 - allele_freq), self.alpha))
        return beta

class TraitModelT(TraitModel):
    def __init__(self, kwargs):
        trait_mean = kwargs["trait_mean"]
        trait_var = kwargs["trait_var"]
        df = kwargs["df"]
        alpha = kwargs["alpha"]
        super().__init__("t", trait_mean, trait_var, alpha)
        if not isinstance(df, numbers.Number):
            raise TypeError("Degrees of freedom should be a number")
        if df <= 0:
            raise ValueError("Degrees of freedom should be larger than 0")
        self.df = df
    
    def sim_effect_size(self, num_causal, allele_freq, rng):
        super().sim_effect_size(num_causal, allele_freq, rng)
        beta = rng.standard_t(self.df)
        beta = (beta*np.sqrt(self.trait_var) + self.trait_mean)/num_causal
        beta *= np.sqrt(pow(2 * allele_freq * (1 - allele_freq), self.alpha))
        return beta

class TraitModelGamma(TraitModel):
    def __init__(self, kwargs):
        shape = kwargs["shape"]
        scale = kwargs["scale"]
        alpha = kwargs["alpha"]
        if not isinstance(shape, numbers.Number):
            raise TypeError("Shape should be a number")
        if shape <= 0:
            raise ValueError("Shape should be greater than 0")
        if not isinstance(scale, numbers.Number):
            raise TypeError("Scale should be a number")
        if scale <= 0:
            raise ValueError("Scale should be greater than 0")
        trait_mean = shape * scale
        trait_var = shape * (scale ** 2)
        super().__init__("gamma", trait_mean, trait_var, alpha)
        self.shape = shape
        self.scale = scale
    
    def sim_effect_size(self, num_causal, allele_freq, rng):
        super().sim_effect_size(num_causal, allele_freq, rng)
        beta = rng.gamma(self.shape, self.scale)
        beta /= num_causal
        beta *= np.sqrt(pow(2 * allele_freq * (1 - allele_freq), self.alpha))
        return beta

MODEL_MAP = {
    "normal": TraitModelNormal,
    "exponential": TraitModelExponential,
    "fixed": TraitModelFixed,
    "t": TraitModelT,
    "gamma": TraitModelGamma
}


def trait_model(distribution, **kwargs):
    if not isinstance(distribution, str):
        raise TypeError("Distribution must be a string")
    lower_model = distribution.lower()
    if lower_model not in MODEL_MAP:
        raise ValueError(
            "Distribution '{}' unknown. Choose from {}".format(
                distribution, sorted(MODEL_MAP.keys())
            )
        )
    model_instance = MODEL_MAP[lower_model](kwargs)
    
    return model_instance
        
