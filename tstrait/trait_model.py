import numpy as np
import numbers


class TraitModel:
    """Superclass of the trait model. See the :ref:`sec_trait_model` section for
    more details on the available models and examples.

    :param model_name: Name of the trait model.
    :type model_name: str
    :param trait_mean: Mean value of the simulated effect sizes.
    :type trait_mean: float
    :param trait_sd: Standard deviation of the simulated effect sizes.
    :type trait_sd: float
    """

    def __init__(self, model_name, trait_mean, trait_sd):
        if not isinstance(trait_mean, numbers.Number):
            raise TypeError("Mean value of traits should be a number")
        if not isinstance(trait_sd, numbers.Number):
            raise TypeError("Standard deviation of traits should be a number")
        if trait_sd < 0:
            raise ValueError(
                "Standard deviation of traits should be a non-negative number"
            )
        self._model_name = model_name
        self.trait_mean = trait_mean
        self.trait_sd = trait_sd

    def sim_effect_size(self, num_causal, allele_freq, rng):
        """
        This method simulates an effect size of a causal mutation assuming that it
        follows a normal distribution with a constant standard deviation. The
        `allele_freq` input is not used to simulate an effect size.

        :param num_causal: Number of causal sites.
        :type num_causal: int
        :param allele_freq: Allele frequency of the causal mutation.
        :type allele_freq: float
        :param rng: Random generator that will be used to simulate effect size.
        :type rng: numpy.random.Generator
        :return: Simulated effect size of a causal mutation.
        :rtype: float
        """
        if not isinstance(num_causal, numbers.Number):
            raise TypeError("Number of causal sites should be a number")
        if not isinstance(allele_freq, numbers.Number):
            raise TypeError("Allele frequency should be a number")
        if int(num_causal) != num_causal or num_causal <= 0:
            raise ValueError("Number of causal sites should be a positive integer")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng should be a numpy random generator")

        if self.trait_sd == 0:
            beta = self.trait_mean
        else:
            beta = rng.normal(
                loc=self.trait_mean, scale=self.trait_sd / np.sqrt(num_causal)
            )
        return beta

    @property
    def name(self):
        """
        Name of the trait model.
        """
        return self._model_name


class TraitModelAdditive(TraitModel):
    """Additive trait model class, where the distribution of effect size does not
    depend on allele frequency. The effect size will be simulated from a normal
    distribution. See the :ref:`sec_trait_model_additive` section for more details on
    this model.

    :param trait_mean: Mean value of the simulated effect sizes.
    :type trait_mean: float
    :param trait_sd: Standard deviation of the simulated effect sizes.
    :type trait_sd: float
    """

    def __init__(self, trait_mean, trait_sd):
        super().__init__("additive", trait_mean, trait_sd)

    def sim_effect_size(self, num_causal, allele_freq, rng):
        """
        This method uses `sim_effect_size` from `TraitModel` to simulate an effect size
        of a causal mutation.

        :param num_causal: Number of causal sites.
        :type num_causal: int
        :param allele_freq: Allele frequency of causal mutation.
        :type allele_freq: float
        :param rng: Random generator that will be used to simulate effect size.
        :type rng: numpy.random.Generator
        :return: Simulated effect size of a causal mutation.
        :rtype: float
        """
        beta = super().sim_effect_size(num_causal, allele_freq, rng)
        return beta


class TraitModelAlleleFrequency(TraitModel):
    """Allele frequency trait model class, where the distribution of effect size
    depends on allele frequency.

    The `alpha` parameter modifies the relative emphasis placed on rarer variants to
    simulate the effect sizes of causal mutations. The same results as the
    :class:`TraitModelAdditive` model can be determined by setting the `alpha`
    parameter to be zero. See the :ref:`sec_trait_model_allele` section for more details
    on the model.

    :param trait_mean: Mean value of the simulated traits.
    :type trait_mean: float
    :param trait_sd: Standard deviation of the simulated traits.
    :type trait_sd: float
    :param alpha: Parameter that determines the relative weight on rarer variants.
        Negative `alpha` value can increase the magnitude of effect sizes coming
        from rarer variants. The effects of allele frequency on effect size
        simulation can be ignored by setting `alpha` to be zero.
    :type alpha: float
    """

    def __init__(self, trait_mean, trait_sd, alpha):
        super().__init__("allele", trait_mean, trait_sd)
        if not isinstance(alpha, numbers.Number):
            raise TypeError("Alpha should be a number")
        self.alpha = alpha

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
        beta = super().sim_effect_size(num_causal, allele_freq, rng)
        if allele_freq >= 1 or allele_freq <= 0:
            raise ValueError("Allele frequency should be 0 < Allele frequency < 1")
        beta *= np.sqrt(pow(2 * allele_freq * (1 - allele_freq), self.alpha))
        return beta
