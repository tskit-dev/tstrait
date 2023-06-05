import numpy as np
import numbers


class TraitModel:
    """Superclass of the trait model

    This class can be used to create a trait model that simulates effect sizes with
    custom distributions.

    :param model_name: Name of the trait model
    :type model_name: str
    :param trait_mean: Mean value of the simulated traits
    :type trait_mean: float
    :param trait_sd: Standard deviation of the simulated traits
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
        follows a normal distribution with a constant standard deviation. The mean
        of the normal distrbution is the `trait_mean` attribute of the class
        `tstrait.TraitModel` object, and the standard deviation is the `trait_sd`
        attribute of the class `trait.TraitModel` object divided by the square-root
        of the number of causal sites given by `num_causal`. The `allele_freq` input
        is not used in the simulation process.

        :param num_causal: Number of causal sites
        :type num_causal: int
        :param allele_freq: Allele frequency of the causal mutation
        :type allele_freq: float
        :param rng: Random generator that will be used to simulate effect size
        :type rng: class `np.random.Generator`
        :return: Simulated effect size of a causal mutation
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
        return self._model_name


class TraitModelAdditive(TraitModel):
    """Additive trait model class, where the distribution of effect size does not
    depend on allele frequency.

    :param trait_mean: Mean value of the simulated traits
    :type trait_mean: float
    :param trait_sd: Standard deviation of the simulated traits
    :type trait_sd: float
    """

    def __init__(self, trait_mean, trait_sd):
        super().__init__("additive", trait_mean, trait_sd)

    def sim_effect_size(self, num_causal, allele_freq, rng):
        """
        Simulates an effect size of a causal mutation, assuming that it follows a
        normal distribution. The mean value is the `trait_mean` attribute of the class
        `tskit.TraitModelAdditive` object, and the standard deviation is the
        `trait_sd` attribute of the `tskit.TraitModelAdditive` object divided by the
        square-root of the number of causal sites, which is given by the `num_causal`
        input. The `allele_freq` input is not used in the simulation process.

        :param num_causal: Number of causal sites
        :type num_causal: int
        :param allele_freq: Allele frequency of causal mutation
        :type allele_freq: float
        :param rng: Random generator that will be used to simulate effect size
        :type rng: class `np.random.Generator`
        :return: Simulated effect size of a causal mutation
        :rtype: float
        """
        beta = super().sim_effect_size(num_causal, allele_freq, rng)
        return beta


class TraitModelAllele(TraitModel):
    """Allele frequency trait model class, where the distribution of effect size
    depends on allele frequency.

    The alpha parameter modifies the relative emphasis placed on rarer variants to
    simulate the effect sizes of causal mutations. The same results as the additive
    trait model can be determined by setting the alpha parameter to be zero.

    :param trait_mean: Mean value of the simulated traits
    :type trait_mean: float
    :param trait_sd: Standard deviation of the simulated traits
    :type trait_sd: float
    :param alpha: Parameter that determines the relative weight on rarer variants
    :type alpha: float
    """

    def __init__(self, trait_mean, trait_sd, alpha):
        super().__init__("allele", trait_mean, trait_sd)
        if not isinstance(alpha, numbers.Number):
            raise TypeError("Alpha should be a number")
        self.alpha = alpha

    def sim_effect_size(self, num_causal, allele_freq, rng):
        """
        Simulates an effect size of a causal mutation, assuming that it follows a
        normal distribution. The mean value is the `trait_mean` attribute of the class
        `tskit.TraitModelAllele` object, and the standard deviation is the
        `trait_sd` attribute of the `tskit.TraitModelAllele` object divided by the
        square-root of the number of causal sites, which is given by the `num_causal`
        input.

        After the effect size gets simulated from the normal distribution, it will be
        multiplied by a constant that depends on `allele_freq` and `alpha` input of
        the method. Negative `alpha` value can increase the effect size of rarer
        variants, and greater emphasis on rarer variants can be given by decreasing
        the `alpha` value. The effects of allele frequency on simulating effect size
        can be ignored by setting `alpha` to be zero.

        :param num_causal: Number of causal sites
        :type num_causal: int or array_like(int)[int]
        :param allele_freq: Allele frequency of causal mutation
        :type allele_freq: float or array_like(float)[int]
        :param rng: Random generator that will be used to simulate effect size
        :type rng: class `np.random.Generator`
        :return: Simulated effect size of a causal mutation
        :rtype: float
        """
        beta = super().sim_effect_size(num_causal, allele_freq, rng)
        if allele_freq >= 1 or allele_freq <= 0:
            raise ValueError("Allele frequency should be 0 < Allele frequency < 1")
        beta *= np.sqrt(pow(2 * allele_freq * (1 - allele_freq), self.alpha))
        return beta
