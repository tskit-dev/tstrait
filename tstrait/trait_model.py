import numpy as np

class TraitModel:
# Trait model class
    def __init__(self, model_name, trait_mean, trait_sd):
        try:
            trait_sd > 0
        except:
            raise TypeError("Standard deviation of traits should be a non-negative number")
        if trait_sd < 0:
            raise ValueError("Standard deviation of traits should be a non-negative number")
        try:
            trait_mean > 0
        except:
            raise TypeError("Mean value of traits should be a number")
        self._model_name = model_name
        self.trait_mean = trait_mean
        self.trait_sd = trait_sd

    def sim_effect_size(self, num_causal, allele_freq, rng):
        """
        Simulates an effect size from a normal distribution, assuming that it won't be affected by allele frequency
        """
        try:
            num_causal > 0
        except:
            raise TypeError("Number of causal sites should be a positive integer")
        try:
            int(allele_freq)
        except:
            raise TypeError("Allele frequency should be a number")  
        if int(num_causal) != num_causal or num_causal <= 0:
            raise ValueError("Number of causal sites should be a positive integer")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng should be a numpy random generator")
        
        if self.trait_sd == 0:
            beta = self.trait_mean
        else:
            beta = rng.normal(loc=self.trait_mean, scale=self.trait_sd / np.sqrt(num_causal))
        return beta         
 
    @property
    def name(self):
        return self._model_name

class TraitModelAdditive(TraitModel):
# GCTA model (Effect size simulation won't be affected by allele frequency)
    def __init__(self, trait_mean, trait_sd):
        super().__init__('additive', trait_mean, trait_sd)
    def sim_effect_size(self, num_causal, allele_freq, rng):
        """
        Simulates an effect size from the GCTA model
        """
        beta = super().sim_effect_size(num_causal, allele_freq, rng)
        return beta

class TraitModelAllele(TraitModel):
# Allele model (Effect size will be affected by allele frequency)
    def __init__(self, trait_mean, trait_sd):
        super().__init__('allele', trait_mean, trait_sd) 
        
    def sim_effect_size(self, num_causal, allele_freq, rng):
        """
        Simulates an effect size from the allele frequency model
        """ 
        beta = super().sim_effect_size(num_causal, allele_freq, rng)
        if allele_freq >= 1 or allele_freq <= 0:
            raise ValueError("Allele frequency should be 0 < Allele frequency < 1")
        beta /= np.sqrt(2 * allele_freq * (1 - allele_freq))
        return beta
    
class TraitModelLDAK(TraitModel):
# LDAK model (Effect size will be affected by allele frequency and alpha parameter)
    def __init__(self, trait_mean, trait_sd, alpha):  
        super().__init__('ldak', trait_mean, trait_sd)
        try:
            alpha > 0
        except:
            raise TypeError("Alpha should be a number")
        
        self.alpha = alpha
    
    def sim_effect_size(self, num_causal, allele_freq, rng):
        """
        Simulates an effecet size from the LDAK model
        """
        beta = super().sim_effect_size(num_causal, allele_freq, rng)
        if allele_freq >= 1 or allele_freq <= 0:
            raise ValueError("Allele frequency should be 0 < Allele frequency < 1")
        beta *= pow(allele_freq * (1 - allele_freq), self.alpha)
        return beta