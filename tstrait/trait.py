import numpy as np

class TraitModel:
# Trait model class
    def __init__(self, model_name, trait_mean, trait_sd):
        if (not isinstance(trait_mean, int) and not isinstance(trait_mean, float)):
            raise TypeError("Mean value of traits should be a float or an integer")
        if (not isinstance(trait_sd, int) and not isinstance(trait_sd, float)):
            raise TypeError("Standard deviation of traits should be a float or an integer")  
        if trait_sd < 0:
            raise ValueError("Standard deviation of traits should be a non-negative number") 
        self._model_name = model_name
        self.trait_mean = trait_mean
        self.trait_sd = trait_sd

    def sim_effect_size(self, num_causal, rng):
        """
        Simulates an effect size from a normal distribution, assuming that it won't be affected by allele frequency
        """
        if not isinstance(num_causal, int):
            raise TypeError("Number of causal sites should be an integer")
        if num_causal <= 0:
            raise ValueError("Number of causal sites should be a positive integer")
        if type(rng) != np.random._generator.Generator:
            raise TypeError("rng should be a numpy random generator")
        
        if self.trait_sd == 0:
            beta = self.trait_mean
        else:
            beta = rng.normal(loc=self.trait_mean, scale=self.trait_sd / np.sqrt(num_causal))
        return beta         
 
    @property
    def name(self):
        return self._model_name
    
    @property
    def _require_allele_freq(self):
        return False

class TraitModelGCTA(TraitModel):
# GCTA model (Effect size simulation won't be affected by allele frequency)
    def __init__(self, trait_mean, trait_sd):
        super().__init__('gcta', trait_mean, trait_sd)
    def sim_effect_size(self, num_causal, rng):
        """
        Simulates an effect size from the GCTA model
        """
        beta = super().sim_effect_size(num_causal, rng)
        return beta

class TraitModelAllele(TraitModel):
# Allele model (Effect size will be affected by allele frequency)
    def __init__(self, trait_mean, trait_sd):
        super().__init__('allele', trait_mean, trait_sd) 
        
    def sim_effect_size(self, num_causal, allele_freq, rng):
        """
        Simulates an effect size from the allele frequency model
        """
        if not isinstance(allele_freq, float):
            raise TypeError("Allele frequency should be a float")
        if allele_freq >= 1 or allele_freq <= 0:
            raise ValueError("Allele frequency should be 0 < Allele frequency < 1")
        
        beta = super().sim_effect_size(num_causal, rng)
        beta /= np.sqrt(2 * allele_freq * (1 - allele_freq))
        return beta  
        
    @property
    def _require_allele_freq(self):
        return True
    
class TraitModelLDAK(TraitModel):
# LDAK model (Effect size will be affected by allele frequency and alpha parameter)
    def __init__(self, trait_mean, trait_sd, alpha):
        if (not isinstance(alpha, int) and not isinstance(alpha, float)):
            raise TypeError("Alpha should be a float or an integer")    
        super().__init__('ldak', trait_mean, trait_sd)
        self.alpha = alpha
    
    def sim_effect_size(self, num_causal, allele_freq, rng):
        """
        Simulates an effecet size from the LDAK model
        """
        if not isinstance(allele_freq, float):
            raise TypeError("Allele frequency should be a float")
        if allele_freq >= 1 or allele_freq <= 0:
            raise ValueError("Allele frequency should be 0 < Allele frequency < 1")
        beta = super().sim_effect_size(num_causal, rng)
        beta *= pow(allele_freq * (1 - allele_freq), self.alpha)
        return beta
        
    @property
    def _require_allele_freq(self):
        return True  