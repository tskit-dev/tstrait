import numpy as np

class TraitModel:
# Trait model class
    def __init__(self, model_name, trait_mean, trait_sd):
        self._model_name = model_name
        self.trait_mean = trait_mean
        self.trait_sd = trait_sd
    
    def sim_effect_size(self, num_causal, rng):
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

class TraitModelAllele(TraitModel):
# Allele model (Effect size will be affected by allele frequency)
    def __init__(self, trait_mean, trait_sd):
        super().__init__('allele', trait_mean, trait_sd)
    def sim_effect_size(self, num_causal, allele_freq, rng):
        beta = super().sim_effect_size(num_causal, rng)
        beta /= np.sqrt(2 * allele_freq * (1 - allele_freq))
        return beta
        
    @property
    def _require_allele_freq(self):
        return True
    
class TraitModelLDAK(TraitModel):
# LDAK model (Effect size will be affected by allele frequency and alpha parameter)
    def __init__(self, trait_mean, trait_sd, alpha):
        super().__init__('ldak', trait_mean, trait_sd)
        self.alpha = alpha
    def sim_effect_size(self, num_causal, allele_freq, rng):
        beta = super().sim_effect_size(num_causal, rng)
        beta *= pow(allele_freq * (1 - allele_freq), self.alpha)
        return beta
        
    @property
    def _require_allele_freq(self):
        return True
        
"""
    if (not isinstance(trait_mean, int) and not isinstance(trait_mean, float)):
        raise TypeError("Mean value of traits should be a float or an integer")
    if (not isinstance(trait_sd, int) and not isinstance(trait_sd, float)):
        raise TypeError("Standard deviation of traits should be a float or an integer")  
    if trait_sd < 0:
        raise ValueError("Standard deviation of traits should be a non-negative number")  
"""        