import numpy as np

MODEL_MAP = {
    "gcta": TraitModelGCTA,
    "allele": TraitModelAllele,
    # "ldak": TraitModelLDAK, Needs alpha argument
}


def effect_size_model(model, trait_mean, trait_sd, num_causal, allele_freq):
    """
    Returns a mutation model corresponding to the specified model.
    - If model is None, the default mutation model is returned.
    - If model is a string, return the corresponding model instance.
    - If model is an instance of MutationModel, return it.
    - Otherwise raise a type error.
    """

    if model is None:
        model_instance = TraitModelGCTA(trait_mean, trait_sd, num_causal, allele_freq)
    elif isinstance(model, str):
        lower_model = model.lower()
        if lower_model not in MODEL_MAP:
            raise ValueError(
                "Model '{}' unknown. Choose from {}".format(
                    model, sorted(MODEL_MAP.keys())
                )
            )
        model_instance = MODEL_MAP[lower_model](trait_mean, trait_sd, num_causal, allele_freq)
    elif isinstance(model, TraitModel):
        model_instance = model
    else:
        raise TypeError(
            "Mutation model must be a string or an instance of TraitModel"
        )
    return model_instance

class TraitModel:
# Trait model class
    def __init__(self, model_name, trait_mean, trait_sd, num_causal, allele_freq):
        self._model_name = model_name
        self.trait_mean = trait_mean
        self.trait_sd = trait_sd
        self.num_causal = num_causal
        self.allele_freq = allele_freq
    
    def sim_effect_size(self, rng):
        beta = rng.normal(loc=self.trait_mean, scale=self.trait_sd / np.sqrt(self.num_causal))
        return beta   
    @property
    def name(self):
        return self._model_name

class TraitModelGCTA(TraitModel):
# GCTA model (Effect size simulation won't be affected by allele frequency)
    def __init__(self, trait_mean, trait_sd, num_causal, allele_freq):
        super().__init__('gcta', trait_mean, trait_sd, num_causal, allele_freq)

class TraitModelAllele(TraitModel):
# Allele model (Effect size will be affected by allele frequency)
    def __init__(self, trait_mean, trait_sd, num_causal, allele_freq):
        super().__init__('allele', trait_mean, trait_sd, num_causal, allele_freq)
    def sim_effect_size(self, rng):
        beta = super().sim_effect_size(rng)
        beta /= np.sqrt(2 * self.allele_freq * (1 - self.allele_freq))
        return beta
    
class TraitModelLDAK(TraitModel):
# LDAK model (Effect size will be affected by allele frequency and alpha parameter)
    def __init__(self, trait_mean, trait_sd, num_causal, allele_freq, alpha):
        super().__init__('ldak', trait_mean, trait_sd, num_causal, allele_freq)
        self.alpha = alpha
    def sim_effect_size(self, rng):
        beta = super().sim_effect_size(rng)
        beta *= pow(self.allele_freq * (1 - self.allele_freq), self.alpha)
        return beta    
       