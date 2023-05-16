import pytest
import numpy as np
import tstrait.trait as trait
          
class Test_TraitModelGCTA:
    @pytest.mark.parametrize("trait_mean", [0, 1, 1.1, -1, -1.1])
    @pytest.mark.parametrize("trait_sd", [0.1, 1])
    @pytest.mark.parametrize("num_causal", [1, 5])
    @pytest.mark.parametrize("random_seed", [1,2,None])
    def test_pass_condition(self, trait_mean, trait_sd, num_causal, random_seed):
        model = trait.TraitModelGCTA(trait_mean, trait_sd)
        assert model.name == "gcta"
        assert model._require_allele_freq == False
        
        rng = np.random.default_rng(random_seed)
        beta = model.sim_effect_size(num_causal, rng)
        
        assert isinstance(beta, float)
        
    @pytest.mark.parametrize("trait_mean", ["a", "1", [1, 2], {"trait_mean": 1}, None])
    def test_mean(self, trait_mean):
        with pytest.raises(TypeError, match="Mean value of traits should be a float or an integer"):
            model = trait.TraitModelGCTA(trait_mean, 1)

    @pytest.mark.parametrize("trait_sd", ["a", "1", [1, 2], {"trait_sd": 1}, None])
    def test_sd(self, trait_sd):
        with pytest.raises(TypeError, match="Standard deviation of traits should be a float or an integer"):
            model = trait.TraitModelGCTA(0, trait_sd)
    
    @pytest.mark.parametrize("trait_sd", [-0.1, -1])
    def test_negative_sd(self, trait_sd):
        with pytest.raises(ValueError, match="Standard deviation of traits should be a non-negative number"):
            model = trait.TraitModelGCTA(0, trait_sd)

    @pytest.mark.parametrize("trait_mean", [0, 1, 1.1, -1, -1.1])
    @pytest.mark.parametrize("num_causal", [1, 5])
    @pytest.mark.parametrize("random_seed", [1,2])
    def test_zero_sd(self, trait_mean, num_causal, random_seed):
        model = trait.TraitModelGCTA(trait_mean, 0)
        assert model.name == "gcta"
        assert model._require_allele_freq == False
        
        rng = np.random.default_rng(random_seed)
        beta = model.sim_effect_size(num_causal, rng)
        
        assert beta == trait_mean

    @pytest.mark.parametrize("num_causal", [1.0, 0.1, -1.0, "a",None])
    def test_num_causal(self, num_causal):
        model = trait.TraitModelGCTA(0,1)
        rng = np.random.default_rng(5)
        with pytest.raises(TypeError, match="Number of causal sites should be an integer"):
            beta = model.sim_effect_size(num_causal, rng)
            
    @pytest.mark.parametrize("num_causal", [0, -1])
    def test_num_causal_negative(self, num_causal):
        model = trait.TraitModelGCTA(0,1)
        rng = np.random.default_rng(1)
        with pytest.raises(ValueError, match="Number of causal sites should be a positive integer"):
            beta = model.sim_effect_size(num_causal, rng)

    @pytest.mark.parametrize("rng", [1, 1.1, "a", 0, None])
    def test_rng(self, rng):
        model = trait.TraitModelGCTA(0,1)

        with pytest.raises(TypeError, match="rng should be a numpy random generator"):
            beta = model.sim_effect_size(5, rng)

class Test_TraitModelAllele:
    @pytest.mark.parametrize("trait_mean", [0, 1, 1.1, -1, -1.1])
    @pytest.mark.parametrize("trait_sd", [0.1, 1])
    @pytest.mark.parametrize("num_causal", [1, 5])
    @pytest.mark.parametrize("random_seed", [1,2,None])
    @pytest.mark.parametrize("allele_freq", [0.1, 0.99])    
    def test_pass_condition(self, trait_mean, trait_sd, num_causal, allele_freq, random_seed):
        model = trait.TraitModelAllele(trait_mean, trait_sd)
        assert model.name == "allele"
        assert model._require_allele_freq == True
        
        rng = np.random.default_rng(random_seed)
        beta = model.sim_effect_size(num_causal, allele_freq, rng)
        
        assert isinstance(beta, float)

    @pytest.mark.parametrize("trait_mean", ["a", "1", [1, 2], {"trait_mean": 1}, None])    
    def test_mean(self, trait_mean):
        with pytest.raises(TypeError, match="Mean value of traits should be a float or an integer"):
            model = trait.TraitModelAllele(trait_mean, 1)

    @pytest.mark.parametrize("trait_sd", ["a", "1", [1, 2], {"trait_sd": 1}, None])
    def test_sd(self, trait_sd):
        with pytest.raises(TypeError, match="Standard deviation of traits should be a float or an integer"):
            model = trait.TraitModelAllele(0, trait_sd)
            
    @pytest.mark.parametrize("trait_sd", [-0.1, -1])
    def test_negative_sd(self, trait_sd):
        with pytest.raises(ValueError, match="Standard deviation of traits should be a non-negative number"):
            model = trait.TraitModelAllele(0, trait_sd)
            
    @pytest.mark.parametrize("trait_mean", [0, 1, 1.1, -1, -1.1])
    @pytest.mark.parametrize("num_causal", [1, 5])
    @pytest.mark.parametrize("random_seed", [1,2])
    @pytest.mark.parametrize("allele_freq", [0.1, 0.99]) 
    def test_zero_sd(self, trait_mean, num_causal, allele_freq, random_seed):
        model = trait.TraitModelAllele(trait_mean, 0)
        assert model.name == "allele"
        assert model._require_allele_freq == True
        
        rng = np.random.default_rng(random_seed)
        beta = model.sim_effect_size(num_causal, allele_freq, rng)
        
        assert np.isclose(beta, trait_mean / np.sqrt(2 * allele_freq * (1 - allele_freq)))
        
    @pytest.mark.parametrize("num_causal", [1.0, 0.1, -1.0, "a", None])   
    def test_num_causal(self, num_causal):
        model = trait.TraitModelAllele(0,1)
        rng = np.random.default_rng(5)
        with pytest.raises(TypeError, match="Number of causal sites should be an integer"):
            beta = model.sim_effect_size(num_causal, 0.5, rng)
            
    @pytest.mark.parametrize("num_causal", [0, -1])
    def test_num_causal_negative(self, num_causal):
        model = trait.TraitModelAllele(0,1)
        rng = np.random.default_rng(5)
        with pytest.raises(ValueError, match="Number of causal sites should be a positive integer"):
            beta = model.sim_effect_size(num_causal, 0.3, rng)

    @pytest.mark.parametrize("rng", [1, 1.1, "a", 0, None])
    def test_rng(self, rng):
        model = trait.TraitModelAllele(0,1)

        with pytest.raises(TypeError, match="rng should be a numpy random generator"):
            beta = model.sim_effect_size(1, 0.3, rng)

    @pytest.mark.parametrize("allele_freq", [2, "a", [0.1, 0.2], None])    
    def test_allele_freq_type(self, allele_freq):
        model = trait.TraitModelAllele(0,1)
        rng = np.random.default_rng(1)

        with pytest.raises(TypeError, match="Allele frequency should be a float"):
            beta = model.sim_effect_size(2, allele_freq, rng)
            
    @pytest.mark.parametrize("allele_freq", [0.0, 1.0, -0.1, 1.01])    
    def test_allele_freq_value(self, allele_freq):
        model = trait.TraitModelAllele(0,1)
        rng = np.random.default_rng(1)

        with pytest.raises(ValueError, match="Allele frequency should be 0 < Allele frequency < 1"):
            beta = model.sim_effect_size(5, allele_freq, rng)
    
       
class Test_TraitModelLDAK:
    @pytest.mark.parametrize("trait_mean", [0, 1, 1.1, -1, -1.1])
    @pytest.mark.parametrize("trait_sd", [0.1, 1])
    @pytest.mark.parametrize("num_causal", [1, 5])
    @pytest.mark.parametrize("random_seed", [1,2,None])
    @pytest.mark.parametrize("allele_freq", [0.1, 0.99])  
    @pytest.mark.parametrize("alpha", [0, 1, 1.1, -1, -1.1])
    def test_pass_condition(self, trait_mean, trait_sd, num_causal, allele_freq, alpha, random_seed):
        model = trait.TraitModelLDAK(trait_mean, trait_sd, alpha)
        assert model.name == "ldak"
        assert model._require_allele_freq == True
        
        rng = np.random.default_rng(random_seed)
        beta = model.sim_effect_size(num_causal, allele_freq, rng)
        
        assert isinstance(beta, float)

    @pytest.mark.parametrize("trait_mean", ["a", "1", [1, 2], {"trait_mean": 1}, None])
    def test_mean(self, trait_mean):
        with pytest.raises(TypeError, match="Mean value of traits should be a float or an integer"):
            model = trait.TraitModelLDAK(trait_mean, 1, -1)

    @pytest.mark.parametrize("trait_sd", ["a", "1", [1, 2], {"trait_sd": 1}, None])   
    def test_sd(self, trait_sd):
        with pytest.raises(TypeError, match="Standard deviation of traits should be a float or an integer"):
            model = trait.TraitModelLDAK(0, trait_sd, -1)
    
    @pytest.mark.parametrize("trait_sd", [-0.1, -1]) 
    def test_negative_sd(self, trait_sd):
        with pytest.raises(ValueError, match="Standard deviation of traits should be a non-negative number"):
            model = trait.TraitModelLDAK(0, trait_sd, -1)
            
    @pytest.mark.parametrize("trait_mean", [0, 1, 1.1, -1, -1.1])
    @pytest.mark.parametrize("num_causal", [1, 5])
    @pytest.mark.parametrize("random_seed", [1,2])
    @pytest.mark.parametrize("allele_freq", [0.1, 0.99])
    @pytest.mark.parametrize("alpha", [0, 1, 1.1, -1, -1.1])
    def test_zero_sd(self, trait_mean, num_causal, allele_freq, alpha, random_seed):
        model = trait.TraitModelLDAK(trait_mean, 0, alpha)
        assert model.name == "ldak"
        assert model._require_allele_freq == True
        
        rng = np.random.default_rng(random_seed)
        beta = model.sim_effect_size(num_causal, allele_freq, rng)
        assert np.isclose(beta, trait_mean * pow(allele_freq * (1 - allele_freq), alpha))    

    @pytest.mark.parametrize("num_causal", [1.0, 0.1, -1.0, "a", None])  
    def test_num_causal(self, num_causal):
        model = trait.TraitModelLDAK(0,1,-1)        
        rng = np.random.default_rng(1)
        with pytest.raises(TypeError, match="Number of causal sites should be an integer"):
            beta = model.sim_effect_size(num_causal, 0.3, rng)
            
    @pytest.mark.parametrize("num_causal", [0, -1])
    def test_num_causal_negative(self, num_causal):
        model = trait.TraitModelLDAK(0,1,-1)
        rng = np.random.default_rng(3)
        with pytest.raises(ValueError, match="Number of causal sites should be a positive integer"):
            beta = model.sim_effect_size(num_causal, 0.3, rng)
            
    @pytest.mark.parametrize("rng", [1, 1.1, "a", 0])
    def test_rng(self, rng):
        model = trait.TraitModelLDAK(0,1,-1)

        with pytest.raises(TypeError, match="rng should be a numpy random generator"):
            beta = model.sim_effect_size(5, 0.5, rng)

    @pytest.mark.parametrize("allele_freq", [2, "a", [0.1, 0.2], None])
    def test_allele_freq_type(self, allele_freq):
        model = trait.TraitModelLDAK(0,1,-1)
        rng = np.random.default_rng(1)

        with pytest.raises(TypeError, match="Allele frequency should be a float"):
            beta = model.sim_effect_size(5, allele_freq, rng)
            
    @pytest.mark.parametrize("allele_freq", [0.0, 1.0, -0.1, 1.01])  
    def test_allele_freq_value(self, allele_freq):
        model = trait.TraitModelLDAK(0,1,-1)
        rng = np.random.default_rng(1)

        with pytest.raises(ValueError, match="Allele frequency should be 0 < Allele frequency < 1"):
            beta = model.sim_effect_size(5, allele_freq, rng)
            
    @pytest.mark.parametrize("alpha", [[1,1], "a", {"alpha":1}, None])
    def test_alpha_type(self, alpha):
        with pytest.raises(TypeError, match="Alpha should be a float or an integer"):
            model = trait.TraitModelLDAK(0, 1, alpha)
            
    @pytest.mark.parametrize("trait_mean", [0, 1, 1.1, -1, -1.1])
    @pytest.mark.parametrize("trait_sd", [0.1, 1])
    @pytest.mark.parametrize("num_causal", [1, 5])
    @pytest.mark.parametrize("random_seed", [1,2])
    @pytest.mark.parametrize("allele_freq", [0.1, 0.99])  
    def test_alpha_zero(self, trait_mean, trait_sd, num_causal, allele_freq, random_seed):
        model1 = trait.TraitModelLDAK(trait_mean, trait_sd, 0)
        model2 = trait.TraitModelGCTA(trait_mean, trait_sd)
        assert model1.name == "ldak"
        assert model2.name == "gcta"
        
        rng = np.random.default_rng(random_seed)
        beta1 = model1.sim_effect_size(num_causal, allele_freq, rng)
        rng = np.random.default_rng(random_seed)
        beta2 = model2.sim_effect_size(num_causal, rng)
        
        assert np.isclose(beta1, beta2)