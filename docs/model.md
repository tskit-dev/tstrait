---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Trait Model

The effect size $\beta_j$ of SNP $j$ in the {class}`.TraitModelAllele` model is simulated from a Gaussian distribution which depends on causal allele frequency $p_j$,

$$
    \beta_j\sim N\left(\mu,[2p_j(1-p_j)]^\alpha\cdot \frac{\sigma^2}{m}\right).
$$

In the above equation, $m$ is the number of causal sites inside the model, which is set in {func}`.sim_phenotype` function. The parameters $\mu$, $\sigma$ and $\alpha$ are specified in `trait_mean`, `trait_sd` and `alpha` arguments when we set the trait model in the simulation. For example,

```Python
model = tstrait.TraitModelAllele(trait_mean = 0, trait_sd = 1, alpha = -1)
```

sets the {class}`.TraitModelAllele` model with $\mu=0$, $\sigma=0$ and $\alpha=-1$.

The distribution of the effect size in the {class}`.TraitModelAllele` model depends on the allele frequency, as it has been suggested that negative selection can lead to rare variants having effect sizes with larger magnitude compared with common variants. Negative $\alpha$ value can increase the magnitude of effect sizes on rare variants. The relationship between $\alpha$ and the predictability of human traits are indicated in [Schoech et al. (2019)](https://doi.org/10.1038/s41467-019-08424-6) and [Speed et al. (2017)](https://doi.org/10.1038/ng.3865).

Two trait models are supported in **tstrait**:

- {class}`.TraitModelAdditive`
- {class}`.TraitModelAllele`


The effect size simulation in {class}`.TraitModelAdditive` model does not depend on allele frequency, and it is a special case of {class}`.TraitModelAllele` model with $\alpha = 0$. Only $\mu$ and $\sigma$ need to be defined in {class}`.TraitModelAllele` model.