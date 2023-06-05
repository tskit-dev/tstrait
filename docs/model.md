# Trait Model

The effect size $\beta_j$ of SNP $j$ is simulated from a Gaussian distribution which depends on causal allele frequency $p_j$,

$$
    \beta_j\sim N\left(\mu,[2p_j(1-p_j)]^\alpha \frac{\sigma^2}{m}\right),
$$

where $m$ in the number of causal sites inside the model. The parameters $\mu$, $\sigma$ and $\alpha$ are specified in **tstrait** when we set the trait model in the simulation. For example,

```Python
model = tstrait.TraitModelAllele(trait_mean = 0, trait_sd = 1, alpha = -1)
```

sets the model with $\mu=0$, $\sigma=0$ and $\alpha=-1$. Negative $\alpha$ value can increase the magnitude of effect sizes on rarer variants. The distribution of the effect size depends on the allele frequency, as it has been suggested that rare variants can have effect sizes with larger magnitude compared with common variants due to negative selection. The detailed descriptions of the relationship between $\alpha$ and the predictability of human traits are indicated in [Schoech et al. (2019)](https://doi.org/10.1038/s41467-019-08424-6) and [Speed et al. (2017)](https://doi.org/10.1038/ng.3865).

The effect size simulation in `tstrait.TraitModelAdditive` model does not depend on allele frequency, and it is a special case of the `tstrait.TraitModelAllele` model with $\alpha = 0$. Only $\mu$ and $\sigma$ parameters need to be defined in `tstrait.TraitModelAllele` model.