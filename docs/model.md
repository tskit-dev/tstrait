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

(sec_trait_model)=

# Trait Model

The trait model determines how SNP effect sizes are simulated. **tstrait** currently supports two trait models, {ref}`sec_trait_model_additive` and {ref}`sec_trait_model_allele`.

(sec_trait_model_additive)=

## TraitModelAdditive

With this model, the effect size $\beta_j$ of SNP $j$ is simulated from a Gaussian distribution,

$$
\beta_j\sim N\left(\mu, \frac{\sigma^2}{m}\right).
$$

Here $m$ is the user-defined number of causal sites, and $\mu$ and $\sigma^2$ are the specified `trait_mean` and `trait_var` controlling the shape of the distribution. For example,

```Python
model = tstrait.TraitModelAdditive(trait_mean=0, trait_var=1)
```

sets the {class}`.TraitModelAdditive` model with $\mu=0$ and $\sigma^2=1$.

### Example

In the below example, we will be simulating quantitative traits from {class}`.TraitModelAdditive` by using the simulated tree sequence data in {ref}`msprime <msprime:sec_intro>` to show the relationship between allele frequency and effect size. The distribution of effect size does not depend on the allele frequency.

```{code-cell} ipython3
import msprime
import tstrait
import matplotlib.pyplot as plt

num_ind = 500
ts = msprime.sim_ancestry(num_ind, sequence_length=1_000_000, recombination_rate=1e-8,
                          population_size=10**4, random_seed=1)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)

model = tstrait.TraitModelAdditive(trait_mean=0, trait_var=1)
sim_result = tstrait.sim_phenotype(ts, num_causal=1000, model=model, h2=0.3, random_seed=1)

plt.scatter(sim_result.genotype.allele_frequency, sim_result.genotype.effect_size)
plt.xlabel("Allele frequency")
plt.ylabel("Effect size")
plt.axhline(y=0, color='r', linestyle='-')
plt.title("TraitModelAdditive")
plt.show()
```

(sec_trait_model_allele)=

## TraitModelAlleleFrequency

The effect size $\beta_j$ of SNP $j$ in the {class}`.TraitModelAlleleFrequency` model is simulated from a Gaussian distribution which depends on causal allele frequency $p_j$,

$$
    \beta_j\sim N\left(\mu,[2p_j(1-p_j)]^\alpha\cdot \frac{\sigma^2}{m}\right).
$$

In the above equation, $m$ is the number of causal sites inside the simulation model, and it is set in {func}`.sim_phenotype` function. The parameters $\mu$, $\sigma^2$ and $\alpha$ are specified in `trait_mean`, `trait_var` and `alpha` arguments when we set the trait model in the simulation. For example,

```Python
model = tstrait.TraitModelAlleleFrequency(trait_mean=0, trait_var=1, alpha=-1)
```

sets the {class}`.TraitModelAlleleFrequency` model with $\mu=0$, $\sigma^2=1$ and $\alpha=-1$.

The distribution of effect size in the {class}`.TraitModelAlleleFrequency` model depends on the allele frequency, as it has been shown that rare variants have increased effect sizes compared with common variants. Many simulation studies employ this frequency dependent architecture with a negative $\alpha$ value, as it increases the magnitude of effect sizes on rare variants. The details of the model and the relationship between $\alpha$ and the predictability of human traits are indicated in [Speed et al. (2017)](https://doi.org/10.1038/ng.3865). The {ref}`sec_trait_model_additive` model is a special case of {ref}`sec_trait_model_allele` model with $\alpha=0$.

In the below example, we will be simulating quantitative traits by using the same simulated tree sequence data that was used in the {ref}`sec_trait_model_additive` example to show the relationship between effect sizes and the `alpha` parameter.

### Example with $\alpha$=-0.3:

The simulation model puts some emphasis on effect sizes from rarer variants when $\alpha$ is a negative number.

```{code-cell} ipython3
model = tstrait.TraitModelAlleleFrequency(trait_mean=0, trait_var=1, alpha=-0.3)
sim_result = tstrait.sim_phenotype(ts, num_causal=1000, model=model, h2=0.3, random_seed=1)

plt.scatter(sim_result.genotype.allele_frequency, sim_result.genotype.effect_size)
plt.xlabel("Allele frequency")
plt.ylabel("Effect size")
plt.axhline(y=0, color='r', linestyle='-')
plt.title("TraitModelAlleleFrequency, alpha = -0.3")
plt.show()
```

### Example with $\alpha$=-0.6:

When $\alpha$ is set to be a smaller number, the simulation model puts greater emphasis on effect sizes from rarer variants compared with the previous example.

```{code-cell} ipython3
model = tstrait.TraitModelAlleleFrequency(trait_mean=0, trait_var=1, alpha=-0.6)
sim_result = tstrait.sim_phenotype(ts, num_causal=1000, model=model, h2=0.3, random_seed=1)

plt.scatter(sim_result.genotype.allele_frequency, sim_result.genotype.effect_size)
plt.xlabel("Allele frequency")
plt.ylabel("Effect size")
plt.axhline(y=0, color='r', linestyle='-')
plt.title("TraitModelAlleleFrequency, alpha = -0.6")
plt.show()
```
