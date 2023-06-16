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

# Quickstart

This page provides an example of how to use **tstrait** to simulate quantitative traits of individuals in the tree sequence data. See the [Installation](installation.md) page for instructions on installing **tstrait**.

## Example

In the following example, we will be using {ref}`msprime <msprime:sec_intro>` to simulate genetic information of 2000 individuals with 1Mb chromosome by using human-like parameters for mutation and recombination rate. We will then be using **tstrait** to simulate quantitative traits of those simulated individuals, assuming that there are 1000 causal sites. We will be setting a trait model {class}`.TraitModelAlleleFrequency`, and simulating quantitative traits of individuals in {func}`.sim_phenotype`. Afterwards, the results will be visualized by using [matplotlib](https://matplotlib.org/) package.

```{code-cell} ipython3
import msprime
import tstrait
import matplotlib.pyplot as plt

num_ind = 2000
ts = msprime.sim_ancestry(num_ind, sequence_length=1_000_000, recombination_rate=1e-8,
                          population_size=10**4, random_seed=1)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)

model = tstrait.TraitModelAlleleFrequency(trait_mean=0, trait_sd=1, alpha=-0.3)
phenotype_result, genetic_result = tstrait.sim_phenotype(ts, num_causal=1000, model=model,
                                                         h2=0.3, random_seed=1)
```

In the above quantitative trait simulation, we set the narrow-sense heritability $h^2$ to be `0.3`, the trait mean to be `0` and the trait standard deviation to be `1`. The parameters of the model are described in detail in [Simulation Model](simulation.md) and [Trait Model](model.md) page. We set the `random_seed` to ensure that the output of the simulation model is the same.

The distribution of simulated phenotype of 500 individuals is shown in the histogram below.

```{code-cell} ipython3
plt.hist(phenotype_result.phenotype, bins=20)
plt.xlabel("Phenotype")
plt.show()
```

The relationship between allele frequency and SNP effect sizes is shown in the scatter plot below.

```{code-cell} ipython3
plt.scatter(genetic_result.allele_frequency, genetic_result.effect_size)
plt.xlabel("Allele frequency")
plt.ylabel("Effect size")
plt.axhline(y=0, color='r', linestyle='-')
plt.show()
```

The output of {func}`.sim_phenotype` is described in detail in [Output](output.md) page.