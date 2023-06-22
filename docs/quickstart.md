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

(sec_quickstart)=

# Quickstart

This page provides an example of how to use **tstrait** to simulate quantitative traits of individuals in the tree sequence data. See the {ref}`sec_installation` page for instructions on installing **tstrait**.

## Example

In the following example, we will be using {ref}`msprime <msprime:sec_intro>` to simulate genetic information of 2000 individuals with 1Mb chromosome by using human-like parameters for mutation and recombination rate. We will then be using **tstrait** to simulate quantitative traits of those simulated individuals, assuming that there are 1000 causal sites. We will be setting a trait model {ref}`sec_trait_model_allele`, and simulating quantitative traits of individuals by using the {func}`.sim_phenotype` function. Afterwards, the results will be visualized by using [matplotlib](https://matplotlib.org/) package.

```{code-cell} ipython3
import msprime
import tstrait
import matplotlib.pyplot as plt

num_ind = 2000
ts = msprime.sim_ancestry(num_ind, sequence_length=1_000_000, recombination_rate=1e-8,
                          population_size=10**4, random_seed=1)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)

model = tstrait.TraitModelAlleleFrequency(trait_mean=0, trait_sd=1, alpha=-0.3)
sim_result = tstrait.sim_phenotype(ts, num_causal=1000, model=model, h2=0.3, random_seed=1)

phenotype_result = sim_result.phenotype
genotype_result = sim_result.genotype
```

In the above quantitative trait simulation, we set the narrow-sense heritability $h^2$ to be `0.3`, the trait mean to be `0` and the trait standard deviation to be `1`. The parameters of the model are described in detail in {ref}`sec_simulation` and {ref}`sec_trait_model` page. We set the `random_seed` to ensure that the output of the simulation model is the same.

The output of the {func}`.sim_phenotype` function is described in detail in {ref}`sec_simulation_output` section.

The distribution of simulated phenotype of 500 individuals is shown in the histogram below.

```{code-cell} ipython3
plt.hist(phenotype_result.phenotype, bins=20)
plt.xlabel("Phenotype")
plt.show()
```

The relationship between causal allele frequency and effect size for each causal site is shown in the scatter plot below.

```{code-cell} ipython3
plt.scatter(genotype_result.allele_frequency, genotype_result.effect_size)
plt.xlabel("Causal allele frequency")
plt.ylabel("Effect size")
plt.axhline(y=0, color='r', linestyle='-')
plt.show()
```