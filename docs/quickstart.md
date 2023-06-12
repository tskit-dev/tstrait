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

This page provides an example of how to use **tstrait** to simulate quantitative traits of individuals in the tree sequence data. We will be using **msprime** to simulate tree sequence data, and users who are interested in the details of **msprime** should consult {ref}`msprime manual <msprime:sec_intro>`.

See the [Installation](installation.md) page for instructions on installing **tstrait**.

## Trait Model

**tstrait** supports simulation of quantitative traits assuming that they are all additive. As a beginning, we need to specify the trait model to determine how effect sizes of genes are being simulated. The details of the model are indicated in [trait model](model.md) page.

It would be necessary for the user to set the mean and standard deviation of the genetic effect sizes when they define the trait model. For example,

```Python
import tstrait
model = tstrait.TraitModelAllele(trait_mean = 0, trait_sd = 1, alpha = -0.3)
```

sets a trait model {class}`.TraitModelAllele`, where the mean value of traits is 0 and the standard deviation is 1. We will then be using this `model` in {func}`.sim_phenotype` to simulate quantitative traits of individuals in the tree sequence data.

## Example

In the following example, we will be using **msprime** to simulate a tree sequence dataset, and simulate quantitative traits of simulated individuals having 1Mb chromosome. The simulation will be conducted by using human-like parameters. We will be setting a trait model {class}`.TraitModelAllele`, and simulating quantitative traits of individuals in {func}`.sim_phenotype`.

```{code-cell} ipython3
import msprime
import tstrait
import matplotlib.pyplot as plt

num_ind = 500
ts = msprime.sim_ancestry(num_ind, sequence_length=1_000_000, recombination_rate=1e-8,
                          population_size=10**4, random_seed=1)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)

model = tstrait.TraitModelAllele(trait_mean = 0, trait_sd = 1, alpha = -0.3)
phenotype_result, genetic_result = tstrait.sim_phenotype(ts,num_causal = 1000, model = model,
                                                         h2 = 0.3, random_seed = 1)
```

The example above simulates tree sequence data with 500 individuals and then simulates quantitative traits of those individuals with 1000 causal sites. The narrow sense heritability `h2` is used to determine the environmental noise. We also set the `random_seed` to ensure that the output of the simulation model is the same.

The distribution of simulated phenotype of 500 individuals is shown below.

```{code-cell} ipython3
plt.hist(phenotype_result.phenotype)
plt.xlabel("Phenotype")
plt.show()
```

The relationship between allele frequency and SNP effect sizes is shown below.

```{code-cell} ipython3
plt.scatter(genetic_result.allele_frequency, genetic_result.effect_size)
plt.xlabel("Allele frequency")
plt.ylabel("Effect size")
plt.axhline(y = 0, color = 'r', linestyle = '-')
plt.show()
```

The detailed explanation of the output of {func}`.sim_phenotype` is described in [Output](output.md) page.