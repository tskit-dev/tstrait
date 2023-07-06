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

In the following example, we will use {ref}`msprime <msprime:sec_intro>` to simulate the ancestry of 2000 individuals with mutations, along a 1Mb chromosome. This generates a tree sequence which serves as the input of **tstrait** to simulate quantitative traits for all individuals in our simulated sample.


```{code-cell} ipython3
import msprime
import tstrait
import matplotlib.pyplot as plt

num_ind = 2000
ts = msprime.sim_ancestry(num_ind, sequence_length=1_000_000, recombination_rate=1e-8,
                          population_size=10**4, random_seed=1)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)

model = tstrait.TraitModelAlleleFrequency(trait_mean=0, trait_var=1, alpha=-0.3)
sim_result = tstrait.sim_phenotype(ts, num_causal=1000, model=model, h2=0.3, random_seed=1)
```

We first specify {ref}`sec_trait_model_allele`. In this case the model specifies that the effect sizes for the random subset of a 1000 causal variants are a function of allele frequency. The model further requires us to set the narrow-sense heritability $h^2$ and a trait mean and standard deviation. The parameters required for each model are described in detail in {ref}`sec_simulation` and {ref}`sec_trait_model` page.

The output of the {func}`.sim_phenotype` function is twofold. It returns both information on the random set of causal variants (`sim_result.genotype`) as well as the simulated phenotypes (`sim_result.phenotype`). All information in the output is described in detail in the {ref}`sec_simulation_output` section.

We can now visualize the results using [matplotlib](https://matplotlib.org/). The distribution of simulated phenotypes of the 2000 individuals in the tree sequence data is shown in the histogram below.

```{code-cell} ipython3
plt.hist(sim_result.phenotype.phenotype, bins=20)
plt.xlabel("Phenotype")
plt.show()
```

The relationship between causal allele frequency and effect size for each causal site is shown in the scatter plot below.

```{code-cell} ipython3
plt.scatter(sim_result.genotype.allele_frequency, sim_result.genotype.effect_size)
plt.xlabel("Causal allele frequency")
plt.ylabel("Effect size")
plt.axhline(y=0, color='r', linestyle='-')
plt.show()
```
