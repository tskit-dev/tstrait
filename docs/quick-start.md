---
kernelspec:
  name: python3
  display_name: python3
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.13'
    jupytext_version: 1.13.8
---

```{eval-rst}
.. currentmodule:: tstrait
```

(quickstart)=

# Quick start

This page provides a quick overview of tstrait. We will be using {func}`sim_phenotype` to
demonstrate how quantitative traits can be simulated in tstrait.

To work with the examples, you will need to install
[msprime](msprime:sec_intro) and {mod}`matplotlib <matplotlib>` in
addition to tstrait.

**Learning Objectives**

After this quick start page, you will be able to:

- Understand how to simulate quantitative traits in tstrait
- Understand how to read the output of tstrait

## Input

The main objective of tstrait is to simulate quantitative traits from a tree sequence input. The important
inputs of {func}`sim_phenotype` are:

ts

: The tree-sequence input with mutation

num_causal

: Number of causal sites

model

: The trait model where the effect sizes are going to be simulated

h2

: Narrow-sense heritability

## Example

```{code-cell}

  import tstrait
  import msprime

  ts = msprime.sim_ancestry(
      samples=10_000,
      recombination_rate=1e-8,
      sequence_length=100_000,
      population_size=10_000,
      random_seed=100,
  )
  ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=101)
  ts.num_individuals
```

Here, we have simulated a sample tree sequence with 10,000 individuals in [msprime](msprime:sec_intro).
We will be using it in {func}`sim_phenotype` to simulate quantitative traits.

```{code-cell}

  model = tstrait.trait_model(distribution="normal", mean=0, var=1)
  sim_result = tstrait.sim_phenotype(
      ts=ts, num_causal=100, model=model, h2=0.3, random_seed=1
  )
```

The `sim_result` variable created above contains simulated information of phenotype and genetic effect sizes
which will be shown below.

(effect_size_output)=

## Effect Size Output

Simulated effect sizes from {func}`sim_phenotype` can be extracted through `.effect_size`.

```{code-cell}

  effect_size_df = sim_result.effect_size
  effect_size_df.columns
  effect_size_df.head()
```

The `effect_size_df` is a {class}`pandas.DataFrame` object that includes the following 4 columns:

> - **site_id**: ID of sites that have causal mutation
> - **effect_size**: Genetic effect size of causal mutation
> - **trait_id**: Trait ID and will be used in multi-trait simulation.
> - **causal_state**: Causal state.
> - **allele_frequency**: Allele frequency of causal mutation.

(phenotype_output)=

## Phenotype Output

Simulated phenotypes from {func}`sim_phenotype` can be extracted through `.phenotype`.

```{code-cell}

  phenotype_df = sim_result.phenotype
  phenotype_df.columns
  phenotype_df.head()
```

The `phenotype_df` is a {class}`pandas.DataFrame` object that includes the following 5 columns:

> - **trait_id**: Trait ID and will be used in multi-trait simulation.
> - **individual_id**: Individual ID inside the tree sequence input.
> - **genetic_value**: Genetic value of individuals.
> - **environmental_noise**: Simulated environmental noise of individuals.
> - **phenotype**: Simulated phenotype, and it is a sum of **genetic_value** and **environmental_noise**.

We will be using {mod}`matplotlib <matplotlib>` to create a histogram of the simulated phenotype and
environmental noise.

```{code-cell}

  import matplotlib.pyplot as plt
  plt.hist(phenotype_df["phenotype"], bins=40)
  plt.title("Phenotype")
  plt.show()
```

```{code-cell}

  plt.hist(phenotype_df["environmental_noise"], bins=40)
  plt.title("Environmental Noise")
  plt.show()
```

The environmental noise in tstrait follows a normal distribution. Please see [](phenotype_model)
for mathematical details on the phenotype model and [](effect_size_dist) for details on
specifying the effect size distribution.
