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

(multi_trait)=

# Multi-trait simulation

This page describes how to simulate multiple correlated traits in tstrait.

**Learning Objectives**

After this effect size page, you will be able to:

- Understand how multi-trait simulation is conducted in tstrait.
- Understand the trait models and inputs of multi-trait simulation.

## Pleiotropy

tstrait supports simulation of multiple correlated traits, assuming that they are
influenced by pleiotropic genes. Pleiotropy is used to describe the phenomenon that
a single gene contributes to multiple traits (See
[here](https://www.nature.com/scitable/topicpage/pleiotropy-one-gene-can-affect-multiple-traits-569/)
for details). The only trait model that is supported to simulate multiple traits
in tstrait is multivariate normal distribution
({py:class}`TraitModelMultivariateNormal`). We will be showing how to simulate
traits by using a multivariate normal distribution trait model.

## Multivariate normal distribution

The multivariate normal distribution trait model can be specified by inputting the
mean vector and covariance matrix in {py:func}`trait_model`. Note that the covariance
matrix must be symmetric and positive-semidefinite, and the dimensions of the mean vector
and covariance matrix must match.

In the following example, we will be generating a multivariate normal distribution trait
model with mean vector being a vector of zeros and covariance matrix being an identity
matrix.

```{code-cell}

import tstrait
import numpy as np

model = tstrait.trait_model(
    distribution="multi_normal", mean=np.zeros(2), cov=np.eye(2)
)
model.num_trait
```

Note that 2 traits will be simulated from this model, as the dimension of the mean vector
is 2.

## Multi-trait simulation

We will now be simulating multiple traits by using a tree sequence data with 3 individuals
and 2 causal sites.

```{code-cell}

import msprime

ts = msprime.sim_ancestry(
    samples=3,
    recombination_rate=1e-8,
    sequence_length=1_000_000,
    population_size=10_000,
    random_seed=5,
)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=5)

sim_result = tstrait.sim_phenotype(
    ts=ts, num_causal=2, model=model, h2=[0.3, 0.3], random_seed=1
)

sim_result.phenotype
```

```{code-cell}
sim_result.trait
```

:::{note}
The dimension of narrow-sense heritability `h2` must match the number of traits being
simulated, and the multivariate normal distribution trait model can also be used as a model
input of {py:func}`sim_trait` as well.
:::

In the above example, phenotypic and effect size information of 2 traits are being simulated.
The `trait_id` column represents the trait ID of the simulated traits.

As a next example, we will be illustrating correlated quantitative traits by simulating
correlated traits with 1000 causal sites and 1000 individuals.

```{code-cell}

ts = msprime.sim_ancestry(
    samples=1000,
    recombination_rate=1e-8,
    sequence_length=1_000_000,
    population_size=10_000,
    random_seed=5,
)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=5)

cov = np.array([[1, 0.9], [0.9, 1]])

model = tstrait.trait_model(distribution="multi_normal", mean=np.zeros(2), cov=cov)

sim_result = tstrait.sim_phenotype(
    ts=ts, num_causal=100, model=model, h2=[0.8, 0.8], random_seed=1
)
```

We will be showing the correlation by creating a scatterplot in {py:mod}`matplotlib`.

```{code-cell}

import matplotlib.pyplot as plt

phenotype_df = sim_result.phenotype
trait0 = phenotype_df.loc[phenotype_df["trait_id"] == 0].phenotype
trait1 = phenotype_df.loc[phenotype_df["trait_id"] == 1].phenotype
plt.scatter(trait0, trait1)
plt.xlabel("Trait 0")
plt.ylabel("Trait 1")
plt.title("Polygenic Traits")
plt.show()
```
