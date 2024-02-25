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

(genetic_value_doc)=

# Genetic value

This page will describe how to simulate genetic values by using {py:func}`genetic_value` function. Genetic
values of individuals are described by $X\beta$, where $X$ is the matrix that describes
the number of causal alleles in each individual and $\beta$ is the vector of effect sizes. See
{ref}`phenotype_model` for mathematical details on the quantitative trait model.

**Learning Objectives**

After this genetic value page, you will be able to:

- Understand how to generate genetic value in tstrait
- Understand how to use the user's defined effect sizes to generate genetic values

# Algorithm Overview

Genetic value is computed in tstrait by using the trait information in the input trait dataframe.
It uses a tree traversal algorithm to count the number of causal allele in each individual and adds
the corresponding effect size to individual's genetic value.

## Input

The input of {py:func}`genetic_value` function is the following:

ts

: Tree sequence input.

trait_df

: Trait dataframe that is described in [](effect_size_sim). There are some requirements for the trait
  dataframe input, which are described in [](req_trait_df).

The details of the parameters and how they influence the genetic value simulation are described in detail
below.

## Genetic value computation

In this example, we will be showing how to compute genetic values by using the simulated trait dataframe
in {py:func}`genetic_value`.

We will be simulating a sample tree sequence data with 5 individuals by using [msprime](msprime:sec_intro),
and we will simulate a trait dataframe with 3 causal sites. The dataframe output of {py:func}`sim_trait`
can be automatically used as an input of {py:func}`genetic_value`, so there is no need to worry about input
data requirements.

:::{seealso}
- [msprime](msprime:sec_intro) for simulating whole genome in tree sequence data.
- [](effect_size) for simulating trait dataframe in tstrait.
:::

```{code-cell}

import msprime
import tstrait

ts = msprime.sim_ancestry(
    samples=5,
    recombination_rate=1e-8,
    sequence_length=100_000,
    population_size=10_000,
    random_seed=100,
)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=101)
model = tstrait.trait_model(distribution="normal", mean=0, var=1)
trait_df = tstrait.sim_trait(ts, num_causal=3, model=model, random_seed=5)
trait_df
```

Please refer to [](effect_size_sim) for details on the trait dataframe output and effect size
simulation by using {py:func}`sim_trait`.

Next, we will be using the trait dataframe to generate genetic values by using {py:func}`genetic_value`.

```{code-cell}

genetic_df = tstrait.genetic_value(ts, trait_df)
genetic_df
```

The output of {py:func}`genetic_value` is a {py:class}`pandas.DataFrame` object that describes
the genetic values of individuals.

The genetic value dataframe includes the following columns:

> - **trait_id**: Trait ID that will be used in multi-trait simulation.
> - **individual_id**: Individual ID inside the tree sequence input.
> - **genetic_value**: Simulated genetic values.

(genetic_user)=

### User defined trait dataframe

We will be demonstrating how to simulate genetic values from a user defined trait dataframe.

(req_trait_df)=

### Dataframe requirements

There are some requirements for the trait dataframe input:

#### Columns

The following columns must be included in `trait_df`:

> - **site_id**: Site IDs that have causal allele.
> - **effect_size**: Simulated effect size of causal allele.
> - **causal_allele**: Causal allele.
> - **trait_id**: Trait ID.

:::{note}
The IDs in the trait_id column must start from 0 and be a consecutive integer. The IDs in the site_id
column must be non-decreasing.
:::

If you are simulating single traits, `trait_id` must be an array of zeros. The method
{py:meth}`pandas.DataFrame.sort_values` can be used to sort the dataframe based on `site_id`.

#### Data requirements

> - Site IDs in **site_id** column must be sorted in an ascending order. Please refer to
>   {py:meth}`pandas.DataFrame.sort_values` for details on sorting values in a
>   {class}`pandas.DataFrame`.
> - Trait IDs in **trait_id** column must start from zero and be consecutive.

We will be demonstrating how to simulate genetic values with 1 causal site.

```{code-cell}

import pandas as pd

data = {"site_id": [0], "effect_size": [1], "trait_id": [0],
        "causal_allele": ["1"]}
trait_df = pd.DataFrame(data)
trait_df
```

```{code-cell}
ts = msprime.sim_ancestry(
    samples=5,
    recombination_rate=1e-8,
    sequence_length=100_000,
    population_size=10_000,
    random_seed=100,
)
ts = msprime.sim_mutations(ts, rate=1e-6, model="binary", random_seed=10)
genetic_df = tstrait.genetic_value(ts, trait_df)
genetic_df
```

As seen in the genetic value dataframe, the genetic values of individuals are multiples
of 1. This is because 1 is the only effect size in the effect size dataframe, and the
{py:func}`genetic_value` function is counting the number of causal alleles of site 0
inside the individual.

Next, we will be inputting an effect size dataframe with multiple causal sites.

```{code-cell}

data = {"site_id": [0, 2, 4], "effect_size": [1, 10, 100],
        "causal_allele": ["1", "1", "1"], "trait_id": [0, 0, 0]}
trait_df = pd.DataFrame(data)
trait_df = trait_df.sort_values(by=["site_id"])
trait_df
```

The site IDs in `"site_id"` column in the trait dataframe must be sorted in an
ascending order, so it is suggested to use `.sort_values(by=['site_id'])`
code before inputting it inside the {py:func}`genetic_value` function.

```{code-cell}

genetic_df = tstrait.genetic_value(ts, trait_df)
genetic_df.head()
```

(numericalise_genotype)=

# Numericalisation of Genotypes

The genotypes are numericalised as the number of causal alleles in each
individual (Please see [](phenotype_model) for mathematical details on the phenotype
model), but it would be possible to change the numericalisation by modifying the
genetic value dataframe based on the effect size dataframe. For example, in the
diploid setting, if you are interested in simulating phenotypes from the genotype
$(aa=-1, Aa=0, AA=1)$, where $A$ is the causal allele, we can simply subtract the
sum of effect sizes from the genetic value.

In the following, we will provide a quick example on how genetic values are being computed
based on the mutation information and effect sizes. We will assume that the `A` allele
represents the causal allele in site 1 and the `B` allele represents the causal allele in
site 2. The effect sizes will be encoded as $\beta_1$ and $\beta_2$ for sites 1 and 2,
respectively. The genotype and genetic value of individuals based on the tstrait's
numericalisation of genotypes $(aa=0, Aa=1, AA=2)$ are shown in the table below:

| **Individual ID** | **Site 1** | **Site 2** | **Genetic Value**  |
|-------------------|------------|------------|--------------------|
| 1                 | Aa         | BB         | $\beta_1+2\beta_2$ |
| 2                 | aa         | Bb         | $\beta_2$          |

If we modify the numericalisation of genotypes to be $(aa=-1, Aa=0, AA=1)$, we get the following:

| **Individual ID** | **Site 1** | **Site 2** | **Genetic Value**  |
|-------------------|------------|------------|--------------------|
| 1                 | Aa         | BB         | $\beta_2$ |
| 2                 | aa         | Bb         | $-\beta_1$          |

When we compare these outputs, we see that the genetic value of individuals in the new encoding
$(aa=-1, Aa=0, AA=1)$ is obtained by subtracting the sum of effect sizes $(\beta_1+\beta_2)$
from the original genetic value.

This can be done in the following example:

```{code-cell}

trait_df = tstrait.sim_trait(ts, num_causal=3, model=model, random_seed=5)
genetic_df = tstrait.genetic_value(ts, trait_df)

# The original dataframe
genetic_df.head()
```

```{code-cell}

genetic_df["genetic_value"] = genetic_df["genetic_value"] - trait_df["effect_size"].sum()
# New dataframe
genetic_df.head()
```

The new genetic value dataframe can be used in {py:func}`sim_env` to simulate phenotypes.

(normalise_genetic_value)=

# Normalise Genetic Value

The computed genetic values can be scaled by using the {py:func}`normalise_genetic_value` function. The function
will first normalise the genetic value by subtracting the mean of the input genetic value and divide it
by the standard devitation of the input genetic value.
Afterwards, it scales the normalised genetic value based on the mean and variance input.
The output of {py:func}`normalise_genetic_value` is a {py:class}`pandas.DataFrame` object with the scaled genetic values.
It is suggested to use this function on the genetic value dataframe that is obtained by
{py:func}`genetic_value`, and use the normalised genetic value dataframe to simulate phenotypes
by using {py:func}`sim_env`.

An example usage of this function is shown below:

```{code-cell}

mean = 0
var = 1
ts = msprime.sim_ancestry(
    samples=10_000,
    recombination_rate=1e-8,
    sequence_length=100_000,
    population_size=10_000,
    random_seed=1000,
)
ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=1001)
trait_df = tstrait.sim_trait(ts, num_causal=1000, model=model, random_seed=500)
genetic_df = tstrait.genetic_value(ts, trait_df)
normalised_df = tstrait.normalise_genetic_value(genetic_df, mean=mean, var=var)
normalised_df.head()
```

The distribution of the normalised genetic value is shown below, and wee that the mean and variance
of the normalised genetic values are 0 and 1.

```{code-cell}

import matplotlib.pyplot as plt
plt.hist(normalised_df["genetic_value"], bins=40)
plt.title("Normalised Genetic Value")
plt.show()
```
