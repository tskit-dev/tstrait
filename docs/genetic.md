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
- Understand the details of frequency dependence model that is supported in tstrait

# Algorithm Overview

Genetic value is computed in tstrait by using the trait information in the input trait dataframe.
It uses a tree traversal algorithm to count the number of causal allele in each individual and adds
the corresponding effect size to individual's genetic value depending on the presence of causal
mutation in that individual.

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
