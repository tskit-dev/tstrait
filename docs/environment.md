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

(environment_noise)=

# Environmental noise

This page describes how environmental noise is simulated in tstrait. Please refer to
{ref}`phenotype_model` for mathematical details on the phenotype model.

**Learning Objectives**

After this effect size page, you will be able to:

- Understand how to simulate environmental noise in tstrati
- Understand how to use the user's defined distribution to simulate environmental noise

## Environmental noise simulation

Environmental noise can be simulated by using {py:func}`sim_env`. The required inputs are

genetic_df

: Genetic value dataframe. Please see {ref}`req_genetic_df` for requirements.

h2

: Narrow-sense heritability.

(req_genetic_df)=

### Dataframe requirements

The simplest way to simulate environmental noise is by using the genetic value dataframe
output of {py:func}`genetic_value`. If you would like to define your own genetic value
dataframe, there are some requirements that you must follow.

#### Columns

The following columns must be included in `genetic_df`:

> - **trait_id**: Trait ID. This will be used in multi-trait simulation.
> - **individual_id**: Individual ID.
> - **genetic_value**: Simulated genetic value.

#### Data requirement

Trait IDs in **trait_id** column must start from 0 and be consecutive. If you are simulating
a single trait, you should be using an array that is filled with zeros.

## Example

We will be showing an example of environmental noise simulating by using a simulated tree
sequence data with 10,000 individuals. The narrow-sense heritability is set to be 0.3.

:::{seealso}
- [msprime](msprime:sec_intro) for simulating whole genome in tree sequence data.
- [](sim_trait_doc) for simulating trait dataframe in tstrait.
- [](genetic_value_doc) for simulating the genetic value dataframe in tstrait.
:::

```{code-cell}

import msprime
import tstrait

ts = msprime.sim_ancestry(
    samples=10_000,
    recombination_rate=1e-8,
    sequence_length=1_000_000,
    population_size=10_000,
    random_seed=5,
)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=5)

model = tstrait.trait_model(distribution="normal", mean=0, var=1)
trait_df = tstrait.sim_trait(ts, num_causal=1000, model=model, random_seed=5)
genetic_df = tstrait.genetic_value(ts, trait_df)

phenotype_df = tstrait.sim_env(genetic_df, h2=0.3, random_seed=5)
phenotype_df.head()
```

The resulting dataframe has the following columns:

> - **trait_id**: Trait ID.
> - **individual_id**: Individual ID inside the tree sequence input.
> - **genetic_value**: Simulated genetic values.
> - **environmental_noise**: Simulated environmental noise.
> - **phenotype**: Simulated phenotype.

The distribution of simulated environmental noise is shown below.

```{code-cell}

import matplotlib.pyplot as plt

plt.hist(phenotype_df["environmental_noise"], bins=40)
plt.title("Environmental Noise")
plt.show()
```

The simulated environmental noise is following a normal distribution as expected.

## User-defined environmental noise

It would be possible for the user to define their own environmental noise, and
there are several options available for the user.

### Simulating from the output of {py:func}`genetic_value`

The output of {py:func}`genetic_value` only includes relevant information regarding
genetic values, and it doesn't simulate environmental noise. For example, if the
user wants to simulate environmental noise from a normal distribution with mean 0
and variance 1, it would be possible to run the following code:

```{code-cell}

import numpy as np

genetic_df = tstrait.genetic_value(ts, trait_df)

rng = np.random.default_rng(seed=50)
env_noise = rng.normal(loc=0, scale=1, size=len(genetic_df))
genetic_df["environmental_noise"] = env_noise
genetic_df["phenotype"] = (
    genetic_df["environmental_noise"] + genetic_df["genetic_value"]
)
genetic_df.head()
```

We will be drawing random samples from a normal distribuion by using
{py:meth}`numpy.random.Generator.normal`.

### Setting `h2` to be 1

When `h2` is set to be 1 in {py:func}`sim_phenotype` or {py:func}`sim_env`, the
environmental noise will be a vector of zeros. After obtaining the output, the user
can define their own environmental noise.

```{code-cell}

sim_result = tstrait.sim_phenotype(
    ts=ts, num_causal=100, model=model, h2=1, random_seed=1
)
sim_result.phenotype.head()
```

We see that all values in the `environmental_noise` column are zero.