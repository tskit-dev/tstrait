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

(genetic_value)=

# Genetic value

This page will describe how to simulate genetic values by using {py:func}`sim_genetic` function. Genetic
values of individuals are described by $X\beta$, where $X$ is the matrix that describes
the number of causal alleles in each individual and $\beta$ is the vector of effect sizes. See
{ref}`phenotype_model` for mathematical details on the quantitative trait model.

**Learning Objectives**

After this genetic value page, you will be able to:

- Understand how to simulate genetic value in tstrait
- Understand how to use the user's defined effect sizes to simulate genetic values
- Understand the details of frequency dependence model that is supported in tstrait

## Input

The input of {py:func}`sim_genetic` function is the following:

ts

: Tree sequence input.

trait_df

: Trait dataframe that is described in [](effect_size_sim). There are some requirements for the trait
  dataframe input, which are described in [](req_trait_df).

alpha

: Parameter of frequency dependence model. The default value is 0, and the details are described
  in [](frequency_dependence).

The details of the parameters and how they influence the genetic value simulation are described in detail
below.

## Genetic value simulation

In this example, we will be showing how to simulate genetic values by using the simulated trait dataframe
in {py:func}`sim_trait`.

We will be simulating a sample tree sequence data with 5 individuals by using [msprime](msprime:sec_intro),
and we will simulate a trait dataframe with 3 causal sites. The dataframe output of {py:func}`sim_trait`
can be automatically used as an input of {py:func}`sim_genetic`, so there is no need to worry about input
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

Next, we will be using the trait dataframe to simulate genetic values by using {py:func}`sim_trait`.

```{code-cell}

    genetic_result = tstrait.sim_genetic(ts, trait_df, random_seed=500)
```

The output of {py:func}`sim_genetic` is a {py:class}`GeneticResult` object containing two dataframes
to represent simulated effect sizes. we will now be showing the details of the output.

(effect_size_genetic)=

## Effect size output

The effect size dataframe can be extracted from the output of {py:func}`sim_genetic` by using
`.effect_size`.

```{code-cell}

    genetic_result.effect_size
```

The effect size dataframe output is very similar as the output of {py:func}`sim_trait`,
but it has two additional columns:

> - **causal_state**: Causal state of the causal mutation
> - **allele_frequency**: Allele frequency of the causal mutation

:::{note}
Mutations that have a different derived state as the causal state will not have any effect on
genetic values of individuals, even if it is located in the causal site.
:::

The effect sizes in **effect_size** column might change compared with the output of
{py:func}`sim_trait` depending on the frequency dependence architecture. Please refer to
[](frequency_dependence) for details.

(genetic_value_output)=

## Genetic value output

Genetic value dataframe can be extracted from the output of {py:func}`sim_genetic` by using
`.genetic`.

```{code-cell}

    genetic_result.genetic
```

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

> - **site_id**: Site IDs that have causal mutation.
> - **effect_size**: Simulated effect size of causal mutation.
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

    data = {"site_id": [0], "effect_size": [1], "trait_id": [0]}
    trait_df = pd.DataFrame(data)
    trait_df
```

```{code-cell}

    genetic_result = tstrait.sim_genetic(ts, trait_df, random_seed=500)
    genetic_result.effect_size
    genetic_result.genetic
```

As seen in the genetic value dataframe, the genetic values of individuals are multiples
of 1. This is because 1 is the only effect size in the effect size dataframe, and the
{py:func}`sim_genetic` function is counting the number of causal mutations of site 0
inside the individual.

Next, we will be inputting an effect size dataframe with multiple mutations.

```{code-cell}

    data = {"site_id": [0, 2, 4], "effect_size": [1, 10, 100], "trait_id": [0, 0, 0]}
    trait_df = pd.DataFrame(data)
    trait_df = trait_df.sort_values(by=["site_id"])
    trait_df
```

The site IDs in `"site_id"` column in the trait dataframe must be sorted in an
ascending order, so it is suggested to use `.sort_values(by=['site_id'])`
code before inputting it inside the {py:func}`sim_genetic` function.

```{code-cell}

    genetic_result = tstrait.sim_genetic(ts, trait_df, random_seed=500)
    genetic_result.effect_size
    genetic_result.genetic
```

(frequency_dependence)=

## Frequency dependence

Tstrait supports frequency dependence simulation. It has been shown that rare variants
have increased effect sizes compared with common variants
[Speed et al. (2017)](https://doi.org/10.1038/ng.3865), so more realistic simulations
can be made possible by increasing the effect size on rarer variants. The `alpha`
parameter in {py:func}`sim_phenotype` and {py:func}`sim_genetic` are used to control
the degree of frequency dependence on simulated effect sizes.

In the frequency dependence model, the following value is multiplied to the effect size:

$$
[2p(1-p)]^\alpha
$$

In the above expression, $p$ is the frequency of the causal mutation, and
$\alpha$ is the `alpha` input of {py:func}`sim_phenotype` and
{py:func}`sim_genetic`. Putting a negative $\alpha$ value increases the
magnitude of effect sizes on rare variants.

:::{note}
The default `alpha` parameter in {py:func}`sim_phenotype` and
{py:func}`sim_genetic` are 0, and frequency dependent model is not used. Please
ignore the `alpha` parameter if you are not interested in implementing the
frequency dependent model.
:::

The frequency dependence architecture is still an ongoing research topic. While the
frequency dependence model can be used for any trait models in tstrait, it is
suggested that you use the normal distribution with mean 0 as a trait model and
`alpha` to be -1/2 to conduct simulations that are widely used in
simulation-based research projects (See
[Speed et al. (2017)](https://doi.org/10.1038/ng.3865) for details).

In the below example, we will be demonstrating how `alpha` influences the simulated
effect sizes by using a simulated tree sequence with 10,000 individuals.

```{code-cell}

    ts = msprime.sim_ancestry(
        samples=10_000,
        recombination_rate=1e-8,
        sequence_length=1_000_000,
        population_size=10_000,
        random_seed=300,
    )
    ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=303)

    model = tstrait.trait_model(distribution="normal", mean=0, var=1)
    trait_df = tstrait.sim_trait(ts, num_causal=1000, model=model, random_seed=100)
```

We will first simulate effect sizes by using the non-frequency dependent model
(`alpha` = 0).

```{code-cell}

    import matplotlib.pyplot as plt

    non_freq_genetic_result = tstrait.sim_genetic(ts, trait_df, random_seed=500)
    effect_size_df = non_freq_genetic_result.effect_size

    plt.scatter(effect_size_df.allele_frequency, effect_size_df.effect_size)
    plt.xlabel("Allele frequency")
    plt.ylabel("Effect size")
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title("Non-frequency dependent model, alpha = 0")
    plt.show()
```

We see no relationship between allele frequency and effect size. As a next example,
we will be simulating effect sizes with `alpha` = -1/2.

```{code-cell}

    freq_genetic_result = tstrait.sim_genetic(ts, trait_df, alpha=-1/2, random_seed=500)
    effect_size_df = freq_genetic_result.effect_size

    plt.scatter(effect_size_df.allele_frequency, effect_size_df.effect_size);
    plt.xlabel("Allele frequency")
    plt.ylabel("Effect size")
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title("Frequency dependent model, alpha = -1/2")
    plt.show()
```

We see that rarer variants have increased effect sizes.

The effect sizes in effect size dataframe of the {py:func}`sim_genetic` function
is given by multiplying the constant in the frequency dependent model. Thus, we see
that the effect sizes in the trait dataframe output of {py:func}`sim_trait` is
different compared with the effect sizes in the output of {py:func}`sim_genetic`
function.

The effect size output of the non-frequency dependent model is the same as the effect
size output of {py:func}`sim_trait`:

```{code-cell}

    trait_df.head()
    non_freq_genetic_result.effect_size.head()
```

The effect size output of the frequency dependent model is different than the effect
size output of {py:func}`sim_trait`:

```{code-cell}

    trait_df.head()
    freq_genetic_result.effect_size.head()
```

:::{note}
The effect size output of {py:func}`sim_genetic` is used in simulating
the genetic values, and not the `trait_df` input in the frequency dependent
model.
:::
