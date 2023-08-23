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

(effect_size)=

# Effect size

This page describes how tstrait simulates effect sizes and how to specify effect size
distributions in tstrait.

**Learning Objectives**

After this effect size page, you will be able to:

- Understand the mathematical details of the phenotype model in tstrait
- Understand how to specify distributions of simulated effect sizes
- Understand how to simulate effect size in tstrait and read its output

(phenotype_model)=

## Phenotype Model

Tstrait simulates a vector of quantitative trait $y$ from the following additive model,

$$
y = X\beta+\epsilon
$$

where $X$ is the matrix that describes the number of causal alleles in each individual, $\beta$
is the vector of effect sizes, and $\epsilon$ is the vector of environmental noise. Environmental
noise is simulated from the following distribution,

$$
\epsilon\sim N\left(0,V_G\cdot\frac{1-h^2}{h^2} \right),
$$

where $V_G=Var(X\beta)$ and $h^2$ is the narrow-sense heritability that is defined by the
user.

:::{seealso}
- [](genetic_value) for simulating the genetic value $X\beta$.
- [](environment_noise) for simulating environmental noise $\epsilon$.
:::

In this documentation, we will be describing how to simulate effect sizes in tstrait.

(effect_size_dist)=

## Effect size distribution

The first step of effect size simulation is to specify the effect size distribution, which can be specified in
`distribution` input of {py:func}`trait_model`. We also specify other parameters of the distribution
in the function as well. For example,

```{code-cell}

   import tstrait

   model = tstrait.trait_model(distribution="normal", mean=0, var=1)
```

sets a trait model, where the effect sizes are simulated from a normal distribution with
mean $0$ and variance $1$. We can check the distribution name by using `.name` attribute
of a model instance.

```{code-cell}

   model.name
```

The following effect size distributions are supported in tstrait, and please refer to **Details** for details on
the input and distribution.

:::{seealso}
[](effect_size_distribution) for details on the supported distributions.
:::

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - Name
     - Distribution
     - Input
     - Details

   * - ``"normal"``
     - Normal distribution
     - ``mean, var``
     - :py:class:`TraitModelNormal`

   * - ``"t"``
     - Student's t distribution
     - ``mean, var, df``
     - :py:class:`TraitModelT`

   * - ``"fixed"``
     - Fixed value
     - ``value``
     - :py:class:`TraitModelFixed`

   * - ``"exponential"``
     - Exponential distribution
     - ``scale, negative``
     - :py:class:`TraitModelExponential`

   * - ``"gamma"``
     - Gamma distribution
     - ``shape, scale, negative``
     - :py:class:`TraitModelGamma`

   * - ``"multi_normal"``
     - Multivariate normal distribution
     - ``mean, cov``
     - :py:class:`tstrait.TraitModelMultivariateNormal`
```

(effect_size_sim)=

## Effect size simulation

Effect sizes can be simulated in tstrait by using {py:func}`tstrait.sim_trait`. In the example below,
we will be simulating effect sizes of 5 causal sites from a simulated tree sequence data in
[msprime](msprime:sec_intro).

```{code-cell}

  import msprime

  ts = msprime.sim_ancestry(
      samples=10_000,
      recombination_rate=1e-8,
      sequence_length=100_000,
      population_size=10_000,
      random_seed=200,
  )
  ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=200)

  trait_df = tstrait.sim_trait(ts, num_causal=5, model=model, random_seed=1)
  trait_df
```

The trait dataframe has 3 columns:

> - **site_id**: Site IDs that have causal mutation.
> - **effect_size**: Simulated effect size of causal mutation.
> - **trait_id**: Trait ID.

:::{note}
The simulated effect sizes will be divided by the number of causal sites, such that the overall
trait variance does not explode. Please keep this in mind while selecting the parameters of the
trait model.
:::

This division can be illustrated by using a fixed value trait model.

```{code-cell}

  model = tstrait.trait_model(distribution="fixed", value=1)
  trait_df = tstrait.sim_trait(ts, num_causal=1, model=model, random_seed=1)
  trait_df
```

The effect size will be 1.0, as there is 1 causal site. When the number of causal sites is
selected to be 4, the effect size will be 1/4=0.25, even though it is using the same
trait model.

```{code-cell}

  trait_df = tstrait.sim_trait(ts, num_causal=4, model=model, random_seed=1)
  trait_df
```

The {py:func}`sim_trait` function simulates effect sizes from the distribution specified in the
`model` input and divides it by the number of causal sites. The site ID represents the IDs of causal
sites, and information regarding the site can be extracted by using `` .site() ``.

```{code-cell}

  # Extract information of site with ID 0
  ts.site(0)
```

The details of sites in tree sequences can be found [here](tskit:sec_site_table_definition).

The trait ID column is used for multi-trait simulation, which is described in [](multi_trait).
