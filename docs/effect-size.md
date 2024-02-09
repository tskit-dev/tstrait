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

(sim_trait)=

# Trait simulation

This page describes how tstrait simulates traits and how to specify the effect size
distribution.

**Learning Objectives**

After this effect size page, you will be able to:

- Understand the mathematical details of the phenotype model in tstrait
- Understand how to specify distributions of simulated effect sizes
- Understand how to simulate effect size in tstrait and read its output
- Understand how to simulate effect sizes from the frequency dependence architecture

(phenotype_model)=

## Phenotype Model

tstrait simulates a vector of quantitative trait $y$ from the following additive model,

```{math}
:label: eq:phenotype-model
y = X\beta+\epsilon,
```

where $X$ is the matrix that describes the number of causal alleles in each individual (the values
in each row will be $0$, $1$, or $2$ in the diploid setting, for example), $\beta$
is the vector of effect sizes, and $\epsilon$ is the vector of environmental noise. Environmental
noise is simulated from the following distribution,

```{math}
\epsilon\sim N\left(0,V_G\cdot\frac{1-h^2}{h^2} \right),
```

where $V_G=Var(X\beta)$ and $h^2$ is the narrow-sense heritability that is defined by the
user.

The genetic values ($X\beta$) are obtained by simply adding up over all the genomes in each individual,
regardless of ploidy.

:::{seealso}
- [](genetic_value_doc) for obtaining the genetic value $X\beta$.
- [](environment_noise) for simulating environmental noise $\epsilon$.
- [](numericalise_genotype) for modifying the numericalisation of genotypes.
:::

In this documentation, we will be describing how to simulate effect sizes in tstrait.

(effect_size_dist)=

## Effect Size Distribution

The first step of trait simulation is to specify the distribution where the effect sizes will be
simulated. It can be specified in `distribution` input of {py:func}`trait_model`. We also specify
other parameters of the distribution in the function as well. For example,

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

tstrait uses this distribution to simulate effect size $\beta$ in the phenotype model described
in [](eq:phenotype-model). 

The following effect size distributions are supported in tstrait, and please refer to links under
**Details** for details on the input and distribution.

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
     - ``value, random_sign``
     - :py:class:`TraitModelFixed`

   * - ``"exponential"``
     - Exponential distribution
     - ``scale, random_sign``
     - :py:class:`TraitModelExponential`

   * - ``"gamma"``
     - Gamma distribution
     - ``shape, scale, random_sign``
     - :py:class:`TraitModelGamma`

   * - ``"multi_normal"``
     - Multivariate normal distribution
     - ``mean, cov``
     - :py:class:`tstrait.TraitModelMultivariateNormal`
```

(effect_size_sim)=

## Effect Size Simulation

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
ts = msprime.sim_mutations(ts, rate=1e-6, random_seed=200)

trait_df = tstrait.sim_trait(ts, num_causal=5, model=model, random_seed=1)
trait_df
```

The trait dataframe has 6 columns:

> - **position**: Position of sites that have causal allele in genome coordinates.
> - **site_id**: Site IDs that have causal allele.
> - **effect_size**: Simulated effect size of causal allele.
> - **causal_allele**: Causal allele.
> - **allele_freq**: Allele frequency of causal allele.
> - **trait_id**: Trait ID that will be used in multi-trait simulation (See
[](multi_trait) for details).

Next, we will be showing the distribution of simulated effect sizes by using
a normal distribution trait model with 1,000 causal sites.

```{code-cell}

import matplotlib.pyplot as plt
model = tstrait.trait_model(distribution="normal", mean=0, var=1)
trait_df = tstrait.sim_trait(ts, num_causal=1000, model=model, random_seed=1)
trait_df.head()
plt.hist(trait_df["effect_size"], bins=40)
plt.title("Simulated Effect Size")
plt.show()
```

We see that the simulated effect sizes are approximately following a $N(0,1)$
distribution, which coincides with the fact that we had specified a normal
distribution trait model with mean 0 and variance 1.

:::{note}
The site ID represents the IDs of causal sites, and information regarding the site can be
extracted by using `` .site() ``.
:::

The below code is used to extract information of site with ID 0 from `ts` tree sequence.

```{code-cell}

# Extract information of site with ID 0
ts.site(0)
```

The details of sites in tree sequences can be found [here](tskit:sec_site_table_definition).


(trait_frequency_dependence)=

## Frequency Dependence

Tstrait supports frequency dependence simulation. It has been shown that rare variants
have increased effect sizes compared with common variants
[Speed et al. (2017)](https://doi.org/10.1038/ng.3865), so more realistic simulations
can be made possible by increasing the effect size on rarer variants. The `alpha`
parameter in {py:func}`sim_phenotype` and {py:func}`sim_trait` are used to control
the degree of frequency dependence on simulated effect sizes.

In the frequency dependence model, the following value is multiplied to the effect size:

```{math}
:label: eq:freq-dep
\Big[\sqrt{2p(1-p)}\Big]^\alpha
```

In the above expression, $p$ is the frequency of the causal allele, and
$\alpha$ is the `alpha` input of {py:func}`sim_phenotype` and
{py:func}`sim_trait`. Putting a negative $\alpha$ value increases the
magnitude of effect sizes on rare variants.

:::{note}
The default `alpha` parameter in {py:func}`sim_phenotype` and
{py:func}`sim_trait` are 0, and frequency dependent model is not used. Please
ignore the `alpha` parameter if you are not interested in implementing the
frequency dependent model.
:::

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
```

We will first simulate effect sizes by using the non-frequency dependent model
(`alpha` = 0), and visualize the simulated effect sizes. This is the default
`alpha` parameter in {py:func}`sim_trait`, so there is no need for you to specify
it in your trait simulation.

```{code-cell}

# trait.sim_trait(ts, num_causal=1000, model=model, random_seed=1)
# also works here
trait_df = tstrait.sim_trait(ts, num_causal=1000, model=model, 
                             alpha=0, random_seed=1)

plt.scatter(trait_df.allele_freq, trait_df.effect_size)
plt.xlabel("Allele frequency")
plt.ylabel("Effect size")
plt.axhline(y=0, color='r', linestyle='-')
plt.title("Non-frequency dependent model, alpha = 0")
plt.show()
```

We see no relationship between allele frequency and effect size. As a next example,
we will be simulating effect sizes with `alpha` = -1/2.

```{code-cell}

trait_df = tstrait.sim_trait(ts, num_causal=1000, model=model, 
                             alpha=-1/2, random_seed=1)

plt.scatter(trait_df.allele_freq, trait_df.effect_size)
plt.xlabel("Allele frequency")
plt.ylabel("Effect size")
plt.axhline(y=0, color='r', linestyle='-')
plt.title("Frequency dependent model, alpha = -1/2")
plt.show()
```

We see that rarer variants have increased effect sizes, as expected from
the mathematical expression given in [](eq:freq-dep).

(trait_causal_sites)=

## Specifying Causal Sites

Instead of specifying the number of causal sites, users can directly provide the
causal sites as an input of {py:func}`sim_phenotype` and {py:func}`sim_trait`.
When `causal_sites` argument of these functions are provided, tstrait will use
the inputted site IDs as causal sites, instead of randomly selecting causal
site IDs.

Example:

```{code-cell}

trait_df = tstrait.sim_trait(ts, causal_sites=[0, 3, 4], model=model, 
                             alpha=-1/2, random_seed=1)
trait_df
```