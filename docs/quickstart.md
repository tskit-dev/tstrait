# Quickstart

This page provides an example of how to use **tstrait** to simulate quantitative traits of individuals in the tree sequence data. We will be using **msprime** to simulate tree sequence data, and users who are interested in the details of **msprime** should consult [msprime manual](https://tskit.dev/msprime/docs/stable/intro.html).

See the [Installation](installation.md) page for instructions on installing **tstrait**.

## Trait Model

**tstrait** supports simulation of quantitative traits assuming that they are all additive. As a beginning, we need to specify the trait model to determine how effect sizes of genes are being simulated. The details of the model are indicated in [trait model](model.md). Currently, two trait models are supported in **tstrait**:

- TraitModelAdditive
- TraitModelAllele

It would be necessary for the user to set the mean and standard deviation of the genetic effect sizes when they define the trait model. For example,

```Python
import tstrait
model = tstrait.TraitModelAllele(trait_mean = 0, trait_sd = 1, alpha = -0.3)
```

sets a trait model {class}`.TraitModelAllele`, where the mean value of traits is 0 and the standard deviation is 1. We will then be using this `model` in {func}`.sim_phenotype` to simulate quantitative traits of individuals in the tree sequence data.

## Example

In the following example, we will be using **msprime** to simulate a tree sequence dataset, and simulate quantitative traits of simulated individuals.

The [Output](output.md) page has a detailed explanation on the output of **tstrait**.

```{eval-rst}
.. jupyter-execute::

    import msprime
    import tstrait

    num_ind = 500
    ts = msprime.sim_ancestry(num_ind, sequence_length=100_000, recombination_rate=1e-8,
                            population_size=10**4, random_seed=1)
    ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)

    model = tstrait.TraitModelAdditive(trait_mean = 0, trait_sd = 1)
    phenotype_result, genetic_result = tstrait.sim_phenotype(ts,num_causal = 1000, model = model,
                                                             h2 = 0.3, random_seed = 1)
```

The example above simulates tree sequence data with 500 individuals and then simulates quantitative traits of those individuals with 1000 causal sites. The narrow sense heritability `h2` is used to determine the environmental noise. We also set the `random_seed` to ensure that the output of the simulation model is the same.

The distribution of simulated phenotype of 500 individuals is shown below.





`phenotype_result` and `genetic_result` above are dataclass objects, and their information can be extracted as following:

```{eval-rst}
.. jupyter-execute::

    # Individual ID
    phenotype_result.individual_id
```

```{eval-rst}
.. jupyter-execute::

    # Phenotype value
    phenotype_result.phenotype
```

```{eval-rst}
.. jupyter-execute::

    # Environmental noise
    phenotype_result.environment_noise
```

```{eval-rst}
.. jupyter-execute::

    # Genetic value
    phenotype_result.genetic_value
```

````{eval-rst}
.. jupyter-execute::

    # Site ID
    print(genetic_result.site_id)
    # Causal allele
    print(genetic_result.causal_allele)
    # Effect size
    print(genetic_result.effect_size)
    # Causal allele frequency
    print(genetic_result.allele_frequency)
```

Detailed information regarding the site and individual, such as ancestral state, genomic position and relatedness of individuals, can be obtained by using [tskit](https://tskit.dev/tskit/docs/stable/introduction.html).