# Quickstart

This page provides an example of how to use **tstrait** to simulate quantitative traits of individuals in the tree sequence data. We will be using **msprime** to simulate tree sequence data, and users who are interested in the details of **msprime** should consult [msprime manual](https://tskit.dev/msprime/docs/stable/intro.html).

## Trait Model

**tstrait** supports simulation of quantitative traits assuming that they are all additive. As a beginning, we need to specify the trait model to determine how effect sizes of genes are being simulated. The details of the model are indicated in [trait model](model.md). Currently, two trait models are supported in **tstrait**:

- TraitModelAdditive
- TraitModelAllele

It would be necessary for the user to set the mean and standard deviation of the genetic effect sizes when they define the trait model. For example,

```Python
import tstrait
model = tstrait.TraitModelAdditive(trait_mean = 0, trait_sd = 1)
```

sets a trait model, where the mean value of traits is 0 and the standard deviation is 1. We will then be using this `model` in `sim_phenotype()` to simulate quantitative traits of individuals in the tree sequence data.

## Example

In the following example, we will be using **msprime** to simulate a tree sequence dataset, and simulate quantitative traits of simulated individuals.

```Python
import msprime
import tstrait

num_ind = 100
ts = msprime.sim_ancestry(num_ind, sequence_length=100_000, recombination_rate=1e-8,population_size=10**4, random_seed=1)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)

model = tstrait.TraitModelAdditive(trait_mean = 0, trait_sd = 1)
phenotype_result, genetic_result = tstrait.sim_phenotype(ts,num_causal = 5, model = model, h2 = 0.3, random_seed = 1)
```

The example above simulates tree sequence data with 100 individuals and then simulates quantitative traits of those individuals with 5 causal sites. The narrow sense heritability `h2` is used to determine the environmental noise.

`phenotype_result` and `genetic_result` above are dataclass objects, and their information can be extracted as following:

```Python
# Individual ID
phenotype_result.individual_id
# Phenotype value
phenotype_result.phenotype
# Environmental noise
phenotype_result.environment_noise
# Genetic value
phenotype_result.genetic_value
```

```Python
# Site ID
genetic_result.site_id
# Causal allele
genetic_result.causal_allele
# Effect size
genetic_result.effect_size
# Causal allele frequency
genetic_result.allele_frequency
```
Detailed information regarding the site and individual, such as ancestral state, genomic position and relatedness of individuals, can be obtained by using [tskit](https://tskit.dev/tskit/docs/stable/introduction.html).