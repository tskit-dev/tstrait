---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Output

The outputs of the {func}`.sim_phenotype` function are two dataclass objects. The first output is a {class}`.PhenotypeResult` object, which includes information regarding the simulated individuals. It includes the individual ID, simulated value of phenotype, genetic value and environmental noise. The second output is a {class}`.GeneticValueResult` object, and it includes the site ID, causal allele, effect size and causal allele frequency.

To ensure that the output of the example is the same, we set a `random_seed`. In the below example, we will be simulating 5 individuals with 3 causal sites from {class}`.TraitModelAllele` model. The other simulation parameters are set to be the same as the example in the [Quickstart](quickstart.md) page.

```{code-cell} ipython3
import msprime
import tstrait

num_ind = 5
ts = msprime.sim_ancestry(num_ind, sequence_length=1_000_000, recombination_rate=1e-8,
                        population_size=10**4, random_seed=1)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)

model = tstrait.TraitModelAllele(trait_mean = 0, trait_sd = 1, alpha = -0.3)
phenotype_result, genetic_result = tstrait.sim_phenotype(ts,num_causal = 3, model = model,
                                                            h2 = 0.3, random_seed = 1)
```

We will show how we can extract information from `phenotype_result`. It is an object of {class}`.PhenotypeResult` and it includes information regarding the simulated individuals.

```{code-cell} ipython3
print("Individual ID:", phenotype_result.individual_id)
print("Phenotype:", phenotype_result.phenotype)
print("Environmental Noise:", phenotype_result.environment_noise)
print("Genetic Value:", phenotype_result.genetic_value)
```

In the above output, phenotype, environmental noise and genetic value are aligned based on individual IDs. Since we assume an additive model, phenotype is the sum of environmental noise and genetic value.

We will next show how we can extract information from `genetic_result`. It is an object of {class}`.GeneticValueResult` and it includes information regarding the causal sites inside the simulation model.

```{code-cell} ipython3
print("Causal Site ID:", genetic_result.site_id)
print("Causal Allele:", genetic_result.causal_allele)
print("Effect Size:", genetic_result.effect_size)
print("Allele Frequency:", genetic_result.allele_frequency)
```

In the above output, causal allele, effect size and allele frequency are aligned based on causal site IDs. Causal allele is randomly selected among the mutations inside the side, and the allele frequency represents the frequency of the causal allele inside the site.

Detailed information regarding the site and individual, such as ancestral state, genomic position and relatedness of individuals, can be obtained by using {ref}`tskit <tskit>` package.