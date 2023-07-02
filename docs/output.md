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

(sec_output)=

# Output

The outputs of the {func}`.sim_phenotype` function are two dataclass objects. The first output is a {class}`.PhenotypeResult` object, which includes simulated information regarding the individuals. The second output is a {class}`.GeneticValueResult` object, which includes simulated information regarding the causal sites.

In the below example, we will be simulating 5 individuals with 3 causal sites from {class}`.TraitModelAlleleFrequency` model. The other simulation parameters are set to be the same as the example in the [Quickstart](quickstart.md) page.

```{code-cell} ipython3
import msprime
import tstrait

num_ind = 5
ts = msprime.sim_ancestry(num_ind, sequence_length=1_000_000, recombination_rate=1e-8,
                          population_size=10**4, random_seed=1)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)

model = tstrait.TraitModelAlleleFrequency(trait_mean=0, trait_sd=1, alpha=-0.3)
phenotype_result, genetic_result = tstrait.sim_phenotype(ts, num_causal=3, model=model,
                                                         h2=0.3, random_seed=1)
```

(sec_output_phenotype)=

## Individual

In the below code, we extract information from `phenotype_result`, which is a {class}`.PhenotypeResult` object that includes information regarding the individuals.

```{code-cell} ipython3
print("Individual ID:", phenotype_result.individual_id)
print("Phenotype:", phenotype_result.phenotype)
print("Environmental Noise:", phenotype_result.environment_noise)
print("Genetic Value:", phenotype_result.genetic_value)
```

In the above output, phenotype, environmental noise and genetic value are [numpy.ndarray](https://numpy.org/doc/stable/reference/arrays.ndarray.html#arrays-ndarray) objects. Their length is the number of individuals in the input tree sequence data, and the elements are aligned based on individual IDs. The $i$th entry of each array represents the value associated with the $i$th individual. Further information regarding the individuals can be accessed by using the {ref}`Individual Table<tskit:sec_individual_table_definition>` in the tree sequence data through the individual IDs in `phenotype_result.individual_id`.

For example, information regarding the first individual in the output can be obtained as following,

```{code-cell} ipython3
ts.individual(phenotype_result.individual_id[0])
```

(sec_output_genetic)=

## Causal Site

Next, we extract information from `genetic_result`, which is a {class}`.GeneticValueResult` object that includes information regarding the causal sites.

```{code-cell} ipython3
print("Causal Site ID:", genetic_result.site_id)
print("Causal Allele:", genetic_result.causal_allele)
print("Effect Size:", genetic_result.effect_size)
print("Allele Frequency:", genetic_result.allele_frequency)
```

In the above output, causal allele, effect size and allele frequency are [numpy.ndarray](https://numpy.org/doc/stable/reference/arrays.ndarray.html#arrays-ndarray) objects. Their length is the number of causal sites in the simulation model, and the elements are aligned based on causal site IDs. The $i$th entry of each array represents the value associated with the $i$th causal site. Causal allele is randomly selected among the mutations in the causal side, and the allele frequency represents the frequency of the causal allele. Further information regarding the causal sites can be accessed by using the {ref}`Site Table<tskit:sec_site_table_definition>` in the tree sequence data through the site IDs in `genetic_result.site_id`.

For example, information regarding the first causal site in the output can be obtained as following,

```{code-cell} ipython3
ts.site(genetic_result.site_id[0])
```