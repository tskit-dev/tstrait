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

(sec_simulation)=

# Running Simulation

This model describes how simulations are performed by using **tstrait**. The definitions of some key terms in this documentation are indicated {ref}`here <tskit:sec_glossary>`.

## Phenotype Model

**tstrait** assumes that the individual's phenotypes are obtained from the following additive model,

$$
y=X\beta+\epsilon.
$$

In the above equation, $y$ is the vector of phenotypes, $X$ is the matrix that denotes the number of causal alleles inside the individual, $\beta$ is the vector of effect sizes, and $\epsilon$ is the vector of environmental noise.

The simulation model initially chooses $m$ causal sites at random among the sites in the tree sequence data, and causal allele for each site is chosen at random among the mutations in the site. Ancestral state is not chosen to be the causal allele. For each causal site $i$, **tstrait** calculates the causal allele frequency $p_i$ and simulates the effect size $\beta_i$ as indicated in [Trait Model](model.md) page.

After the genetic value, $G=X\beta$, is simulated from the tree sequence data, the environmental noise $\epsilon$ is simulated from

$$
\epsilon_j\sim N\left(0,Var(G)\cdot\frac{(1-h^2)}{h^2}\right),
$$

where $Var(G)$ is the variance of the simulated genetic values and $h^2$ is the narrow-sense heritability which is defined by the user. The phenotypic values of individuals are determined by adding the genetic values and environmental noise.

It is possible for the user to set $h^2=0$ or $h^2=1$. When $h^2=0$, the phenotype will be exactly the same as the environmental noise, and when $h^2=1$, the environmental noise $\epsilon$ will be a vector of zeros.

The number of causal sites $m$ and the narrow-sense heritability $h^2$ of the simulation model are specified in `num_causal` and `h2` arguments in {func}`.sim_phenotype` function. For example,

```Python
import tstrait
tstrait.sim_phenotype(ts, num_causal=1000, h2=0.3, model=model)
```

simulates quantitative traits of individuals in `ts` tree sequence data from `model` trait model with 1000 causal sites and narrow-sense heritability being 0.3.

(sec_output)=

## Output

We will now be showing the output of **tstrait**. Simulation results in **tstrait** are given by using the {class}`.Result` object. It includes two dataclasses:

- {class}`.PhenotypeResult`: Includes simulated phenotypic information of individuals
- {class}`.GenotypeResult`: Includes simulated genotypic information of causal sites

To show how the output can be obtained, we will be simulating 5 individuals with 3 causal sites from {class}`.TraitModelAlleleFrequency` model. The other simulation parameters are set to be the same as the example in the [Quickstart](quickstart.md) page.

```{code-cell} ipython3
import msprime
import tstrait

num_ind = 5
ts = msprime.sim_ancestry(num_ind, sequence_length=1_000_000, recombination_rate=1e-8,
                          population_size=10**4, random_seed=1)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)

model = tstrait.TraitModelAlleleFrequency(trait_mean=0, trait_sd=1, alpha=-0.3)
sim_result = tstrait.sim_phenotype(ts, num_causal=3, model=model, h2=0.3, random_seed=1)
```

(sec_output_phenotype)=

### Phenotype

In the below code, we extract information from `sim_result.phenotype`, which is a {class}`.PhenotypeResult` object that includes information regarding the individuals.

```{code-cell} ipython3
print(sim_result.phenotype)
```

In the above output, phenotype, environmental noise and genetic value are [numpy.ndarray](https://numpy.org/doc/stable/reference/arrays.ndarray.html#arrays-ndarray) objects. Their length is the number of individuals in the input tree sequence data, and the elements are aligned based on individual IDs. The $i$th entry of each array represents the value associated with the $i$th individual. We can obtain the array of each element as in the following output:

```{code-cell} ipython3
phenotype_result = sim_result.phenotype
print(phenotype_result.individual_id)
```

Further information regarding the individuals can be accessed by using the {ref}`Individual Table<tskit:sec_individual_table_definition>` in the tree sequence data through the individual IDs in `phenotype_result.individual_id`.

For example, information regarding the first individual in the output can be obtained as following,

```{code-cell} ipython3
ts.individual(phenotype_result.individual_id[0])
```

(sec_output_genotype)=

### Genotype

Next, we extract information from `sim_result.genotype`, which is a {class}`.GenotypeResult` object that includes information regarding the causal sites.

```{code-cell} ipython3
print(sim_result.genotype)
```

In the above output, causal allele, effect size and allele frequency are [numpy.ndarray](https://numpy.org/doc/stable/reference/arrays.ndarray.html#arrays-ndarray) objects. Their length is the number of causal sites in the simulation model, and the elements are aligned based on causal site IDs. The $i$th entry of each array represents the value associated with the $i$th causal site. Causal allele is randomly selected among the mutations in the causal side, and the allele frequency represents the frequency of the causal allele. We can obtain the array of each element as in the following output:

```{code-cell} ipython3
genotype_result = sim_result.genotype
print(sim_result.genotype.site_id)
```


Further information regarding the causal sites can be accessed by using the {ref}`Site Table<tskit:sec_site_table_definition>` in the tree sequence data through the site IDs in `genetic_result.site_id`.

For example, information regarding the first causal site in the output can be obtained as following,

```{code-cell} ipython3
ts.site(genotype_result.site_id[0])
```

## Heritability

Narrow-sense heritability in **tstrait** controls the variance of the simulated environmental noise. In the below example, we will be simulating quantitative traits by using the simulated tree sequence data to show how narrow-sense heritability influences the relationship between genetic values and phenotype.

```{code-cell} ipython3
import msprime
import tstrait
import matplotlib.pyplot as plt

num_ind = 500
ts = msprime.sim_ancestry(num_ind, sequence_length=1_000_000, recombination_rate=1e-8,
                          population_size=10**4, random_seed=1)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)
model = tstrait.TraitModelAdditive(trait_mean=0, trait_sd=1)
```

### Example with $h^2=0.1$

When narrow-sense heritability is set to be a low number, most of phenotypic variation is coming from environmental variance.

```{code-cell} ipython3
sim_result = tstrait.sim_phenotype(ts, num_causal=1000, model=model, h2=0.1, random_seed=1)
phenotype_result = sim_result.phenotype

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Narrow-sense heritability = 0.1')
ax1.hist(phenotype_result.environment_noise, bins=20)
ax1.set_xlabel("Environmental Noise")
ax1.set_xlim([-4, 4])
ax2.hist(phenotype_result.phenotype, bins=20)
ax2.set_xlabel("Phenotype")
ax2.set_xlim([-4, 4])
plt.show()
```

### Example with $h^2=0.9$

When narrow-sense heritability is set to a high number, most of phenotypic variation is coming from additive genetic values, and environmental variance is smaller compared with phenotypic variance.

```{code-cell} ipython3
sim_result = tstrait.sim_phenotype(ts, num_causal=1000, model=model, h2=0.9, random_seed=1)
phenotype_result = sim_result.phenotype

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Narrow-sense heritability = 0.9')
ax1.hist(phenotype_result.environment_noise, bins=20)
ax1.set_xlabel("Environmental Noise")
ax1.set_xlim([-1.5, 1.5])
ax2.hist(phenotype_result.phenotype, bins=20)
ax2.set_xlabel("Phenotype")
ax2.set_xlim([-1.5, 1.5])
plt.show()
```