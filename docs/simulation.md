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

# Running Simulations

This page describes how simulations are performed by using **tstrait**. We refer to the tskit {ref}`docs <tskit:sec_glossary>` for the definition of some key terms in this documentation.

## Phenotype Model

**tstrait** assumes that the vector describing the phenotypes of all individuals can be described by the following additive model,

$$
y=X\beta+\epsilon.
$$

In the above equation, $y$ is the phenotype vector, $X$ is the matrix that denotes the number of the causal alleles for each individual, $\beta$ is the effect size vector, and $\epsilon$ is the environmental noise vector.

Depending on the specified number of causal sites, $m$ random sites are picked among those in the tree sequence. A causal allele for each causal site is then picked at random anong the non-ancestral alleles present at that site. For each causal site $i$, **tstrait** simulates an effect size $\beta_i$ based on the specified trait model. Details on the different trait models and how to specify them can be found here: {ref}`sec_trait_model`.

In the last step, environmental noise $\epsilon$ is added to the genetic component $G=X\beta$. These values are drawn from a normal distribiution:

$$
\epsilon_j\sim N\left(0,Var(G)\cdot\frac{(1-h^2)}{h^2}\right),
$$

where $Var(G)$ is the variance of the simulated genetic values and $h^2$ is the narrow-sense heritability which is defined by the user. The phenotypic values of individuals are determined by adding the genetic values and the environmental noise.

Heritability can be set to $h^2=0$ or $h^2=1$. When $h^2=0$, the phenotype will be exactly the same as the environmental noise, and when $h^2=1$, the environmental noise $\epsilon$ will be a vector of zeros.

The number of causal sites $m$ and the narrow-sense heritability $h^2$ are specified by the `num_causal` and `h2` arguments in {func}`.sim_phenotype` function. For example,

```Python
import tstrait
tstrait.sim_phenotype(ts, num_causal=1000, h2=0.3, model=model)
```

The relationship between narrow-sense heritability and the simulation result is shown here: {ref}`sec_simulation_heritability`.

(sec_simulation_output)=

## Output

**tstrait** returns a {class}`.Result` object. This object consists of two dataclasses:

- {class}`.PhenotypeResult`: Includes simulated phenotypic information of individuals
- {class}`.GenotypeResult`: Includes simulated genotypic information of causal sites

To show how the simulation results can be obtained from **tstrait**, we will be simulating quantitative traits of 5 individuals with 3 causal sites.

```{code-cell} ipython3
import msprime
import tstrait

num_ind = 5
ts = msprime.sim_ancestry(num_ind, sequence_length=1_000_000, recombination_rate=1e-8,
                          population_size=10**4, random_seed=1)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)

model = tstrait.TraitModelAlleleFrequency(trait_mean=0, trait_var=1, alpha=-0.3)
sim_result = tstrait.sim_phenotype(ts, num_causal=3, model=model, h2=0.3, random_seed=1)
```

(sec_simulation_output_phenotype)=

### Phenotype Output

Each {class}`.PhenotypeResult` object contains four [numpy.ndarray](https://numpy.org/doc/stable/reference/arrays.ndarray.html#arrays-ndarray) objects of equal length (`TreeSequence.num_individuals`): the first three are the the phenotypes and its genetic and environmental components. The fourth array keeps track of the individual IDs.

```{code-cell} ipython3
print(sim_result.phenotype)
```

All arrays are indexed based on the individual IDs. The $i$th entry of each array represents the value associated with the $i$th individual. We can obtain the array of individual ids

```{code-cell} ipython3
print(sim_result.phenotype.individual_id)
```

Further information regarding the individuals can be accessed by using the {ref}`Individual Table<tskit:sec_individual_table_definition>` in the tree sequence data through the individual IDs in `sim_result.phenotype.individual_id`. Note that `sim_result.phenotype.individual_id == np.array(list(ts.individuals()))`

For example, information regarding the first individual in the output can be obtained as following:

```{code-cell} ipython3
ts.individual(sim_result.phenotype.individual_id[0])
```

(sec_simulation_output_genotype)=

### Genotype Output

Each {class}`.GenotypeResult` object also contains four [numpy.ndarray](https://numpy.org/doc/stable/reference/arrays.ndarray.html#arrays-ndarray) objects of equal length (`num_causal`): the first three provide information on the causal allele, its random effect size and its causal allele frequency. The fourth array keeps track of `site_id` of the causal alleles.


```{code-cell} ipython3
print(sim_result.genotype)
```

The $i$th entry of each array represents the value associated with the $i$th causal site. Causal allele is randomly selected among the non-ancestral alleles at the causal site, and the allele frequency represents the frequency of the causal allele. We can obtain the array of site ids by running the following code:

```{code-cell} ipython3
print(sim_result.genotype.site_id)
```

Further information regarding the causal sites can be accessed by using the {ref}`Site Table<tskit:sec_site_table_definition>` in the tree sequence data through the site IDs in `sim_result.genotype.site_id`.

For example, information regarding the first causal site in the output can be obtained as following:

```{code-cell} ipython3
ts.site(sim_result.genotype.site_id[0])
```

(sec_simulation_heritability)=

## Narrow-Sense Heritability

Narrow-sense heritability in **tstrait** controls the variance of the simulated environmental noise. In the below example, we will be simulating quantitative traits by using the simulated tree sequence data in {ref}`msprime <msprime:sec_intro>` to show how narrow-sense heritability influences the relationship between environmental noise and phenotype.

```{code-cell} ipython3
import msprime
import tstrait
import matplotlib.pyplot as plt

num_ind = 500
ts = msprime.sim_ancestry(num_ind, sequence_length=1_000_000, recombination_rate=1e-8,
                          population_size=10**4, random_seed=1)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)
model = tstrait.TraitModelAdditive(trait_mean=0, trait_var=1)
```

### Example with $h^2=0.1$

When the narrow-sense heritability is set to a low number, most of the phenotypic variation is coming from the environmental variance.

```{code-cell} ipython3
sim_result = tstrait.sim_phenotype(ts, num_causal=1000, model=model, h2=0.1, random_seed=1)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Narrow-sense heritability = 0.1')
ax1.hist(sim_result.phenotype.environment_noise, bins=20)
ax1.set_xlabel("Environmental Noise")
ax1.set_xlim([-4, 4])
ax2.hist(sim_result.phenotype.phenotype, bins=20)
ax2.set_xlabel("Phenotype")
ax2.set_xlim([-4, 4])
plt.show()
```

### Example with $h^2=0.9$

When the narrow-sense heritability is set to a high number, most of the phenotypic variation is coming from the additive genetic variance, and the environmental variance is much smaller compared to the phenotypic variance.

```{code-cell} ipython3
sim_result = tstrait.sim_phenotype(ts, num_causal=1000, model=model, h2=0.9, random_seed=1)
phenotype_result = sim_result.phenotype

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Narrow-sense heritability = 0.9')
ax1.hist(sim_result.phenotype.environment_noise, bins=20)
ax1.set_xlabel("Environmental Noise")
ax1.set_xlim([-1.5, 1.5])
ax2.hist(sim_result.phenotype.phenotype, bins=20)
ax2.set_xlabel("Phenotype")
ax2.set_xlim([-1.5, 1.5])
plt.show()
```
