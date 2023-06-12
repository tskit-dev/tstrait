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

# Simulation Model

The definitions of some key terms in this documentation are indicated {ref}`here <tskit:sec_glossary>`.

**tstrait** assumes that the individual's phenotypes are obtained from the following additive model,

$$
y=X\beta+\epsilon.
$$

Here, $X$ is the matrix that denotes the number of causal alleles inside the individual, $\beta$ is the vector of effect sizes, and $\epsilon$ is the vector of environmental noise.

The simulation model initially chooses $m$ causal sites at random among the sites in the tree sequence data, and causal allele for each site is chosen at random among the mutations in the site. Ancestral state is not chosen to be the causal allele. For each causal site $i$, **tstrait** calculates the causal allele frequency $p_i$ and simulates the effect size $\beta_i$ as indicated in [trait model](model.md).

After the genetic value, $G=X\beta$, is simulated from the tree sequence data, the environmental noise $\epsilon$ is simulated from

$$
\epsilon_j\sim N\left(0,Var(G)\cdot\frac{(1-h^2)}{h^2}\right),
$$

where $Var(G)$ is the variance of the simulated genetic values and $h^2$ is the narrow-sense heritability which is defined by the user. The phenotypic values of individuals are determined by adding the genetic values and environmental noise.

It is possible for the user to set $h^2=0$ or $h^2=1$. When $h^2=0$, the phenotypic values will be exactly the same as the environmental noise, and when $h^2=1$, the environmental noise $\epsilon$ will be a vector of zeros.

The number of causal sites $m$ and the narrow-sense heritability $h^2$ of the simulation model are specified in `num_causal` and `h2` arguments in {func}`.sim_phenotypes` function. For example,

```Python
tstrait.sim_phenotypes(ts, num_causal = 5, h2 = 0.3, model = model)
```
simulates quantitative traits of individuals in `ts` tree sequence data from the trait model `model` with 5 causal sites and narrow-sense heritability being 0.3.

The example usage of `tstrait` is shown in [quickstart](quickstart.md).

## Output

The outputs of the `sim_phenotypes()` function are two dataclass objects, `PhenotypeResult` and `GeneticValueResult`. The first output is a `PhenotypeResult` object, and it includes the individual ID, simulated value of phenotype, genetic value and environmental noise. The second output is a `GeneticValueResult` object, and it includes the site ID, causal allele, effect size and causal allele frequency.