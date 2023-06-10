# Output

The outputs of the `sim_phenotypes()` function are two dataclass objects. The first output is a `PhenotypeResult` object, which includes information regarding the simulated individuals. The second output is a `GeneticValueResult` 
describes the phenotypic information regarding the simulated  and it includes the individual ID, simulated value of phenotype, genetic value and environmental noise. The second output is a `GeneticValueResult` object, and it includes the site ID, causal allele, effect size and causal allele frequency.

To ensure that the output of the example is the same, we set a random seed.