# Welcome!

**tstrait** is a quantitative trait simulator of tree sequence data. It supports the simulation of quantitative traits under the additive model and the allele frequency model, which are described in detail in [simulation model](simulation.md) and [trait model](model.md). The input of the simulator will be the tree sequence structured data and the parameters that determine the nature of the quantitative trait simulation algorithm. The tree sequence format is a data structure that stores a biological structure known as ancestral recombination graph, and it is the output used in various software libraries, including **msprime** ([Kelleher et al., 2016](https://doi.org/10.1038/s41588-019-0483-y), [Kelleher et al., 2019](https://doi.org/10.1038/s41588-019-0483-y), [Baumdicker et al., 2022](https://doi.org/10.1093/genetics/iyab229)). **tstrait** can simulate quantitative traits of individuals with a considerably faster computational speed, as it uses tree traversal algorithm to analyze tree sequence data, instead of analyzing the genotype matrix.

There are a number of resources to learn about tree sequence and **msprime**:

- The [msprime manual](https://tskit.dev/msprime/docs/stable/intro.html) explains how genetic simulations can be conducted by using **msprime**.
- The [tskit tutorials](https://tskit.dev/tutorials) site contains tutorials on how to analyze simulated tree sequences by using **tskit**.

If you use **tstrait** in your work, please cite

## Contents

```{tableofcontents}
```