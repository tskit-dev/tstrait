# Welcome!

**tstrait** is a quantitative trait simulator of [tree sequence](https://tskit.dev/learn/) data. It supports the simulation of quantitative traits under the additive model and the allele frequency model. The details of the simulation model are indicated in {ref}`sec_simulation` and {ref}`sec_trait_model` sections. The input of the simulator will be the succinct tree sequence and the parameters that determine the nature of the simulation algorithm. The succinct tree sequence is a data structure that stores a biological structure known as ancestral recombination graph, and it is the output used in various software libraries, including {ref}`msprime <msprime:sec_intro>`, [SLiM](https://messerlab.org/slim/) and {ref}`stdpopsim <stdpopsim:sec_introduction>`. **tstrait** can simulate quantitative traits of individuals with a considerably faster computational speed compared with traditional techniques, as it uses a tree traversal algorithm to analyze tree sequence data.

There are a number of resources to learn about tree sequence:

- The [tree sequence tutorial](https://tskit.dev/learn/) describes what tree sequences are, and includes tutorials, publications and videos on tree sequence.
- The {ref}`msprime manual <msprime:sec_intro>` explains how genetic simulations can be conducted by using **msprime**.
- The {ref}`tskit tutorials <tskit-tutorials:sec_intro>` contain tutorials on how to analyze succinct tree sequences by using **tskit**.

If you use **tstrait** in your work, please cite `Tagami et al. (2023)`:

> Daiki Tagami, Gertjan Bisschop, and Jerome Kelleher (2023),
> *tstrait: a quantitative trait simulator of tree sequence data*,

## Contents

```{tableofcontents}
```