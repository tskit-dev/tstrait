# Welcome!

**tstrait** is a quantitative trait simulator that uses [tree sequence](https://tskit.dev/learn/) data as input. It supports the simulation of quantitative traits under the various additive models (see {ref}`sec_trait_model`). 
The details of the simulation model are indicated in {ref}`sec_simulation` page. 

Succinct tree sequences provide a highly efficient way of storing a set of related DNA sequences by encoding their ancestral history as a set of correlated trees along the genome. This data format is used by various software libraries, including {ref}`msprime <msprime:sec_intro>`, [SLiM](https://messerlab.org/slim/) and {ref}`stdpopsim <stdpopsim:sec_introduction>`. By relying on the underlying tree structure **tstrait** can efficiently simulate quantitative traits for large datasets.

Here is a list of resources to learn more about tree sequences:

- The [tree sequence tutorial](https://tskit.dev/learn/) describes what tree sequences are, and includes tutorials, publications and videos on tree sequence.
- The {ref}`msprime manual <msprime:sec_intro>` explains how genetic simulations can be conducted by using **msprime**.
- The {ref}`tskit tutorials <tskit-tutorials:sec_intro>` contain tutorials on how to analyze succinct tree sequences by using **tskit**.

If you use **tstrait** in your work, please cite `Tagami et al. (2023)`:

> Daiki Tagami, Gertjan Bisschop, and Jerome Kelleher (2023),
> *tstrait: a quantitative trait simulator of tree sequence data*,

## Contents

```{tableofcontents}
```