(sec_intro)=

# Introduction to tstrait

Welcome to tstrait package! tstrait is a quantitative trait simulator. It supports simulation of quantitative
traits under various additive models, which are not limited to normal distribution, t-distribution and Gamma
distribution. The details of the supported distributions are in ([](effect_size_dist)). It also supports
multi-trait simulation under pleiotropy (see [](multi_trait) for details). The output of tstrait is given
as a {py:class}`pandas.DataFrame`, which will make the simulation output easy to read.

## Advantages of tstrait

tstrait is built on top of [tskit](https://tskit.dev/), and uses
{ref}`tree sequence <tskit-tutorials:sec_what_is>` data as an input. Tree sequence is a data structure that is
used to efficiently store and process millions of DNA sequences. Due to its computational advantages, tstrait
can simulate quantitative traits substantially faster than working with the genotype matrix or other traditional
data structures.

Quantitative trait simulation in tstrait is transparent, and users can control each step in the simulation. Thus,
it would be possible for the users to simulate their own environmental noise on top of simulated genetic values,
or even use their own defined effect sizes and causal sites. As tree sequence data structure is widely used in
various population genetic simulation packages, including [SLiM](https://messerlab.org/slim/),
[msprime](msprime:sec_intro), and [stdpopsim](stdpopsim:sec_introduction), it will be possible for the
users to use simulation results from these packages to conduct quantitative trait siumulation in tstrait.

## Tree Sequence resources

Here is a list of resources to learn more about tree sequences:

- The [tree sequence tutorial](https://tskit.dev/learn/) explains what tree sequences are, and includes tutorials,
  publications and videos on tree sequence.
- The [PySLiM manual](pyslim:sec_introduction) explains how forward genetic simulation can be conducted by
  using tree sequence.
- The [msprime manual](msprime:sec_intro) explains how tree sequence genetic simulation can be conducted by
  using [msprime](msprime:sec_intro).
- The [tskit tutorials](tskit-tutorials:sec_intro) explain how to analyze succinct tree sequences
  by using [tskit](https://tskit.dev/).
