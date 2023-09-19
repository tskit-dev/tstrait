(sec_intro)=

# Introduction to tstrait

Welcome to tstrait, a quantitative trait simulator based on
{ref}`succinct tree sequences <tskit-tutorials:sec_what_is>`. Tstrait supports simulation of quantitative
traits under various additive models, with effect sizes taken from specified probability distribution
including, but not limited to, the normal, t-, or Gamma distributions. Details of supported
distributions are in ([](effect_size_dist)). It also supports
multi-trait simulation under pleiotropy (see [](multi_trait) for details). For ease of use, the
output of tstrait is given as a {py:class}`pandas.DataFrame`.

## Advantages of tstrait

Tstrait is built on top of [tskit](https://tskit.dev/), and uses
{ref}`tree sequence <tskit-tutorials:sec_what_is>` data as an input. Tree sequences are designed to
efficiently store and process millions of DNA sequences. As a result, tstrait can simulate
quantitative traits substantially faster than working with the genotype matrix or other traditional
data structures.

Quantitative trait simulation in tstrait is transparent, and users can control each step in the simulation. Thus,
it is possible for the users to simulate their own environmental noise on top of simulated genetic values,
or even use their own defined effect sizes and causal sites. The tree sequence data structure is widely used in
various population genetic simulation packages, including [SLiM](https://messerlab.org/slim/),
[msprime](msprime:sec_intro), and [stdpopsim](stdpopsim:sec_introduction); it is therefore easy for
users of these packages to add quantitative traits to their results using tstrait.

## Tree Sequence resources

To learn more about tree sequences:

- The [tskit website](https://tskit.dev/) provides [learning materials](https://tskit.dev/learn/) explaining
  what tree sequences are, and includes tutorials, publications and videos.
- The [PySLiM manual](pyslim:sec_introduction) explains how forward genetic simulation can be create
  tree sequences.
- The [msprime manual](msprime:sec_intro) details an efficient backward-time genetic simulator that outputs
  tree sequences.
- The [tskit tutorials](tskit-tutorials:sec_intro) explain how to analyze succinct tree sequences
  by using [tskit](https://tskit.dev/).
