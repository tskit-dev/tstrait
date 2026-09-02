# Benchmarks

Performance benchmarks for tstrait. These are not run in CI; they are here so
that performance work can be repeated and compared.

## `benchmark_genetic_value.py`

Measures how `tstrait.genetic_value` scales with the number of causal sites in
a trait, which is the regime where the current implementation is slow: a trait
with a large number of weakly causal sites.

```
uv run --group test benchmarks/benchmark_genetic_value.py
```

By default this simulates 100,000 sample nodes (50,000 diploid individuals)
over 5Mb, and times `genetic_value` for 1, 100, 1000, 10,000 and 100,000 causal
sites at each of the `individual`, `node` and `edge` levels, taking the minimum
of three replicates.

The mutation rate defaults to 1e-7, ten times the human rate, so that there are
enough sites in a genome short enough to simulate quickly. The mutation rate
does not affect the allele frequency spectrum, so the causal sites are as
weakly causal as they would be under a realistic rate; only the number of sites
per tree is inflated.

`sim_trait` is timed separately, because it has a per-site Python loop of its
own that we do not want folded into the `genetic_value` numbers. The numba
kernels are compiled by a warm up call that is not timed.

The simulation parameters, the grid of causal site counts, the levels and the
number of replicates are all command line arguments; see `--help`.

### Output

The simulated tree sequence is cached in `_output/`, keyed by the simulation
parameters, so that repeated runs do not resimulate it. Results are written to
`_output/genetic_value.csv` in long format, one row per replicate, together
with the dimensions of the tree sequence they were measured on. `_output/` is
gitignored.
