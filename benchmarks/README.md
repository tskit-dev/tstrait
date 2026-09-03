# Benchmarks

Performance benchmarks for tstrait. These are not run in CI; they are here so
that performance work can be repeated and compared.

## `benchmark_genetic_value.py`

Measures how `tstrait.genetic_value` scales with the number of causal sites in
a trait.

```
uv run --group test benchmarks/benchmark_genetic_value.py
```

The cost of the ARG sweep is the number of nodes that carry a causal allele, so
the allele frequency of the causal sites matters more than how many there are.
`--selections` therefore draws them two ways: `uniform` over all sites, which
the common variants in the tail of the frequency spectrum dominate, and `rare`,
restricted to sites below `--rare-threshold`. The two differ by two orders of
magnitude and behave differently, so a single number for "the cost of a causal
site" is meaningless without saying which.

`sim_trait` is timed separately, because it has a per-site Python loop of its
own that we do not want folded into the `genetic_value` numbers. The numba
kernels are compiled by a warm up call that is not timed.

The mutation rate defaults to 1e-7, ten times the human rate, so that there are
enough sites in a genome short enough to simulate quickly. The mutation rate
does not affect the allele frequency spectrum, so the causal sites are as
weakly causal as they would be under a realistic rate; only the number of sites
per tree is inflated.

### Presets

`--preset small` is the default and takes about 30 seconds; `--preset large` is
the tree sequence the ARG sweep was written against and takes about nine
minutes, with its longest single call at 48 seconds.

| | small | large |
|---|---|---|
| samples | 30,000 | 100,000 |
| sequence length | 1Mb | 5Mb |
| nodes | 63,287 | 217,091 |
| edges | 75,586 | 285,979 |
| trees | 4,108 | 22,523 |
| sites | 42,794 | 235,458 |
| causal sites | 1 … 10,000 | 1 … 100,000 |
| cached tree sequence | 6.5MB | 27MB |

A preset only fills in `--samples`, `--length` and `--num-causal`, each of which
still overrides it when given explicitly. The tree sequence is cached in
`_output/` under a name built from the simulation parameters, so `small`
simulates itself in 0.2s the first time and is loaded after that.

`small` is the default because iterating against a nine minute grid is not
practical. It reproduces the patterns the large one shows, at `level="node"`:

| num_causal | small uniform | small rare | ratio | large uniform | large rare | ratio |
|---|---|---|---|---|---|---|
| 1 | 0.016s | 0.011s | 1.4 | 0.058s | 0.051s | 1.1 |
| 100 | 0.036s | 0.013s | 2.8 | 0.152s | 0.056s | 2.7 |
| 1,000 | 0.189s | 0.015s | 12.3 | 0.593s | 0.065s | 9.1 |
| 10,000 | 1.596s | 0.035s | 45.7 | 5.171s | 0.108s | 48 |
| 100,000 | — | — | | 48.310s | 0.476s | 101 |

Both show a flat per call floor, a uniform µs/site that falls to a plateau from
1,000 causal sites upwards, a rare µs/site that is still falling at the top of
the grid, a uniform to rare ratio that grows with the number of causal sites,
and parity between the three levels. The uniform plateau is 160µs/site against
483µs/site, a factor of 3.0 on a node count ratio of 3.4.

What makes the small preset a fair substitute is not the timings but the
distribution underneath them: the fraction of the nodes that a causal site's
effect reaches, which is what the sweep costs. `--structure` measures it.

| | mean carrier fraction | median |
|---|---|---|
| small | 7.19% | 0.406% |
| large | 5.99% | 0.184% |

Those are 150 sampled sites from a distribution with a heavy tail, so they
agree about as well as they can. Check this again before trusting a new preset.

The grid stops at 10,000 causal sites on `small` because the rare pool is only
about 37% of its 42,794 sites. Raising the mutation rate to lift the ceiling
was tried and rejected: it takes `sites/nodes` from 0.68 to about 2.7 against
the large preset's 1.08, and the per-call setup floor grows with the number of
sites, so the low end of the curve stops being comparable.

### What else it measures

Wall time on its own does not say why a configuration is slow. Four optional
modes say more. `--phases` is the expensive one, roughly doubling the run
because it times the same work again a piece at a time; the other three add
seconds.

`--phases` times `_check_trait_df`, `_GeneticValue.__init__`, the
`_push_down_arg` kernel and the output dataframe separately. The end to end
number stays as the headline. This is what tells an algorithmic win from a
setup win: setup barely grows with the causal sites, going from 8ms to 17ms
across the whole `small` grid and sitting near 100ms on `large`, so at one
causal site the public call is measuring almost nothing else, while at 10,000
the kernel is 98% of it. Most of setup is `tskit.jit.numba.jitwrap`, which runs
three Python-speed `max(map(len, ...))` passes over the site and mutation
tables; `_root_runs` is most of the rest when it fires. The dataframe is about
1ms and is not worth thinking about.

`--counters` runs a counting-only copy of the sweep and reports the work it
does. perf cannot attribute time to source lines inside a numba kernel here
(see below), so counting what the kernel does and dividing is the way to say
where the time goes. On `small`:

```
selection  num_causal   seeds   edge_scans   edge_hits    appends   reached  scans/seed  hit rate  reached
uniform             1       1       35,915      33,818     16,909    16,909     35915.0     94.2%    26.7%
uniform         10000  10,107   56,556,279  53,117,294 26,558,647    32,966      5595.8     93.9%    52.1%
rare                1       1           10          10          5         5        10.0    100.0%     0.0%
rare            10000  10,028      318,619     302,898    151,449    29,775        31.8     95.1%    47.0%
```

`edge_scans` is the trip count of the kernel's innermost loop and is what the
run time is proportional to, so `ns/scan` in the summary table is the constant
an optimisation has to move. Three things fall out of the table above:

- The scan of a node's out edges is not where the waste is: 94% of trips find
  an edge that spans the seed's causal site.
- Uniform selection costs 178 times as many scans as rare at 10,000 causal
  sites but only 46 times the time, because rare is more expensive per scan:
  63ns against 28ns for the kernel alone. Following a few carriers is random
  access; a dense pass over half the nodes is sequential. Compare the `kernel`
  rows rather than `genetic_value`, or the setup floor swamps the rare ones.
- The kernel has an O(num_nodes) floor. It builds a pending entry for every
  node before it starts and visits every node whether or not anything reached
  it, so at one rare causal site — 10 scans — the kernel still takes 2ms.

`--structure` reports the shape of the tree sequence and the carrier fraction
distribution described above.

`--memory` reports the peak resident set size of each call, which is how the
typed list rewrite was justified. VmHWM never falls, so it is reset before each
call by writing to `/proc/self/clear_refs`; on a kernel without that the column
reads `unavailable`. This is the one thing the small preset is a poor substitute
for: 10,000 uniform causal sites peak at 0.02GB over the baseline there, against
the gigabytes the large preset reaches at 100,000.

### Output

Results are written to `_output/genetic_value.csv` in long format, one row per
replicate, together with the dimensions of the tree sequence they were measured
on; `--counters` writes a second file alongside it. `_output/` is gitignored.

`baseline_small.csv` and `baseline_small_counters.csv` are the `small` preset at
the tip of the ARG sweep work, for diffing against. The timings in the first are
specific to the machine they were taken on; the counts in the second are not,
and are the part worth treating as a regression test.

## `profile_genetic_value.py`

Profiles a single cell of the grid, in two modes because the Python setup and
the numba kernel need different tools.

```
uv run --group test benchmarks/profile_genetic_value.py --mode python
uv run --group test benchmarks/profile_genetic_value.py --mode kernel
```

`--mode python` is cProfile around the public call. It is the only way to see
the setup, and it shows `jitwrap` and its `builtins.max` rows plainly. The
kernel appears in it as one opaque dispatcher frame.

`--mode kernel` sets one cell up and runs only the kernel, so that a sampling
profile is not swamped by simulating effect sizes for the whole site pool. Run
on its own it prints the perf commands to copy; `--run` is what those commands
invoke. `perf_event_paranoid` is usually high enough to need `sudo`.

Two things to know before reading a perf profile of this code.

**Source lines inside the kernel are not available.** This llvmlite has no LLVM
`PerfJITEventListener`, so nothing writes a `/tmp/perf-<pid>.map` or a jitdump
and the kernel shows up as raw addresses under `[JIT]`. Use `--counters` for
attribution inside the kernel. What perf does give is the split between the
kernel, the numba runtime, the interpreter and LLVM compilation, with the typed
list functions resolved by name — which is how to size the typed list overhead:

```
64.42%  [JIT] tid 4722                    <- the kernel
12.18%  python3.11                        <- setup
 6.27%  libc.so.6
 6.02%  libllvmlite.so                    <- compilation, not work
 4.93%  _helperlib.cpython-311-...so      <- numba_list_append, numba_list_resize
```

That is `--repeats 10`; the setup is a fixed few seconds, so raise `--repeats`
until the `[JIT]` share stops moving.

**Do not set `NUMBA_ENABLE_PROFILING=1`.** It is the documented way to profile
numba and is wrong here: it would only help through the listener llvmlite does
not have, and it defaults `NUMBA_DEBUGINFO` to 1, which changes the generated
code. The same kernel measured 2.62s a call with it and 1.61s without. perf
finds the JIT mappings by itself, so the recipe does not set it.

## Gotchas

- `_GeneticValue` takes a `_root_runs` branch whenever a drawn causal allele is
  the ancestral state of its site. It is a Python loop over every tree, costing
  7ms on `small` and 38ms on `large`, so on `large` it is over a third of the
  setup. It is not an edge case: drawing 10,000 sites uniformly makes it near
  certain that one of them qualifies, and it fires at 10,000 and 100,000 causal
  sites on both presets. At the low end of the grid it usually does not, so it
  appears part way up the curve and looks like a step in the setup cost. The
  benchmark prints a line when a cell takes it.
- Never compare replicate 0 against replicates 1 and up. `mutations_inherited_state`
  and its neighbours are built lazily and cached on the tree sequence, so the
  first call on a given tree sequence pays for all of them. The warm up call
  covers numba compilation but not this.
- The summary takes the minimum over replicates, not the mean.
