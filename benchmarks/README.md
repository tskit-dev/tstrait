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

`--preset small` is the default and takes about 25 seconds; `--preset large` is
the tree sequence this work was written against and takes about six minutes,
with its longest single call at 33 seconds.

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

`small` is the default because iterating against a six minute grid is not
practical. It reproduces the patterns the large one shows, at `level="node"`:

| num_causal | small uniform | small rare | ratio | large uniform | large rare | ratio |
|---|---|---|---|---|---|---|
| 1 | 0.013s | 0.013s | 1.0 | 0.062s | 0.061s | 1.0 |
| 100 | 0.024s | 0.014s | 1.8 | 0.118s | 0.067s | 1.8 |
| 1,000 | 0.125s | 0.017s | 7.4 | 0.443s | 0.072s | 6.1 |
| 10,000 | 1.088s | 0.023s | 47 | 3.848s | 0.091s | 42 |
| 100,000 | — | — | | 32.589s | 0.216s | 151 |

Both show a flat per call floor, a uniform µs/site that falls to a plateau from
1,000 causal sites upwards, a rare µs/site that is still falling at the top of
the grid, a uniform to rare ratio that grows with the number of causal sites,
and parity between the three levels. The uniform plateau is 109µs/site against
326µs/site, a factor of 3.0 on a node count ratio of 3.4.

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

### The two implementations

There are two, and `--threshold` chooses between them per causal site. Below
the threshold a site goes through the push down of the ARG, which sweeps the
nodes once from the past to the present carrying the effect of each causal
mutation to its carriers. At or above it the site goes through a descent of the
trees, which builds each tree in turn and walks down from each causal mutation.
`0` sends everything to the descent and `inf` everything to the push down.

Both cost the nodes that carry a causal allele. The descent is cheaper per
node, around 20ns against 28ns, because a child is the next link of a list
rather than a search of the out edges of a node, and because it holds nothing
per node while it works. Against that it has to build the trees, one insertion
and one removal per edge, which the push down never does.

That fixed cost is the whole of the argument for a threshold, and the
measurement says it is not enough of one. Comparing thresholds over the whole
grid, on both presets, at all three levels, for one and three traits, and for
causal sites drawn both uniformly and from the rare ones, a threshold of zero
was fastest or tied in every cell:

| | uniform 1,000 | uniform 10,000 | rare 1,000 | rare 10,000 |
|---|---|---|---|---|
| small, all descent | 122ms | 1058ms | 15ms | 23ms |
| small, all push down | 196ms | 1619ms | 16ms | 35ms |
| large, all descent | 427ms | 3709ms | 71ms | 90ms |
| large, all push down | 592ms | 4917ms | 68ms | 108ms |

The push down leads only where there are few causal sites and they are rare, so
the tree pass is not amortised, and there it leads by a millisecond or two on a
call that takes a few. With three traits the gap widens the other way, to 2.27x
on rare causal sites, because the push down sweeps once per trait where the
descent takes all of them in one pass.

So the default threshold is zero and the push down is not used. It is kept, and
`--threshold inf` still runs it, only so that the two can go on being compared;
the intention is to end with the descent alone.

### What else it measures

Wall time on its own does not say why a configuration is slow. Four optional
modes say more. `--phases` is the expensive one, roughly doubling the run
because it times the same work again a piece at a time; the other three add
seconds.

`--phases` times `_check_trait_df`, `_GeneticValue.__init__`, the kernel and
the output dataframe separately. The end to end
number stays as the headline. This is what tells an algorithmic win from a
setup win: setup barely grows with the causal sites, going from 8ms to 17ms
across the whole `small` grid and sitting near 100ms on `large`, so at one
causal site the public call is measuring almost nothing else, while at 10,000
the kernel is 98% of it. Most of setup is `tskit.jit.numba.jitwrap`, which runs
three Python-speed `max(map(len, ...))` passes over the site and mutation
tables. The dataframe is about 1ms and is not worth thinking about.

`--counters` reports the work the descent does. perf cannot attribute time to
source lines inside a numba kernel here (see below), so counting what the
kernel does and dividing is the way to say where the time goes. The kernel
returns the count itself rather than there being a second copy of the loop to
keep in step with the first. On `small`:

```
selection  num_causal           rows         visits  visits/row  of num_nodes
uniform             1              1         33,819     33819.0       53.44%
uniform         10000         10,000     52,900,059      5290.0        8.36%
rare                1              1             11        11.0        0.02%
rare            10000         10,000        131,744        13.2        0.02%
```

`visits` is the number of nodes the descent reached, which is what the run time
is proportional to, so `ns/visit` in the summary table is the constant an
optimisation has to move. Two things fall out of the table:

- `visits/row` as a fraction of the nodes is the carrier fraction that
  `--structure` measures independently, and the two agree: 8.4% for uniformly
  drawn causal sites against a measured mean of 7.2%, and 0.02% for rare ones
  against a measured median of 0.4%. A single uniformly drawn site reaching
  53% is one draw from a distribution with a long tail.
- `ns/visit` is around 20ns once there are enough causal sites to amortise the
  tree pass, and hundreds of nanoseconds below that. The pass is 2.2ms on
  `small` and 12.4ms on `large`, so a rare trait of a hundred causal sites is
  paying for a tree sequence it barely touches. That is the one regime the
  push down still wins, by a millisecond or two.

`--structure` reports the shape of the tree sequence and the carrier fraction
distribution described above.

`--memory` reports the peak resident set size of each call. VmHWM never falls,
so it is reset before each call by writing to `/proc/self/clear_refs`; on a
kernel without that the column reads `unavailable`.

It is the clearest difference between the two implementations. The push down
holds the seeds still in flight, which grows with the causal sites; the descent
holds a fixed handful of arrays the length of the nodes however many causal
sites there are. At 100,000 uniformly drawn causal sites on `large`:

| | peak RSS | over baseline |
|---|---|---|
| descent | 0.42 GB | 0.01 GB |
| push down | 1.30 GB | 0.89 GB |

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

Note that `--mode kernel` runs the push down, not the descent, so its numbers
describe the implementation that `--threshold inf` selects rather than the
default one. The perf figures below were taken that way.

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

- `_GeneticValue` takes a `_root_runs` branch when a drawn causal allele is the
  ancestral state of its site *and* that row goes to the push down. It is a
  Python loop over every tree, costing 7ms on `small` and 38ms on `large`, so on
  `large` it was over a third of the setup. It is not an edge case: drawing
  10,000 sites uniformly makes it near certain that one of them qualifies. At
  the default threshold those rows go to the descent, which has the roots to
  hand, and the branch never runs; `--threshold inf` brings it back. The
  benchmark prints a line when a cell actually takes it.
- Never compare replicate 0 against replicates 1 and up. `mutations_inherited_state`
  and its neighbours are built lazily and cached on the tree sequence, so the
  first call on a given tree sequence pays for all of them. The warm up call
  covers numba compilation but not this.
- The summary takes the minimum over replicates, not the mean.
