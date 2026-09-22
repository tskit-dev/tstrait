# Benchmarks

Performance benchmarks for tstrait. These are not run in CI; they are here so
that performance work can be repeated and compared. Results belong in the
commit that measured them, not in this file.

## `benchmark_genetic_value.py`

Measures how `tstrait.genetic_value` scales with the number of causal sites in
a trait.

```
uv run --group test benchmarks/benchmark_genetic_value.py
```

Every option has a default and `--help` lists them all. The ones that change
what is being measured rather than how long it takes:

`--preset {small,large}`
: Fills in `--samples`, `--length` and `--num-causal`. `small` is the default
  and takes about 25 seconds; `large` takes about six minutes. Either is
  simulated on first use and cached in `_output/`, keyed by the parameters.

`--selections {uniform,rare}`
: How the causal sites are drawn: `uniform` over all sites, or `rare`,
  restricted to those below `--rare-threshold`. The cost of a causal site is
  the number of nodes carrying its allele, so these differ by two orders of
  magnitude and a result quoted without saying which is meaningless.

`--levels {individual,node,edge}`
: Which of the three `genetic_value` levels to time.

`--num-threads`
: Worker threads to divide the causal sites between. 0, the default, does the
  work on the calling thread.

`--replicates`, `--max-seconds`
: How many times each cell is timed, and the budget after which the larger
  numbers of causal sites are skipped.

`sim_trait` is timed separately, because it has a per-site Python loop of its
own that should not be folded into the `genetic_value` numbers, and the numba
kernel is compiled by a warm up call that is not timed.

The mutation rate defaults to 1e-7, ten times the human rate, so that there are
enough sites in a genome short enough to simulate quickly. It does not affect
the allele frequency spectrum, so the causal sites are as weakly causal as they
would be under a realistic rate; only the number of sites per tree is inflated.

### Modes that say why, not just how long

Each roughly doubles the run, `--phases` most of all.

`--phases`
: Times `_check_trait_df`, `_GeneticValue.__init__`, the kernel and the output
  dataframe separately, which is what tells an algorithmic win from a setup one.

`--counters`
: Reports the nodes the descent reached. The run time is proportional to that,
  so seconds over visits is the constant an optimisation has to move. perf
  cannot attribute time inside the kernel (see below), so this is the way to
  say where the time goes.

`--structure`
: Reports the shape of the tree sequence and the distribution of how many nodes
  a causal site reaches. This is what makes one preset a fair substitute for
  another, so check it before trusting a new one.

`--memory`
: Peak resident set size per call. VmHWM never falls, so it is reset before
  each call by writing to `/proc/self/clear_refs`; without that the column
  reads `unavailable`.

### Output

Long format CSV to `_output/genetic_value.csv`, one row per replicate, stamped
with the dimensions of the tree sequence; `--counters` and `--memory` write
files alongside it. `_output/` is gitignored.

Nothing is checked in to diff against. Timings only mean anything on the
machine they came from, so take a baseline on yours before a change and compare
against that. The counts `--counters` writes are machine independent.

## `profile_genetic_value.py`

Profiles a single cell of the grid, in two modes because the Python setup and
the numba kernel need different tools.

```
uv run --group test benchmarks/profile_genetic_value.py --mode python
uv run --group test benchmarks/profile_genetic_value.py --mode kernel
```

`--mode python` is cProfile around the public call, which is the only way to
see the setup. The kernel appears in it as one opaque dispatcher frame.

`--mode kernel` sets one cell up and runs only the kernel, so that a sampling
profile is not swamped by simulating effect sizes for the whole site pool. Run
on its own it prints the perf commands to copy; `--run` is what those commands
invoke. `perf_event_paranoid` is usually high enough to need `sudo`.

Two things to know before reading a perf profile of this code.

**Source lines inside the kernel are not available.** This llvmlite has no LLVM
`PerfJITEventListener`, so nothing writes a `/tmp/perf-<pid>.map` or a jitdump
and the kernel shows up as raw addresses under `[JIT]`. Use `--counters` for
attribution inside the kernel. What perf does give is the split between the
kernel, the interpreter and LLVM compilation. Setup is a fixed few seconds of
that, so raise `--repeats` until the `[JIT]` share stops moving.

**Do not set `NUMBA_ENABLE_PROFILING=1`.** It is the documented way to profile
numba and is wrong here: it would only help through the listener llvmlite does
not have, and it defaults `NUMBA_DEBUGINFO` to 1, which changes the generated
code and measures a slower kernel than the one that runs. perf finds the JIT
mappings by itself.

## Gotchas

- Never compare replicate 0 against replicates 1 and up. `mutations_inherited_state`
  and its neighbours are built lazily and cached on the tree sequence, so the
  first call on a given tree sequence pays for all of them. The warm up call
  covers numba compilation but not this.
- The summary takes the minimum over replicates, not the mean.
- Threads are worth having only when the causal sites are many and not rare;
  each thread walks the whole tree sequence and holds arrays the length of the
  nodes, so a small or rare trait goes slower on more of them.
