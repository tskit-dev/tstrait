"""
Benchmark tstrait.genetic_value as the number of causal sites grows.

The interesting regime is a trait with a large number of weakly causal sites,
i.e. many causal sites that are each carried by a small number of samples. The
cost of the ARG sweep is the number of nodes that carry a causal allele, so the
allele frequency of the causal sites matters more than how many there are, and
``--selections`` draws them either uniformly over all sites or from the rare
ones alone.

Run with ``uv run --group test benchmarks/benchmark_genetic_value.py``.
"""

import argparse
import csv
import functools
import pathlib
import statistics
import sys
import time

import msprime
import numba
import numpy as np
import pandas as pd
import tskit
from numba.core import types
from numba.typed import List

import tstrait
from tstrait import jit
from tstrait.genetic_value import _check_trait_df, _GeneticValue

LEVELS = ["individual", "node", "edge"]
SELECTIONS = ["uniform", "rare"]

# The small preset is the default because the large one takes about six minutes
# and its longest single call takes almost a minute, which is too slow to
# iterate against. It reproduces the same patterns; see the README for the
# evidence, and --structure to check it again after a change.
PRESETS = {
    "small": {
        "samples": 30_000,
        "length": 1e6,
        "num_causal": [1, 100, 1000, 10_000],
    },
    "large": {
        "samples": 100_000,
        "length": 5e6,
        "num_causal": [1, 100, 1000, 10_000, 100_000],
    },
}

# The counting kernel mirrors tstrait.jit._push_down_arg, so it holds what a
# node holds there: the indexes of the seeds whose effect has reached it.
_SEED_LIST = types.ListType(types.int32)

COUNTERS = ["seeds", "edge_scans", "edge_hits", "appends", "reached"]


@numba.njit
def _count_push_down_arg(
    child_index,
    edges_child,
    edges_site_start,
    edges_site_stop,
    nodes_by_time,
    seed_node,
    seed_site,
):
    """
    Count the work that ``tstrait.jit._push_down_arg`` does, without doing any
    of it.

    perf cannot attribute time to source lines inside a numba kernel here, so
    the way to say where the time goes is to count the things the kernel does
    and divide. This is a copy of the sweep with the output writes and the
    weight lookups taken out and a counter put in their place, which is the
    only reason it may diverge from the kernel it mirrors: keep them in step.

    Returns the counters named in ``COUNTERS``:

    ``seeds``       the causal mutations, plus the roots seeded when the causal
                    allele is the ancestral state
    ``edge_scans``  trips of the innermost loop, i.e. the sum over every
                    (swept node, seed held there) pair of the node's out degree
    ``edge_hits``   those trips where the edge spans the seed's causal site, so
                    edge_scans - edge_hits is the scan that was wasted
    ``appends``     seeds pushed onto a node's list
    ``reached``     nodes that held a list, so reached / num_nodes is the
                    fraction of the tree sequence the sweep touched, and, since
                    a list is made for a node the first time anything reaches
                    it, also the number of typed lists allocated
    """
    num_nodes = len(child_index)
    empty = List.empty_list(types.int32)
    pending = List.empty_list(_SEED_LIST)
    for _ in range(num_nodes):
        pending.append(empty)
    reached = np.zeros(num_nodes, dtype=np.bool_)

    edge_scans = 0
    edge_hits = 0
    appends = 0

    for j in range(len(seed_node)):
        u = seed_node[j]
        if child_index[u, 0] < 0:
            continue
        if not reached[u]:
            pending[u] = List.empty_list(types.int32)
            reached[u] = True
        pending[u].append(np.int32(j))
        appends += 1

    for i in range(len(nodes_by_time)):
        parent = nodes_by_time[i]
        if not reached[parent]:
            continue
        items = pending[parent]
        edge_start = child_index[parent, 0]
        edge_stop = child_index[parent, 1]
        for k in range(len(items)):
            item = items[k]
            site = seed_site[item]
            edge_scans += edge_stop - edge_start
            for e in range(edge_start, edge_stop):
                if edges_site_start[e] <= site and site < edges_site_stop[e]:
                    edge_hits += 1
                    child = edges_child[e]
                    if child_index[child, 0] < 0:
                        continue
                    if not reached[child]:
                        pending[child] = List.empty_list(types.int32)
                        reached[child] = True
                    pending[child].append(np.int32(item))
                    appends += 1
        pending[parent] = empty

    return (
        len(seed_node),
        edge_scans,
        edge_hits,
        appends,
        int(np.sum(reached)),
    )


def cached_simulation(args):
    """
    Return the tree sequence defined by the simulation arguments, simulating it
    and caching it to disk if we have not done so already.
    """
    name = (
        f"n={args.samples}_L={args.length:.0f}_r={args.recombination_rate:g}"
        f"_mu={args.mutation_rate:g}_Ne={args.population_size:g}_seed={args.seed}"
    )
    path = args.cache_dir / f"{name}.trees"
    if path.exists():
        print(f"Loading cached {path}")
        return tskit.load(path)

    print(f"Simulating {name}")
    before = time.perf_counter()
    ts = msprime.sim_ancestry(
        args.samples // 2,
        ploidy=2,
        sequence_length=args.length,
        recombination_rate=args.recombination_rate,
        population_size=args.population_size,
        random_seed=args.seed,
    )
    ancestry_time = time.perf_counter() - before
    before = time.perf_counter()
    ts = msprime.sim_mutations(ts, rate=args.mutation_rate, random_seed=args.seed)
    mutation_time = time.perf_counter() - before
    print(f"  ancestry {ancestry_time:.1f}s, mutations {mutation_time:.1f}s")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    ts.dump(path)
    print(f"  cached to {path}")
    return ts


def describe(ts):
    return (
        f"samples={ts.num_samples} individuals={ts.num_individuals} "
        f"nodes={ts.num_nodes} edges={ts.num_edges} trees={ts.num_trees} "
        f"sites={ts.num_sites}"
    )


def time_call(func, replicates):
    """
    Call func the given number of times, returning the result of the last call
    and the elapsed time of each call.

    The first call on a tree sequence is systematically more expensive than the
    ones after it, because tskit builds mutations_inherited_state and its
    neighbours lazily and caches them on the tree sequence, so a replicate is
    only comparable with another replicate of the same call.
    """
    times = []
    for _ in range(replicates):
        before = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - before)
    return result, times


def _read_status(key):
    for line in open("/proc/self/status"):
        if line.startswith(key):
            return int(line.split()[1]) * 1024
    return None


def peak_memory(func):
    """
    Return the result of func and the peak resident set size while it ran.

    VmHWM is a high water mark that never falls, so it has to be reset before
    the call rather than subtracted after it; writing 5 to clear_refs does
    that. Returns None for the memory where the kernel does not support it.
    """
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
    except OSError:
        return func(), None
    before = _read_status("VmRSS")
    result = func()
    return result, (_read_status("VmHWM"), before)


def causal_site_pool(ts, model, seed):
    """
    Simulate effect sizes for every site, so that causal sites can be drawn
    from the pool by allele frequency without resimulating.
    """
    before = time.perf_counter()
    pool = tstrait.sim_trait(ts, model=model, num_causal=ts.num_sites, random_seed=seed)
    print(
        f"Effect sizes for all {ts.num_sites} sites: {time.perf_counter() - before:.1f}s"
    )
    return pool


def select_causal(pool, selection, num_causal, rare_threshold, rng):
    """
    Draw num_causal rows from the pool, either uniformly over the sites or
    restricted to the rare ones. Uniform selection is dominated by the common
    variants in the tail of the frequency spectrum, which behave quite
    differently from the weakly causal sites the ARG sweep is aimed at.
    """
    if selection == "rare":
        pool = pool[pool["allele_freq"] < rare_threshold]
    if len(pool) < num_causal:
        return None
    keep = rng.choice(len(pool), size=num_causal, replace=False)
    return pool.iloc[np.sort(keep)]


def warm_up(ts, model, levels, counters):
    """
    Run the smallest possible simulation through each code path so that the
    numba kernels are compiled before we start timing.
    """
    before = time.perf_counter()
    trait_df = tstrait.sim_trait(ts, model=model, num_causal=1, random_seed=1)
    for level in levels:
        tstrait.genetic_value(ts, trait_df, level=level)
    if counters:
        # A kernel of its own to compile, so only pay for it when it is used.
        count_work(ts, _check_trait_df(ts, trait_df))
    print(f"Warm up (includes numba compilation): {time.perf_counter() - before:.1f}s")


def time_phases(ts, trait_df, level, replicates):
    """
    Time the phases of a genetic_value call separately, returning a list of
    (phase, times) pairs.

    The public call is a single number in which a flat per call setup cost and
    a kernel that grows with the causal sites are indistinguishable, and at one
    causal site it is almost all setup. Splitting them is what tells an
    algorithmic win from a setup win.
    """
    phases = []
    _, times = time_call(functools.partial(_check_trait_df, ts, trait_df), replicates)
    phases.append(("check", times))
    checked = _check_trait_df(ts, trait_df)

    _, times = time_call(functools.partial(_GeneticValue, ts, checked), replicates)
    phases.append(("setup", times))
    genetic = _GeneticValue(ts, checked)

    size = genetic._output_size(level)
    arguments = genetic._descent_arguments(level, 0, np.zeros(size))
    _, times = time_call(
        # A fresh output array each time, since the kernel accumulates into it.
        lambda: jit._push_down_arg(
            **{**arguments, "output": np.zeros(len(arguments["output"]))}
        ),
        replicates,
    )
    phases.append(("kernel", times))

    values = np.zeros((genetic.num_trait, size))
    _, times = time_call(
        functools.partial(_build_frame, genetic.num_trait, size, level, values),
        replicates,
    )
    phases.append(("frame", times))
    return phases


def _build_frame(num_trait, size, level, values):
    """
    The dataframe _run builds from the kernel output, at genetic_value.py:249.
    """
    return pd.DataFrame(
        {
            "trait_id": np.repeat(np.arange(num_trait), size),
            f"{level}_id": np.tile(np.arange(size), num_trait),
            "genetic_value": values.flatten(),
        }
    )


def count_work(ts, trait_df):
    """
    Return the work counters for a trait, as a dict keyed by COUNTERS.

    The counts do not depend on the level, because the same sweep serves all
    three and only the array the contributions land in differs.
    """
    genetic = _GeneticValue(ts, trait_df)
    arguments = genetic._descent_arguments("node", 0, np.zeros(ts.num_nodes))
    counts = _count_push_down_arg(
        arguments["child_index"],
        arguments["edges_child"],
        arguments["edges_site_start"],
        arguments["edges_site_stop"],
        arguments["nodes_by_time"],
        arguments["seed_node"],
        arguments["seed_site"],
    )
    return dict(zip(COUNTERS, counts))


def root_runs_fired(ts, trait_df):
    """
    Whether this trait takes the _root_runs branch in _GeneticValue.

    That branch is a Python loop over every tree, so it costs O(num_trees) at
    Python speed, and it fires only when a drawn causal allele happens to be
    the ancestral state of its site. It therefore appears and vanishes with the
    seed, which makes for confusing non-monotonic timings unless it is reported.
    """
    site_id = trait_df["site_id"].to_numpy()
    causal_allele = trait_df["causal_allele"].to_numpy()
    return bool(np.any(ts.sites_ancestral_state[site_id] == causal_allele))


def carrier_fractions(ts, pool, sample_size, rng):
    """
    Return the fraction of the nodes that each of a sample of causal sites
    reaches.

    The cost of the sweep is the number of nodes a causal allele is carried by,
    so this distribution, not the size of the tree sequence, is what a smaller
    simulation has to reproduce. Giving each sampled site its own trait_id gets
    all of them from a single setup, and a node the effect never reached is
    exactly a node left at zero.
    """
    size = min(sample_size, len(pool))
    keep = np.sort(rng.choice(len(pool), size=size, replace=False))
    trait_df = pool.iloc[keep].copy()
    trait_df["trait_id"] = np.arange(len(trait_df))
    values = tstrait.genetic_value(ts, trait_df, level="node")["genetic_value"]
    reached = values.to_numpy().reshape(len(trait_df), ts.num_nodes)
    return np.count_nonzero(reached, axis=1) / ts.num_nodes


def report_structure(ts, pool, args):
    """
    Print the shape of the tree sequence and the carrier fraction distribution,
    which is the check that a preset still reproduces the large simulation.
    """
    rng = np.random.default_rng(args.seed)
    fractions = carrier_fractions(ts, pool, args.structure_sites, rng)
    print(f"\n{describe(ts)}")
    print(
        f"sites/nodes {ts.num_sites / ts.num_nodes:.2f}    "
        f"edges/nodes {ts.num_edges / ts.num_nodes:.2f}"
    )
    print(f"Carrier fraction of nodes over {len(fractions)} sampled sites:")
    print(
        f"  mean {fractions.mean() * 100:.2f}%    "
        f"median {statistics.median(fractions) * 100:.3f}%    "
        f"max {fractions.max() * 100:.2f}%"
    )


def run_benchmark(ts, args):
    """
    Time each cell of the grid, returning the timings, the work counters, the
    peak memory and the causal site counts that we got through before running
    out of time budget.
    """
    model = tstrait.trait_model(distribution="normal", mean=0, var=1)
    warm_up(ts, model, args.levels, args.counters)

    pool = causal_site_pool(ts, model, args.seed)
    if args.structure:
        report_structure(ts, pool, args)
        print()

    rows = []
    counts = {}
    memory = {}
    completed = []
    for num_causal in args.num_causal:
        _, times = time_call(
            functools.partial(
                tstrait.sim_trait,
                ts,
                model=model,
                num_causal=num_causal,
                random_seed=args.seed,
            ),
            args.replicates,
        )
        for replicate, seconds in enumerate(times):
            rows.append(("sim_trait", num_causal, "uniform", "", replicate, seconds))
        report(rows[-1])
        slowest = max(times)
        for selection in args.selections:
            rng = np.random.default_rng(args.seed)
            trait_df = select_causal(
                pool, selection, num_causal, args.rare_threshold, rng
            )
            if trait_df is None:
                print(f"  too few {selection} sites for num_causal={num_causal}")
                continue
            if root_runs_fired(ts, trait_df):
                print(
                    f"  {selection} num_causal={num_causal} takes the _root_runs "
                    "branch, a Python loop over every tree"
                )
            if args.counters:
                counts[(num_causal, selection)] = count_work(
                    ts, _check_trait_df(ts, trait_df)
                )
            for level in args.levels:
                call = functools.partial(
                    tstrait.genetic_value, ts, trait_df, level=level
                )
                if args.memory:
                    _, memory[(num_causal, selection, level)] = peak_memory(call)
                _, times = time_call(call, args.replicates)
                for replicate, seconds in enumerate(times):
                    rows.append(
                        (
                            "genetic_value",
                            num_causal,
                            selection,
                            level,
                            replicate,
                            seconds,
                        )
                    )
                report(rows[-1])
                slowest = max(slowest, max(times))
                if args.phases:
                    for phase, phase_times in time_phases(
                        ts, trait_df, level, args.replicates
                    ):
                        for replicate, seconds in enumerate(phase_times):
                            rows.append(
                                (phase, num_causal, selection, level, replicate, seconds)
                            )
                        report(rows[-1])
        completed.append(num_causal)
        # The cost grows with the number of causal sites, so once a single call
        # is over budget the next point on the grid is not worth waiting for.
        if slowest > args.max_seconds:
            skipped = args.num_causal[len(completed) :]
            if len(skipped) > 0:
                print(
                    f"  a call took {slowest:.1f}s, over the {args.max_seconds:g}s "
                    f"budget: skipping {skipped}"
                )
            break
    return rows, counts, memory, completed


def report(row):
    phase, num_causal, selection, level, _, seconds = row
    print(f"  {phase:<14} {num_causal:>7} {selection:<8} {level:<11} {seconds:8.3f}s")


def summarise(rows, counts, memory, completed, ts, args):
    """
    Print the minimum time over the replicates of each cell of the grid, along
    with the time per causal site and, where we counted the work, the time per
    trip of the kernel's innermost loop.
    """
    best = {}
    for phase, num_causal, selection, level, _, seconds in rows:
        key = (phase, selection, level, num_causal)
        best[key] = min(best.get(key, seconds), seconds)

    print(f"\n{describe(ts)}")
    print(f"Minimum of {args.replicates} replicates\n")
    columns = f"{'phase':<14} {'selection':<10} {'level':<11} {'num_causal':>10} "
    columns += f"{'seconds':>10} {'us/site':>10}"
    if args.counters:
        columns += f" {'ns/scan':>9}"
    print(columns)
    print("-" * len(columns))
    phases = ["sim_trait", "genetic_value"]
    if args.phases:
        phases += ["check", "setup", "kernel", "frame"]
    combinations = [("sim_trait", "uniform", "")]
    combinations += [
        (p, s, x)
        for p in phases
        if p != "sim_trait"
        for s in args.selections
        for x in args.levels
    ]
    for phase, selection, level in combinations:
        for num_causal in completed:
            key = (phase, selection, level, num_causal)
            if key not in best:
                continue
            seconds = best[key]
            line = (
                f"{phase:<14} {selection:<10} {level:<11} {num_causal:>10} "
                f"{seconds:>10.3f} {seconds / num_causal * 1e6:>10.1f}"
            )
            if args.counters:
                # Only the phases that run the sweep have a per trip cost.
                scans = counts.get((num_causal, selection), {}).get("edge_scans")
                sweeps = phase in ("genetic_value", "kernel")
                line += (
                    f" {seconds / scans * 1e9:>9.2f}"
                    if sweeps and scans is not None
                    else f" {'':>9}"
                )
            print(line)

    if args.counters:
        print()
        header = f"{'selection':<10} {'num_causal':>10} "
        header += " ".join(f"{name:>12}" for name in COUNTERS)
        header += f" {'scans/seed':>11} {'hit rate':>9} {'reached':>8}"
        print(header)
        print("-" * len(header))
        # The sweep visits every node whether or not anything reached it, and
        # builds a pending entry for every node before it starts, so a low
        # reached fraction is a kernel spending its time on the prologue.
        for selection in args.selections:
            for num_causal in completed:
                got = counts.get((num_causal, selection))
                if got is None:
                    continue
                line = f"{selection:<10} {num_causal:>10} "
                line += " ".join(f"{got[name]:>12,}" for name in COUNTERS)
                line += f" {got['edge_scans'] / max(got['seeds'], 1):>11.1f}"
                line += f" {got['edge_hits'] / max(got['edge_scans'], 1) * 100:>8.1f}%"
                line += f" {got['reached'] / ts.num_nodes * 100:>7.1f}%"
                print(line)

    if len(memory) > 0:
        print()
        header = f"{'selection':<10} {'level':<11} {'num_causal':>10} "
        header += f"{'peak RSS':>12} {'over base':>12}"
        print(header)
        print("-" * len(header))
        for (num_causal, selection, level), (peak, base) in memory.items():
            if peak is None:
                shown, added = "unavailable", ""
            else:
                # The peak is mostly the tree sequence and the interpreter, so
                # what the call added on top of them is the interesting number.
                shown = f"{peak / 1e9:.2f} GB"
                added = f"{(peak - base) / 1e9:.2f} GB"
            print(
                f"{selection:<10} {level:<11} {num_causal:>10} {shown:>12} {added:>12}"
            )


def write_csv(rows, ts, args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "phase",
                "num_causal",
                "selection",
                "level",
                "replicate",
                "seconds",
                "num_samples",
                "num_individuals",
                "num_nodes",
                "num_edges",
                "num_trees",
                "num_sites",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    *row,
                    ts.num_samples,
                    ts.num_individuals,
                    ts.num_nodes,
                    ts.num_edges,
                    ts.num_trees,
                    ts.num_sites,
                ]
            )
    print(f"\nWrote {args.output}")


def write_counts_csv(counts, ts, args):
    path = args.output.with_name(args.output.stem + "_counters.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_causal", "selection", *COUNTERS, "num_nodes", "num_edges"])
        for (num_causal, selection), got in counts.items():
            writer.writerow(
                [
                    num_causal,
                    selection,
                    *[got[name] for name in COUNTERS],
                    ts.num_nodes,
                    ts.num_edges,
                ]
            )
    print(f"Wrote {path}")


def write_memory_csv(memory, args):
    path = args.output.with_name(args.output.stem + "_memory.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["num_causal", "selection", "level", "peak_rss_bytes", "base_rss_bytes"]
        )
        for (num_causal, selection, level), (peak, base) in memory.items():
            writer.writerow([num_causal, selection, level, peak, base])
    print(f"Wrote {path}")


def parse_args():
    default_output = pathlib.Path(__file__).parent / "_output"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=list(PRESETS),
        default="small",
        help=(
            "Simulation size. Sets --samples, --length and --num-causal, each "
            "of which overrides it when given explicitly"
        ),
    )
    parser.add_argument("--samples", type=int, help="Number of sample nodes")
    parser.add_argument("--length", type=float, help="Sequence length")
    parser.add_argument("--recombination-rate", type=float, default=1e-8)
    parser.add_argument("--mutation-rate", type=float, default=1e-7)
    parser.add_argument("--population-size", type=float, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-causal", type=int, nargs="+")
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=LEVELS)
    parser.add_argument(
        "--selections", nargs="+", choices=SELECTIONS, default=SELECTIONS
    )
    parser.add_argument(
        "--rare-threshold",
        type=float,
        default=0.001,
        help="Allele frequency below which a causal site counts as rare",
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=60,
        help=(
            "Stop before the next number of causal sites once a single call has "
            "taken longer than this"
        ),
    )
    parser.add_argument(
        "--phases",
        action="store_true",
        help="Also time the check, setup, kernel and dataframe phases separately",
    )
    parser.add_argument(
        "--counters",
        action="store_true",
        help="Also count the work the kernel does, which perf cannot see inside",
    )
    parser.add_argument(
        "--structure",
        action="store_true",
        help="Report the shape of the tree sequence and the carrier fractions",
    )
    parser.add_argument(
        "--structure-sites",
        type=int,
        default=150,
        help="Causal sites to sample for the carrier fraction distribution",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Also measure the peak resident set size of each call",
    )
    parser.add_argument("--cache-dir", type=pathlib.Path, default=default_output)
    parser.add_argument(
        "--output", type=pathlib.Path, default=default_output / "genetic_value.csv"
    )
    args = parser.parse_args()
    for name, value in PRESETS[args.preset].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    ts = cached_simulation(args)
    print(describe(ts))
    max_causal = max(args.num_causal)
    if ts.num_sites < max_causal:
        raise ValueError(
            f"Only {ts.num_sites} sites in the tree sequence, but {max_causal} "
            "causal sites were requested. Increase --length or --mutation-rate."
        )
    rows, counts, memory, completed = run_benchmark(ts, args)
    summarise(rows, counts, memory, completed, ts, args)
    write_csv(rows, ts, args)
    if args.counters:
        write_counts_csv(counts, ts, args)
    if args.memory:
        write_memory_csv(memory, args)


if __name__ == "__main__":
    main()
