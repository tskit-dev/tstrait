"""
Benchmark tstrait.genetic_value as the number of causal sites grows.

The interesting regime is a trait with a large number of weakly causal sites,
i.e. many causal sites that are each carried by a small number of samples.
Causal sites are chosen uniformly at random over all sites, and because the
site frequency spectrum is dominated by rare variants this puts us in that
regime by default.

Run with ``uv run --group test benchmarks/benchmark_genetic_value.py``.
"""

import argparse
import csv
import functools
import pathlib
import sys
import time

import msprime
import numpy as np
import tskit

import tstrait

DEFAULT_NUM_CAUSAL = [1, 100, 1000, 10_000, 100_000]
LEVELS = ["individual", "node", "edge"]
SELECTIONS = ["uniform", "rare"]


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
    """
    times = []
    for _ in range(replicates):
        before = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - before)
    return result, times


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
    differently from the weakly causal sites the ARG descent is aimed at.
    """
    if selection == "rare":
        pool = pool[pool["allele_freq"] < rare_threshold]
    if len(pool) < num_causal:
        return None
    keep = rng.choice(len(pool), size=num_causal, replace=False)
    return pool.iloc[np.sort(keep)]


def warm_up(ts, model, levels):
    """
    Run the smallest possible simulation through each code path so that the
    numba kernels are compiled before we start timing.
    """
    before = time.perf_counter()
    trait_df = tstrait.sim_trait(ts, model=model, num_causal=1, random_seed=1)
    for level in levels:
        tstrait.genetic_value(ts, trait_df, level=level)
    print(f"Warm up (includes numba compilation): {time.perf_counter() - before:.1f}s")


def run_benchmark(ts, args):
    """
    Time each cell of the grid, returning the timings and the causal site
    counts that we got through before running out of time budget.
    """
    model = tstrait.trait_model(distribution="normal", mean=0, var=1)
    warm_up(ts, model, args.levels)

    pool = causal_site_pool(ts, model, args.seed)
    rows = []
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
            for level in args.levels:
                _, times = time_call(
                    functools.partial(tstrait.genetic_value, ts, trait_df, level=level),
                    args.replicates,
                )
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
        completed.append(num_causal)
        # The cost is superlinear in num_causal, so once a single call is over
        # budget the next point on the grid is not worth waiting for.
        if slowest > args.max_seconds:
            skipped = args.num_causal[len(completed) :]
            if len(skipped) > 0:
                print(
                    f"  a call took {slowest:.1f}s, over the {args.max_seconds:g}s "
                    f"budget: skipping {skipped}"
                )
            break
    return rows, completed


def report(row):
    phase, num_causal, selection, level, _, seconds = row
    print(f"  {phase:<14} {num_causal:>7} {selection:<8} {level:<11} {seconds:8.3f}s")


def summarise(rows, completed, ts, args):
    """
    Print the minimum time over the replicates of each cell of the grid, along
    with the time per causal site.
    """
    best = {}
    for phase, num_causal, selection, level, _, seconds in rows:
        key = (phase, selection, level, num_causal)
        best[key] = min(best.get(key, seconds), seconds)

    print(f"\n{describe(ts)}")
    print(f"Minimum of {args.replicates} replicates\n")
    header = (
        f"{'phase':<14} {'selection':<10} {'level':<11} {'num_causal':>10} "
        f"{'seconds':>10} {'us/site':>10}"
    )
    print(header)
    print("-" * len(header))
    combinations = [("sim_trait", "uniform", "")]
    combinations += [
        ("genetic_value", s, x) for s in args.selections for x in args.levels
    ]
    for phase, selection, level in combinations:
        for num_causal in completed:
            key = (phase, selection, level, num_causal)
            if key not in best:
                continue
            seconds = best[key]
            print(
                f"{phase:<14} {selection:<10} {level:<11} {num_causal:>10} "
                f"{seconds:>10.3f} {seconds / num_causal * 1e6:>10.1f}"
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


def parse_args():
    default_output = pathlib.Path(__file__).parent / "_output"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=int, default=100_000, help="Number of sample nodes"
    )
    parser.add_argument("--length", type=float, default=5e6, help="Sequence length")
    parser.add_argument("--recombination-rate", type=float, default=1e-8)
    parser.add_argument("--mutation-rate", type=float, default=1e-7)
    parser.add_argument("--population-size", type=float, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-causal", type=int, nargs="+", default=DEFAULT_NUM_CAUSAL)
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
    parser.add_argument("--cache-dir", type=pathlib.Path, default=default_output)
    parser.add_argument(
        "--output", type=pathlib.Path, default=default_output / "genetic_value.csv"
    )
    return parser.parse_args()


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
    rows, completed = run_benchmark(ts, args)
    summarise(rows, completed, ts, args)
    write_csv(rows, ts, args)


if __name__ == "__main__":
    main()
