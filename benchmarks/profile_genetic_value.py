"""
Profile a single cell of the genetic_value benchmark grid.

The benchmark says how long a call takes; this says where the time goes. Two
modes, because the Python setup and the numba kernel need different tools:

``--mode python``
    cProfile around the public ``genetic_value`` call. This is the only way to
    see the setup, which is a flat per call cost dominated by
    ``tskit.jit.numba.jitwrap``, but the kernel appears in it as a single
    opaque dispatcher frame.

``--mode kernel``
    Set up one cell and run only the kernel, so that a sampling profile is not
    swamped by the effect size simulation over the whole site pool. Run it
    under perf; ``--mode kernel`` on its own prints the commands.

Two things about perf and numba are worth knowing before reading a profile.

perf cannot attribute time to source lines inside the kernel here: this
llvmlite has no LLVM PerfJITEventListener, so nothing emits a
/tmp/perf-<pid>.map or a jitdump and the kernel shows up as raw addresses
under ``[JIT]``. What perf does give is the split between the kernel, the
numba runtime, the interpreter and LLVM compilation, with the typed list
runtime functions resolved by name. For attribution inside the kernel use
``benchmark_genetic_value.py --counters``.

NUMBA_ENABLE_PROFILING=1 is the documented way to profile numba and is the
wrong thing to use here. It would only help by way of the listener llvmlite
does not have, and it defaults NUMBA_DEBUGINFO to 1, which changes the code
that is generated: the kernel measured 2.62s a call with it and 1.61s
without. perf finds the JIT mappings on its own, so the recipe below does not
set it, and a profile taken with it is a profile of a different kernel.

Run with ``uv run --group test benchmarks/profile_genetic_value.py``.
"""

import argparse
import cProfile
import pathlib
import pstats
import sys
import time

import numpy as np
from benchmark_genetic_value import (
    LEVELS,
    PRESETS,
    SELECTIONS,
    cached_simulation,
    causal_site_pool,
    describe,
    select_causal,
)

import tstrait
from tstrait import jit
from tstrait.genetic_value import _check_trait_df, _GeneticValue

PERF_RECORD = """\
sudo perf record -F 999 -g -o {data} -- \\
    {python} {script} {arguments} --run"""

PERF_REPORT = """\
sudo perf report -i {data} --stdio --no-children -g none --sort dso
sudo perf report -i {data} --stdio --no-children -g none \\
    --dsos={helper} --percent-limit 0.01"""


def prepare(args):
    """
    Return the tree sequence and the trait dataframe for one cell of the grid.
    """
    ts = cached_simulation(args)
    print(describe(ts), file=sys.stderr)
    model = tstrait.trait_model(distribution="normal", mean=0, var=1)
    pool = causal_site_pool(ts, model, args.seed)
    rng = np.random.default_rng(args.seed)
    trait_df = select_causal(
        pool, args.selection, args.num_causal, args.rare_threshold, rng
    )
    if trait_df is None:
        raise ValueError(
            f"Too few {args.selection} sites for num_causal={args.num_causal}"
        )
    return ts, trait_df


def profile_python(args):
    """
    cProfile the public call, which is where the setup path is visible.
    """
    ts, trait_df = prepare(args)
    # Neither kernel is compiled with cache=True, so without this the profile
    # is mostly LLVM. The lazily built tskit mutation state arrays are cached
    # on the tree sequence by the same call.
    tstrait.genetic_value(ts, trait_df, level=args.level)

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(args.repeats):
        tstrait.genetic_value(ts, trait_df, level=args.level)
    profiler.disable()
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(args.lines)


def run_kernel(args):
    """
    Run only the kernel, the loop a sampling profiler should be pointed at.
    """
    ts, trait_df = prepare(args)
    genetic = _GeneticValue(ts, _check_trait_df(ts, trait_df))
    size = genetic._output_size(args.level)
    arguments = genetic._descent_arguments(args.level, 0, np.zeros(size))
    jit._push_down_arg(**{**arguments, "output": np.zeros(size)})

    before = time.perf_counter()
    for _ in range(args.repeats):
        jit._push_down_arg(**{**arguments, "output": np.zeros(size)})
    elapsed = time.perf_counter() - before
    print(
        f"{args.repeats} kernel calls in {elapsed:.3f}s "
        f"({elapsed / args.repeats:.3f}s each)",
        file=sys.stderr,
    )


def print_perf_commands(args):
    """
    Print the perf invocation for this cell rather than running it, since it
    needs sudo and writing out the command is more useful than hiding it.
    """
    script = pathlib.Path(__file__).resolve()
    arguments = (
        f"--preset {args.preset} --num-causal {args.num_causal} "
        f"--selection {args.selection} --level {args.level} "
        f"--repeats {args.repeats} --mode kernel"
    )
    print(
        f"# {args.repeats} kernel calls, against a setup of a few seconds that "
        "is in the\n# profile too. Raise --repeats until the [JIT] share stops "
        "moving.\n"
    )
    print("# Record:")
    print(
        PERF_RECORD.format(
            data=args.perf_data,
            python=sys.executable,
            script=script,
            arguments=arguments,
        )
    )
    print("\n# Report, first the split by shared object, then the numba runtime:")
    print(
        PERF_REPORT.format(
            data=args.perf_data,
            helper="_helperlib.cpython-*.so",
        )
    )
    print(
        "\n# The kernel is the [JIT] rows, and libllvmlite is compilation, not\n"
        "# work. numba_list_append and numba_list_resize in the second report\n"
        "# are the typed list traffic. Source lines inside the kernel are not\n"
        "# available; use --counters on the benchmark instead."
    )


def parse_args():
    default_output = pathlib.Path(__file__).parent / "_output"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=list(PRESETS), default="small")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--length", type=float)
    parser.add_argument("--recombination-rate", type=float, default=1e-8)
    parser.add_argument("--mutation-rate", type=float, default=1e-7)
    parser.add_argument("--population-size", type=float, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-causal", type=int, default=10_000)
    parser.add_argument("--selection", choices=SELECTIONS, default="uniform")
    parser.add_argument("--level", choices=LEVELS, default="node")
    parser.add_argument("--rare-threshold", type=float, default=0.001)
    parser.add_argument(
        "--repeats",
        type=int,
        help="Calls to profile. Defaults to 3 for --mode python, 10 for kernel",
    )
    parser.add_argument(
        "--mode",
        choices=["python", "kernel"],
        default="python",
        help="Profile the Python setup path, or run just the kernel for perf",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="With --mode kernel, run it rather than printing the perf commands",
    )
    parser.add_argument(
        "--lines", type=int, default=30, help="Rows of cProfile output to print"
    )
    parser.add_argument("--cache-dir", type=pathlib.Path, default=default_output)
    parser.add_argument(
        "--perf-data", type=pathlib.Path, default=default_output / "perf.data"
    )
    args = parser.parse_args()
    if args.repeats is None:
        args.repeats = 3 if args.mode == "python" else 10
    for name, value in PRESETS[args.preset].items():
        if name != "num_causal" and getattr(args, name) is None:
            setattr(args, name, value)
    return args


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if args.mode == "python":
        profile_python(args)
    elif args.run:
        run_kernel(args)
    else:
        print_perf_commands(args)


if __name__ == "__main__":
    main()
