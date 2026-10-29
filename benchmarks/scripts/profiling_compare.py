#!/usr/bin/env python3
"""Compare Tinyopt micro-profiling workloads between two revisions.

Builds the *same* profiling sources (the ones in this checkout) against the
headers of two revisions, then reports the per-run time ratio. This isolates
library changes from benchmark/options drift, which is easy to hit when each
revision is built with its own `benchmarks/common/options.h`.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILING_DIR = ROOT / "benchmarks" / "tinyopt" / "profiling"
BENCH_LINE = re.compile(r"^BENCH\s+(?P<body>.+)$")
NEUTRAL_PERCENT = 5.0
WORKLOADS = {
    "1d": "one_dimensional.cpp",
    "2d": "two_dimensional.cpp",
    "6d": "six_dimensional.cpp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, help="baseline git revision")
    parser.add_argument("--after", default="HEAD", help="candidate git revision (default: HEAD)")
    parser.add_argument(
        "--workload",
        action="append",
        choices=(*WORKLOADS, "all"),
        default=None,
        help="workload to compare (repeatable, default: all)",
    )
    parser.add_argument("--runs", type=int, default=100000, help="optimization calls per sample")
    parser.add_argument("--samples", type=int, default=5, help="samples per binary invocation")
    parser.add_argument("--repeats", type=int, default=3, help="binary invocations per revision")
    parser.add_argument(
        "--neutral-percent",
        type=float,
        default=NEUTRAL_PERCENT,
        help="changes within this percentage are reported as neutral",
    )
    parser.add_argument(
        "--keep-worktrees", action="store_true", help="keep the temporary git worktrees"
    )
    arguments = parser.parse_args()
    if arguments.runs < 1:
        parser.error("--runs must be positive")
    if arguments.samples < 1:
        parser.error("--samples must be positive")
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    if arguments.workload is None:
        arguments.workload = ["all"]
    return arguments


def git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def compiler() -> str:
    candidate = ROOT / ".pixi" / "envs" / "bench" / "bin" / "c++"
    if candidate.is_file():
        return str(candidate)
    return os.environ.get("CXX", "c++")


def eigen_include() -> str | None:
    candidate = ROOT / ".pixi" / "envs" / "bench" / "include" / "eigen3"
    if candidate.is_dir():
        return str(candidate)
    return os.environ.get("EIGEN3_INCLUDE_DIR") or None


def architecture_flags() -> list[str]:
    if platform.system() != "Darwin":
        return []
    machine = platform.machine()
    architecture = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
    return ["-arch", architecture]


def common_compile_flags() -> list[str]:
    flags = [
        "-O3",
        "-DNDEBUG",
        "-std=c++20",
        "-fPIC",
        "-w",
        "-DHAS_EIGEN",
        "-DTINYOPT_ENABLE_LINEAR_SOLVER_LDLT=1",
        "-DTINYOPT_ENABLE_LINEAR_SOLVER_INCOMPLETE_CHOLESKY=1",
    ]
    flags.extend(architecture_flags())
    return flags


def build(worktree: Path, source: Path, output: Path) -> None:
    command = [compiler(), *common_compile_flags()]
    command.extend(["-I", str(ROOT / "benchmarks" / "common")])
    command.extend(["-I", str(ROOT / "benchmarks" / "tinyopt")])
    command.extend(["-I", str(worktree / "include")])
    eigen = eigen_include()
    if eigen:
        command.extend(["-isystem", eigen])
    command.extend([str(source), "-o", str(output)])
    subprocess.run(command, cwd=ROOT, check=True)


def parse_bench_line(output: str) -> dict[str, float]:
    for line in output.splitlines():
        match = BENCH_LINE.match(line.strip())
        if match is None:
            continue
        values: dict[str, float] = {}
        for token in match.group("body").split():
            key, _, value = token.partition("=")
            try:
                values[key] = float(value)
            except ValueError:
                continue
        if "median_us" in values:
            return values
    raise ValueError(f"No BENCH line found in:\n{output}")


def run_binary(binary: Path, runs: int, samples: int) -> dict[str, float]:
    command = [str(binary), "--runs", str(runs), "--samples", str(samples)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return parse_bench_line(result.stdout)


def format_change(ratio: float, neutral_percent: float) -> str:
    percent = (ratio - 1.0) * 100.0
    if abs(percent) <= neutral_percent:
        return f"{percent:+.1f}% (neutral)"
    if percent < 0:
        return f"{percent:+.1f}% (faster)"
    return f"{percent:+.1f}% (SLOWER)"


def main() -> int:
    arguments = parse_args()
    workloads = list(WORKLOADS)
    if "all" not in arguments.workload:
        workloads = [name for name in workloads if name in arguments.workload]

    temp_root = Path(tempfile.mkdtemp(prefix="tinyopt-profiling-compare-"))
    created: list[Path] = []
    try:
        before_rev = git("rev-parse", "--short", arguments.before)
        after_rev = git("rev-parse", "--short", arguments.after)

        binaries: dict[str, dict[str, Path]] = {"before": {}, "after": {}}
        for label, revision in (("before", before_rev), ("after", after_rev)):
            worktree = temp_root / f"wt-{label}"
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree), arguments.before
                 if label == "before" else arguments.after],
                cwd=ROOT,
                check=True,
                capture_output=True,
            )
            created.append(worktree)
            for workload in workloads:
                source = PROFILING_DIR / WORKLOADS[workload]
                binary = temp_root / f"{label}-{workload}"
                build(worktree, source, binary)
                binaries[label][workload] = binary

        print(
            f"Comparing {before_rev} (before) -> {after_rev} (after); "
            f"runs={arguments.runs} samples={arguments.samples} repeats={arguments.repeats}"
        )
        print(f"{'Workload':<10} {'Before (us/run)':>16} {'After (us/run)':>16} {'Change':>24}")
        print("-" * 72)

        regressions = 0
        for workload in workloads:
            times: dict[str, float] = {}
            for label in ("before", "after"):
                best = float("inf")
                for _ in range(arguments.repeats):
                    values = run_binary(
                        binaries[label][workload], arguments.runs, arguments.samples
                    )
                    # `median_us` is already robust; keep the best invocation to
                    # reduce sensitivity to background load on shared machines.
                    best = min(best, values["median_us"])
                times[label] = best

            ratio = times["after"] / times["before"] if times["before"] > 0 else float("inf")
            change = format_change(ratio, arguments.neutral_percent)
            if "SLOWER" in change:
                regressions += 1
            print(
                f"{workload:<10} {times['before']:>16.3f} "
                f"{times['after']:>16.3f} {change:>24}"
            )

        print("-" * 72)
        if regressions:
            print(f"{regressions} workload(s) slower than {arguments.neutral_percent:.0f}%.")
            return 1
        print(f"No workload slower than {arguments.neutral_percent:.0f}%.")
        return 0
    finally:
        if not arguments.keep_worktrees:
            for worktree in created:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(worktree)],
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                )
            shutil.rmtree(temp_root, ignore_errors=True)
        else:
            print(f"Kept temporary worktrees and binaries in {temp_root}")


if __name__ == "__main__":
    sys.exit(main())
