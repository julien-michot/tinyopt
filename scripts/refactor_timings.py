#!/usr/bin/env python3
"""Measure minimal Optimize() compilation and non-Ceres benchmark runtime."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TIMINGS_DIR = ROOT / "tmp" / "timings"
DEV_RESULT = TIMINGS_DIR / "dev.json"
BENCHMARK_RUNS = 3
CATCH2_BENCHMARK_SAMPLES = 10
NEUTRAL_PERCENT = 5.0


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def pixi(*arguments: str) -> list[str]:
    return ["pixi", "run", *arguments]


def current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def current_commit_datetime() -> str:
    return subprocess.run(
        ["git", "show", "-s", "--format=%cI", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def host_architecture() -> str:
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.arm64"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and result.stdout.strip() == "1":
            return "arm64"
    return platform.machine()


def target_architecture() -> str:
    return os.environ.get("CMAKE_OSX_ARCHITECTURES") or host_architecture()


def cmake_architecture_args() -> list[str]:
    if platform.system() == "Darwin":
        return [f"-DCMAKE_OSX_ARCHITECTURES={target_architecture()}"]
    return []


def benchmark_mean_sum_seconds(xml_path: Path) -> float:
    root = ET.parse(xml_path).getroot()
    means = [
        float(mean.attrib["value"])
        for mean in root.findall(".//BenchmarkResults/mean")
        if "value" in mean.attrib
    ]
    if not means:
        raise ValueError(f"No Catch2 benchmark means found in {xml_path}")
    return sum(means) / 1_000_000_000


def finalize_results() -> int:
    if not DEV_RESULT.is_file():
        print("No tmp/timings/dev.json result to finalize.", file=sys.stderr)
        return 1

    commit = current_commit()
    destination = TIMINGS_DIR / f"{commit}.json"
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
            current = json.loads(DEV_RESULT.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            print(f"Cannot inspect existing baseline: {error}", file=sys.stderr)
            return 1
        existing_metrics = existing.get("metrics", {})
        current_metrics = current.get("metrics", {})
        if "benchmark_mean_sum_seconds" in existing_metrics:
            print(f"Refusing to overwrite existing baseline: {destination}", file=sys.stderr)
            return 1
        if "benchmark_mean_sum_seconds" not in current_metrics:
            print("Current results do not contain XML benchmark means.", file=sys.stderr)
            return 1
        print(f"Replacing legacy wall-time baseline at {destination} with XML-based metrics.")

    try:
        result = json.loads(DEV_RESULT.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("expected a JSON object")
        result["commit"] = commit
        result["commit_datetime"] = current_commit_datetime()
        DEV_RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (json.JSONDecodeError, OSError, ValueError) as error:
        print(f"Cannot finalize timing results: {error}", file=sys.stderr)
        return 1

    DEV_RESULT.rename(destination)
    print(f"Finalized timing results as {destination.relative_to(ROOT)}")
    return 0


def measure() -> dict[str, object]:
    print("\nCleaning and configuring the test build...")
    run(pixi("-e", "tests", "clean"))
    run(
        pixi(
            "-e",
            "tests",
            "cmake",
            "-B",
            "build-tests",
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DTINYOPT_BUILD_TESTS=ON",
            "-DTINYOPT_BUILD_BENCHMARKS=ON",
            *cmake_architecture_args(),
        )
    )

    print("\nMeasuring minimal Optimize() test compilation...")
    start = time.perf_counter()
    run(
        pixi(
            "-e",
            "tests",
            "cmake",
            "--build",
            "build-tests",
            "--target",
            "tinyopt_bench_simple_optimize",
        )
    )
    optimize_compile_seconds = time.perf_counter() - start

    print("\nBuilding the Tinyopt benchmark executable...")
    run(pixi("-e", "bench", "clean"))
    run(
        pixi(
            "-e",
            "bench",
            "cmake",
            "-B",
            "build-bench",
            "-G",
            "Ninja",
            "-DTINYOPT_BUILD_BENCHMARKS=ON",
            "-DTINYOPT_BUILD_CERES=OFF",
        )
    )
    run(
        pixi(
            "-e",
            "bench",
            "cmake",
            "--build",
            "build-bench",
            "--target",
            "tinyopt_bench_tinyopt",
        )
    )

    benchmark_samples_seconds: list[float] = []
    benchmark_reports: list[str] = []
    print(
        f"\nMeasuring tinyopt benchmarks ({BENCHMARK_RUNS} runs, "
        f"{CATCH2_BENCHMARK_SAMPLES} Catch2 samples each)..."
    )
    for sample in range(BENCHMARK_RUNS):
        print(f"Benchmark run {sample + 1}/{BENCHMARK_RUNS}")
        report_path = TIMINGS_DIR / f"benchmark-{current_commit()}-{sample + 1}.xml"
        run(
            pixi(
                "-e",
                "bench",
                "./build-bench/benchmarks/tinyopt/tinyopt_bench_tinyopt",
                "[benchmark]",
                "--benchmark-no-analysis",
                "--benchmark-samples",
                str(CATCH2_BENCHMARK_SAMPLES),
                "--reporter",
                f"XML::out={report_path}",
            )
        )
        benchmark_reports.append(report_path.name)
        benchmark_samples_seconds.append(benchmark_mean_sum_seconds(report_path))

    return {
        "commit": current_commit(),
        "commit_datetime": current_commit_datetime(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "architecture": target_architecture(),
        "metrics": {
            "benchmark_metric": "sum of Catch2 XML mean values",
            "optimize_compile_seconds": optimize_compile_seconds,
            "benchmark_mean_sum_seconds": statistics.median(benchmark_samples_seconds),
            "benchmark_runs_mean_sum_seconds": benchmark_samples_seconds,
            "benchmark_reports": benchmark_reports,
        },
    }


def load_result(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or not isinstance(result.get("metrics"), dict):
            raise ValueError("missing metrics object")
        return result
    except (json.JSONDecodeError, OSError, ValueError) as error:
        print(f"Cannot read timing results from {path}: {error}", file=sys.stderr)
        return None


def format_seconds(seconds: float | None) -> str:
    return "n/a" if seconds is None else f"{seconds:.2f} s"


def comparison(before: float | None, after: float) -> tuple[str, str]:
    if before is None:
        return "n/a", "no baseline"
    delta = after - before
    percent = 0.0 if before == 0 else delta / before * 100.0
    status = "neutral" if abs(percent) <= NEUTRAL_PERCENT else "slower" if delta > 0 else "faster"
    return f"{delta:+.2f} s ({percent:+.1f}%)", status


def print_comparison(baseline: dict[str, object] | None, result: dict[str, object]) -> None:
    architecture_mismatch = False
    if baseline and baseline.get("architecture") != result.get("architecture"):
        print(
            "Skipping timing comparison: baseline architecture "
            f"{baseline.get('architecture', 'unknown')} does not match current "
            f"{result.get('architecture', 'unknown')}."
        )
        baseline = None
        architecture_mismatch = True

    before_metrics = baseline.get("metrics", {}) if baseline else {}
    after_metrics = result["metrics"]
    assert isinstance(before_metrics, dict)
    assert isinstance(after_metrics, dict)

    baseline_name = str(baseline.get("commit", "HEAD")) if baseline else "missing"
    print(f"\nComparison: baseline {baseline_name} vs current dev")
    print("| Metric | Before | After | Delta | Status |")
    print("| --- | ---: | ---: | ---: | --- |")
    rows = (
        ("Minimal Optimize() test compilation", "optimize_compile_seconds"),
        ("Tinyopt benchmark mean sum (median)", "benchmark_mean_sum_seconds"),
    )
    for label, metric in rows:
        before = before_metrics.get(metric)
        after = after_metrics.get(metric)
        before_value = float(before) if isinstance(before, (float, int)) else None
        after_value = float(after) if isinstance(after, (float, int)) else None
        if after_value is None:
            print(f"| {label} | {format_seconds(before_value)} | n/a | n/a | missing result |")
            continue
        delta, status = comparison(before_value, after_value)
        print(
            f"| {label} | {format_seconds(before_value)} | "
            f"{format_seconds(after_value)} | {delta} | {status} |"
        )

    if baseline is None and not architecture_mismatch:
        print(
            f"No baseline exists for HEAD {current_commit()}. Capture the current baseline with "
            "`pixi run refactor-timings --finalize` before starting the experiment."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="rename tmp/timings/dev.json to the current HEAD short hash",
    )
    arguments = parser.parse_args()

    if arguments.finalize:
        return finalize_results()

    TIMINGS_DIR.mkdir(parents=True, exist_ok=True)
    baseline = load_result(TIMINGS_DIR / f"{current_commit()}.json")
    result = measure()
    DEV_RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved current measurements to {DEV_RESULT.relative_to(ROOT)}")
    print_comparison(baseline, result)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode or 1)
