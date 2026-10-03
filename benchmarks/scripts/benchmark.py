#!/usr/bin/env python3
"""Run selected Tinyopt benchmark binaries and write comparison plots."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ("tinyopt", "ceres", "g2o", "gtsam")


def parse_results(xml_text: str, backend: str) -> list[dict[str, object]]:
    root = ET.fromstring(xml_text)
    records = []
    for test_case in root.findall(".//TestCase"):
        test_name = test_case.get("name", "")
        for benchmark in test_case.findall("BenchmarkResults"):
            mean = benchmark.find("mean")
            if mean is None or "value" not in mean.attrib:
                continue
            mean_ns = float(mean.attrib["value"])
            records.append(
                {
                    "backend": backend,
                    "test": test_name,
                    "benchmark": benchmark.get("name", ""),
                    "mean_ns": mean_ns,
                    "mean_us": mean_ns / 1000.0,
                }
            )
    if not records:
        raise ValueError(f"No benchmark means found for {backend}")
    return records


def run_backend(backend: str, build_dir: Path, results_dir: Path | None, samples: int,
                warmup_seconds: float) -> list[dict[str, object]]:
    executable = build_dir / "benchmarks" / backend / f"tinyopt_bench_{backend}"
    if not executable.is_file():
        raise FileNotFoundError(
            f"Missing {executable}. Build it with the matching Pixi benchmark task first."
        )

    command = [
        str(executable),
        "[benchmark]",
        "--benchmark-no-analysis",
        "--benchmark-samples",
        str(samples),
        "--benchmark-warmup-time",
        str(warmup_seconds),
        "--reporter",
        "xml",
    ]
    print(f"Running {backend} benchmarks")
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode:
        print(completed.stdout)
        print(completed.stderr)
        completed.check_returncode()
    xml_start = completed.stdout.find("<Catch2TestRun")
    xml_end = completed.stdout.find("</Catch2TestRun>", xml_start)
    if xml_start < 0 or xml_end < 0:
        raise ValueError(f"Catch2 XML reporter output was not found for {backend}")
    xml_text = completed.stdout[xml_start : xml_end + len("</Catch2TestRun>")]
    if results_dir is not None:
        (results_dir / f"{backend}.xml").write_text(xml_text, encoding="utf-8")
    return parse_results(xml_text, backend)


def benchmark_key(record: dict[str, object]) -> tuple[str, str]:
    return str(record["test"]), str(record["benchmark"])


def dense_dimension(test: str, benchmark: str) -> int:
    label = f"{test} {benchmark}"
    vector_dimension = re.search(r"Vec(\d+)", label)
    if vector_dimension:
        return int(vector_dimension.group(1))
    benchmark_dimension = re.search(r"Prior\s+(\d+)", benchmark)
    if benchmark_dimension:
        return int(benchmark_dimension.group(1))
    if "VecXf" in test:
        return 10
    if test.lower() in {"float", "double"}:
        return 1
    return 0


def dense_group(test: str) -> str:
    return "dynamic" if "VecX" in test else "fixed"


def dense_label(test: str, benchmark: str) -> str:
    if test.lower() in {"float", "double"}:
        return f"{test} scalar: {benchmark}"
    dimension = dense_dimension(test, benchmark)
    scalar_type = "float" if "f" in test.rsplit("Vec", 1)[-1] else "double"
    vector_type = "dynamic" if dense_group(test) == "dynamic" else "fixed"
    return f"{dimension}D {vector_type} {scalar_type}: {benchmark}"


def save_csv(records: list[dict[str, object]], path: Path) -> None:
    fieldnames = ("backend", "test", "benchmark", "mean_ns", "mean_us")
    with path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot_runtime_panels(records: list[dict[str, object]], backends: list[str],
                        output: Path | None) -> None:
    panel_filters = (
        ("Dense systems - fixed vectors", lambda test, benchmark:
         (test.lower().startswith("dense") or test.lower() in {"double", "float"}) and
         dense_group(test) == "fixed"),
        ("Dense systems - dynamic vectors", lambda test, benchmark:
         (test.lower().startswith("dense") or test.lower() in {"double", "float"}) and
         dense_group(test) == "dynamic"),
        ("Sparse systems", lambda test, _benchmark: "sparse" in test.lower()),
        ("Bundle adjustment", lambda test, _benchmark: "ba" in test.lower()),
    )
    values = {
        (benchmark_key(record), str(record["backend"])): float(record["mean_us"])
        for record in records
    }
    grouped_keys = [
        sorted({benchmark_key(record) for record in records
                if test_filter(str(record["test"]), str(record["benchmark"]))})
        for _, test_filter in panel_filters
    ]
    for keys in grouped_keys[:2]:
        keys.sort(key=lambda key: (
            dense_dimension(*key),
            0 if "[AD]" in key[1] else 1,
            key,
        ))
    max_rows = max((len(keys) for keys in grouped_keys), default=1)
    figure, axes = plt.subplots(1, 4, figsize=(27, max(7, max_rows * 0.34 + 2)))

    for axis, (title, _), keys in zip(axes, panel_filters, grouped_keys):
        if not keys:
            axis.set_visible(False)
            continue
        present_backends = [
            backend for backend in backends
            if any((key, backend) in values for key in keys)
        ]
        bar_height = 0.78 / max(len(present_backends), 1)
        centers = list(range(len(keys)))
        for backend_index, backend in enumerate(present_backends):
            offsets = [center - 0.39 + bar_height * (backend_index + 0.5)
                       for center in centers]
            runtimes = [values.get((key, backend), float("nan")) for key in keys]
            axis.barh(offsets, runtimes, height=bar_height * 0.9, label=backend)
        axis.set_xscale("log")
        axis.set_xlabel("Mean runtime (us, log scale)")
        axis.set_title(title)
        axis.set_yticks(centers)
        labels = [dense_label(test, benchmark) if title.startswith("Dense systems")
                  else f"{test}: {benchmark}" for test, benchmark in keys]
        axis.set_yticklabels(labels, fontsize=7)
        axis.invert_yaxis()
        axis.grid(axis="x", which="both", alpha=0.2)

    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=len(labels),
                      bbox_to_anchor=(0.5, 1.0))
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    if output is not None:
        figure.savefig(output, dpi=160)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build-bench"))
    parser.add_argument("--output-dir", default="",
                        help="Save XML, CSV, and plots to this folder; empty disables saving.")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Plot existing backend XML files instead of running benchmarks.")
    parser.add_argument("--plot", action="store_true", help="Display plots after the run.")
    parser.add_argument("--only", nargs="+", choices=BACKENDS, default=None,
                        help="Run only the selected backend(s); defaults to all four.")
    parser.add_argument("--samples", type=int, default=10,
                        help="Catch2 benchmark samples per case.")
    parser.add_argument("--warmup-seconds", type=float, default=0.1)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup_seconds < 0:
        parser.error("--samples must be positive and --warmup-seconds cannot be negative")

    build_dir = (ROOT / args.build_dir).resolve() if not args.build_dir.is_absolute() else args.build_dir
    output_dir = None
    results_dir = None
    if args.output_dir.strip():
        requested_dir = Path(args.output_dir)
        output_dir = (ROOT / requested_dir).resolve() if not requested_dir.is_absolute() else requested_dir
        results_dir = output_dir / "benchmark-results"
        results_dir.mkdir(parents=True, exist_ok=True)
    selected = args.only or list(BACKENDS)

    records = []
    if args.input_dir is None:
        for backend in selected:
            records.extend(run_backend(backend, build_dir, results_dir, args.samples,
                                      args.warmup_seconds))
    else:
        input_dir = args.input_dir
        if not input_dir.is_absolute():
            input_dir = ROOT / input_dir
        for backend in selected:
            xml_path = input_dir / f"{backend}.xml"
            xml_text = xml_path.read_text(encoding="utf-8")
            records.extend(parse_results(xml_text, backend))
            if results_dir is not None:
                (results_dir / xml_path.name).write_text(xml_text, encoding="utf-8")

    if output_dir is not None:
        save_csv(records, output_dir / "benchmark-results.csv")
    if args.plot or output_dir is not None:
        plot_runtime_panels(
            records, selected,
            output_dir / "benchmark-runtime.png" if output_dir is not None else None,
        )
        if args.plot:
            plt.show()
        plt.close("all")
    if output_dir is None:
        print("Finished without saving benchmark files.")
    else:
        print(f"Saved XML, CSV, and plots under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())