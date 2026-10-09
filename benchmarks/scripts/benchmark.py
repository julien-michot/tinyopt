#!/usr/bin/env python3
"""Run benchmark suites and write plots, timing tables, and one HTML report."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import math
import platform
import re
import subprocess
import webbrowser
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ("tinyopt", "ceres", "ceres-tinysolver", "g2o", "gtsam")
ITERATION_LINE = re.compile(
    r"^\[iterations\]\s+(.+?)\s+\|\s+(.+?)\s+\|\s+(\S+)\s+\|\s+(\d+)\s+\|\s+(ok|no)\s*$",
    re.M,
)
CACHE_KEYS = (
    "CMAKE_BUILD_TYPE",
    "CMAKE_CXX_COMPILER",
    "CMAKE_CXX_COMPILER_ID",
    "CMAKE_CXX_COMPILER_VERSION",
    "CMAKE_CXX_FLAGS",
    "CMAKE_CXX_FLAGS_RELEASE",
    "CMAKE_GENERATOR",
    "CMAKE_OSX_ARCHITECTURES",
    "TINYOPT_BUILD_CERES",
    "TINYOPT_BUILD_BENCHMARKS",
    "TINYOPT_BUILD_G2O_BENCHMARKS",
    "TINYOPT_BUILD_GTSAM_BENCHMARKS",
    "TINYOPT_BUILD_TESTS",
    "TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS",
)


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
                    "tags": test_case.get("tags", ""),
                    "mean_ns": mean_ns,
                    "mean_us": mean_ns / 1000.0,
                }
            )
    if not records:
        raise ValueError(f"No benchmark means found for {backend}")
    return records


def parse_iterations(text: str) -> list[dict[str, object]]:
    return [
        {
            "category": match.group(1),
            "problem": match.group(2),
            "backend": match.group(3),
            "iterations": int(match.group(4)),
            "converged": match.group(5) == "ok",
        }
        for match in ITERATION_LINE.finditer(text)
    ]


def run_backend(backend: str, build_dir: Path, results_dir: Path | None, samples: int,
                warmup_seconds: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    binary_name = backend.replace("-", "_")
    executable = (
        build_dir / "benchmarks" / "ceres" / "tinyopt_bench_ceres_tinysolver"
        if backend == "ceres-tinysolver"
        else build_dir / "benchmarks" / backend / f"tinyopt_bench_{binary_name}"
    )
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

    iteration_records = parse_iterations(completed.stdout + "\n" + completed.stderr)
    xml_start = completed.stdout.find("<Catch2TestRun")
    xml_end = completed.stdout.find("</Catch2TestRun>", xml_start)
    if xml_start < 0 or xml_end < 0:
        raise ValueError(f"Catch2 XML reporter output was not found for {backend}")
    xml_text = completed.stdout[xml_start : xml_end + len("</Catch2TestRun>")]
    if results_dir is not None:
        (results_dir / f"{backend}.xml").write_text(xml_text, encoding="utf-8")
    return parse_results(xml_text, backend), iteration_records


def save_csv(records: list[dict[str, object]], path: Path) -> None:
    fieldnames = ("backend", "test", "tags", "benchmark", "mean_ns", "mean_us")
    with path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_iteration_csv(records: list[dict[str, object]], path: Path) -> None:
    fieldnames = ("category", "problem", "backend", "iterations", "converged")
    with path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def normalize_label(record: dict[str, object]) -> tuple[str, str, int]:
    label = str(record["benchmark"])
    test = str(record["test"]).lower()
    if "dense" in test or "dense" in str(record.get("tags", "")).lower():
        match = re.fullmatch(r"(\d+)D (static|dynamic) (float|double)( prior)?", label)
        if match:
            dimension, storage, precision, prior = match.groups()
            short = f"{dimension}{'f' if precision == 'float' else 'd'}"
            if prior:
                short += "p"
            return storage, short, int(dimension)
        return ("dynamic" if "dynamic" in label else "static"), label, 0
    if "sparse chain" in label.lower():
        match = re.match(r"(\d+)D", label)
        dimension = int(match.group(1)) if match else 0
        return "Sparse", f"{dimension}d", dimension
    if "pose" in test or "obs" in label.lower():
        match = re.match(r"(\d+)\s*obs\s*(.*)", label, re.I)
        if match:
            dimension = int(match.group(1))
            suffix = match.group(2).strip()
            return "Robust pose", f"{dimension}o {suffix}".strip(), dimension
        return "Robust pose", label, 0
    match = re.match(r"(\d+)c\s+(\d+)p", label)
    if match or "bundle" in test or test == "ba":
        if match:
            cameras, points = map(int, match.groups())
            return "Bundle adjustment", f"{cameras}c {points}p", cameras * points
    return "Other", label, 0


def records_for_group(records: list[dict[str, object]], group: str) -> list[dict[str, object]]:
    output = []
    for record in records:
        category, label, order = normalize_label(record)
        if category == group:
            output.append({**record, "short_label": label, "order": order, "category": category})
    return output


def table_rows(records: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for record in records:
        label = str(record["short_label"])
        rows.setdefault(label, {})[str(record["backend"])] = float(record["mean_ns"])
    return rows


def plot_group(records: list[dict[str, object]], backends: list[str], title: str,
               output: Path) -> None:
    rows = table_rows(records)
    labels = sorted(
        rows,
        key=lambda label: (
            min(int(record["order"]) for record in records
                if str(record["short_label"]) == label),
            label,
        ),
    )
    present_backends = [
        backend for backend in backends if any(backend in rows[label] for label in labels)
    ]
    unit_scale, unit = (1e6, "ms") if max(
        value for row in rows.values() for value in row.values()
    ) >= 1e6 else (1e3, "µs")
    figure, axis = plt.subplots(
        figsize=(max(11, len(present_backends) * 2.2), max(4.5, len(labels) * 0.42))
    )
    width = 0.8 / max(len(present_backends), 1)
    centers = list(range(len(labels)))
    for backend_index, backend in enumerate(present_backends):
        offsets = [center - 0.4 + width * (backend_index + 0.5) for center in centers]
        runtimes = [rows[label].get(backend, float("nan")) / unit_scale for label in labels]
        axis.barh(offsets, runtimes, height=width * 0.9, label=backend)
    axis.set_xscale("log")
    axis.set_xlabel(f"Mean runtime ({unit}, log scale)")
    axis.set_title(title)
    axis.set_yticks(centers)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.grid(axis="x", which="both", alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=160)


def format_timing(value_ns: float, scale: float) -> str:
    return f"{value_ns / scale:.2f}"


def format_table(records: list[dict[str, object]], backends: list[str], include_category: bool,
                 unit: str, scale: float) -> str:
    rows = table_rows(records)
    order_by_label = {
        str(record["short_label"]): int(record["order"]) for record in records
    }
    labels = sorted(rows, key=lambda label: (order_by_label[label], label))
    if include_category:
        labels.sort(
            key=lambda label: (
                0 if next(record["category"] for record in records
                          if str(record["short_label"]) == label) == "Robust pose" else 1,
                order_by_label[label],
                label,
            )
        )
    category_counts: dict[str, int] = {}
    if include_category:
        for label in labels:
            category = next(
                str(record["category"]) for record in records
                if str(record["short_label"]) == label
            )
            category_counts[category] = category_counts.get(category, 0) + 1
    header_cols = ["<th>Problem</th>"] if include_category else []
    header_cols.extend(["<th>Setup</th>"] if include_category else ["<th>Problem</th>"])
    header_cols.extend(f"<th>{html.escape(backend)}</th>" for backend in backends)
    body = []
    previous_category = None
    for label in labels:
        values = rows[label]
        present = sorted(values.values())
        fastest = present[0] if present else math.nan
        runner_up = present[1] if len(present) > 1 else math.nan
        baseline = values.get("tinyopt", math.nan)
        category = next(
            str(record["category"]) for record in records if str(record["short_label"]) == label
        )
        cells = []
        if include_category:
            cells.append(
                "" if category == previous_category else
                f'<th scope="rowgroup" rowspan="{category_counts[category]}">'
                f"{html.escape(category)}</th>"
            )
        setup = html.escape(label)
        cells.append(f'<th scope="row">{setup}</th>')
        for backend in backends:
            value = values.get(backend, math.nan)
            if not math.isfinite(value):
                cells.append("<td>—</td>")
                continue
            numeric = format_timing(value, scale)
            fastest_here = value == fastest
            runner_up_here = value == runner_up and value != fastest
            if fastest_here:
                numeric = f"<strong>{numeric}</strong>"
            elif runner_up_here:
                numeric = f"<em>{numeric}</em>"
            annotation = ""
            style = "color:#111"
            if math.isfinite(fastest) and value == fastest and len(present) > 1:
                style = "color:#16803c;font-weight:700"
            if backend != "tinyopt" and math.isfinite(baseline):
                speedup = baseline / value
                gain_style = "color:#16803c" if speedup > 1 else "color:#c62828"
                annotation = f' (<span style="{gain_style}">{speedup:.2f}x</span>)'
            cells.append(f'<td style="{style}">{numeric}{annotation}</td>')
        previous_category = category
        body.append(f"<tr>{''.join(cells)}</tr>")
    gains: dict[str, list[float]] = {backend: [] for backend in backends}
    for values in rows.values():
        baseline = values.get("tinyopt")
        if baseline is None or baseline <= 0:
            continue
        for backend in backends:
            value = values.get(backend)
            if backend != "tinyopt" and value is not None and value > 0:
                gains[backend].append(baseline / value)
    average_cells = []
    if include_category:
        average_cells.append("<th>—</th>")
    average_cells.append('<th scope="row">Mean Speedup</th>')
    for backend in backends:
        samples = gains[backend]
        if not samples:
            average_cells.append("<td>—</td>")
            continue
        average = len(samples) / sum(samples)
        color = "#16803c" if average > 1 else "#c62828"
        average_cells.append(
            f'<td><span style="color:{color}">{average:.2f}x</span></td>'
        )
    body.append(f'<tr class="average-row">{"".join(average_cells)}</tr>')
    return (
        f'<table data-unit="{unit}"><caption>Mean runtime ({unit})</caption>'
        f'<thead><tr>{"".join(header_cols)}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def build_configuration(build_dir: Path) -> str:
    cache = build_dir / "CMakeCache.txt"
    values: dict[str, str] = {}
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("//") or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.split(":", 1)[0]
            if key in CACHE_KEYS:
                values[key] = value
    details = [
        ("Architecture", platform.machine() or "unknown"),
        ("Platform", platform.platform()),
        ("Python", platform.python_version()),
        *sorted(values.items()),
    ]
    return "<dl>" + "".join(
        f"<dt>{html.escape(key)}</dt><dd>{html.escape(value)}</dd>" for key, value in details
    ) + "</dl>"


def format_iteration_table(items: list[dict[str, object]], backends: list[str],
                           include_category: bool) -> str:
    rows: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for item in items:
        key = (str(item["category"]), str(item["problem"]))
        rows.setdefault(key, {})[str(item["backend"])] = item
    categories = sorted(
        {key[0] for key in rows},
        key=lambda category: (
            0 if category == "Dense static" else
            1 if category == "Dense dynamic" else
            2 if category == "Sparse" else 3,
            category,
        ),
    )
    labels = []
    for category in categories:
        category_rows = [key for key in rows if key[0] == category]
        category_rows.sort(key=lambda key: (normalize_iteration_order(key[1]), key[1]))
        labels.extend(category_rows)
    category_counts = {category: sum(key[0] == category for key in labels) for category in categories}
    headings = ["<th>Problem</th>"] if include_category else []
    headings.extend(["<th>Setup</th>"] if include_category else ["<th>Problem</th>"])
    headings.extend(f"<th>{html.escape(backend)}</th>" for backend in backends)
    body = []
    previous_category = None
    for category, problem in labels:
        cells = []
        if include_category:
            cells.append(
                "" if category == previous_category else
                f'<th scope="rowgroup" rowspan="{category_counts[category]}">'
                f"{html.escape(category)}</th>"
            )
        cells.append(f'<th scope="row">{html.escape(problem)}</th>')
        for backend in backends:
            record = rows[(category, problem)].get(backend)
            if record is None:
                cells.append("<td>—</td>")
                continue
            status = "🎯" if record["converged"] else "❌"
            cells.append(f"<td>{record['iterations']} {status}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
        previous_category = category
    average_cells = []
    if include_category:
        average_cells.append("<th>—</th>")
    average_cells.append('<th scope="row">Mean</th>')
    for backend in backends:
        values = [
            int(methods[backend]["iterations"])
            for methods in rows.values()
            if backend in methods and methods[backend]["converged"]
        ]
        if not values:
            average_cells.append("<td>—</td>")
            continue
        average_cells.append(f"<td>{sum(values) / len(values):.2f}</td>")
    body.append(f'<tr class="average-row">{"".join(average_cells)}</tr>')
    return (
        f'<table><thead><tr>{"".join(headings)}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def stopping_criteria(category: str, problem: str, backend: str) -> str:
    is_float = problem.endswith("f")
    if backend == "tinyopt":
        if category.startswith("Dense") and is_float:
            return (
                "cost <1e-6; rel. decrease <1e-6; step² <1e-10; "
                "grad² <1e-18; max failures 3; max iter 100"
            )
        minimum_cost = "<1e-14" if category.startswith("Dense") else "<1e-12"
        max_iterations = 20 if problem.endswith("dp") else 100
        max_failures = 20 if category == "Robust pose" else 3
        return (
            f"cost {minimum_cost}; rel. decrease <1e-6; step² <1e-16; "
            f"grad² <1e-18; max failures {max_failures}; max iter {max_iterations}"
        )
    if backend == "ceres-tinysolver" and is_float:
        return "function <1e-6; gradient <1e-5; parameter <1e-5; max iter 100"
    if backend in {"ceres", "ceres-tinysolver"}:
        criteria = "function <1e-6; gradient <1e-9; parameter <1e-8; max iter 100"
        if backend == "ceres":
            invalid_steps = 20 if category == "Robust pose" else 3
            criteria += (
                f"; min relative decrease 1e-12; max invalid steps {invalid_steps}"
            )
        return criteria
    if backend == "g2o":
        max_iterations = 1000 if category == "Robust pose" else 100
        max_trials = 20 if category == "Robust pose" else 3
        return (
            f"rel. decrease <1e-6 OR step² <1e-16; "
            f"max trials {max_trials}; max iter {max_iterations}"
        )
    if backend == "gtsam":
        return (
            "cost <1e-12; relative error <1e-6; absolute decrease 0; model fidelity ≥1e-12; "
            "max iter 100"
        )
    return "—"


def format_stopping_criteria_table(items: list[dict[str, object]],
                                   backends: list[str]) -> str:
    available = {
        (str(item["category"]), str(item["problem"]), str(item["backend"]))
        for item in items
    }
    problems = sorted(
        {(str(item["category"]), str(item["problem"])) for item in items},
        key=lambda item: (
            0 if item[0] == "Dense static" else
            1 if item[0] == "Dense dynamic" else
            2 if item[0] == "Sparse" else 3,
            item[0],
            normalize_iteration_order(item[1]),
        ),
    )
    headings = ["<th>Problem</th>", "<th>Setup</th>"]
    headings.extend(f"<th>{html.escape(backend)}</th>" for backend in backends)
    body = []
    for category, problem in problems:
        cells = [
            f"<th scope=\"rowgroup\">{html.escape(category)}</th>",
            f"<th scope=\"row\">{html.escape(problem)}</th>",
        ]
        for backend in backends:
            if (category, problem, backend) in available:
                criterion = stopping_criteria(category, problem, backend)
            else:
                criterion = "—"
            cells.append(f"<td>{html.escape(criterion)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f'<table class="criteria"><thead><tr>{"".join(headings)}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def normalize_iteration_order(problem: str) -> tuple[int, int, str]:
    match = re.match(r"(\d+)([dfop])(?:\s+(.*))?$", problem)
    if match:
        return int(match.group(1)), 0, match.group(3) or ""
    match = re.match(r"(\d+)c\s+(\d+)p", problem)
    if match:
        return int(match.group(1)), int(match.group(2)), ""
    match = re.match(r"(\d+)d", problem)
    if match:
        return int(match.group(1)), 0, ""
    return 0, 0, problem


def write_html_report(records: list[dict[str, object]], backends: list[str], output_dir: Path,
                      build_dir: Path, iterations: list[dict[str, object]]) -> Path:
    groups = (
        ("Dense static", "static"),
        ("Dense dynamic", "dynamic"),
        ("Sparse", "Sparse"),
        ("Robust pose + bundle adjustment", "combined"),
    )
    sections = []
    for index, (title, group) in enumerate(groups, start=1):
        grouped = (
            [record for record in records if normalize_label(record)[0] in
             ("Robust pose", "Bundle adjustment")]
            if group == "combined"
            else records_for_group(records, group)
        )
        if not grouped:
            continue
        grouped = [
            {
                **record,
                "short_label": normalize_label(record)[1],
                "category": normalize_label(record)[0],
                "order": normalize_label(record)[2],
            }
            for record in grouped
        ]
        unit_scale, unit = (1e6, "ms") if max(
            float(record["mean_ns"]) for record in grouped
        ) >= 1e6 else (1e3, "µs")
        plot_path = output_dir / f"benchmark-{index}.png"
        plot_group(grouped, backends, title, plot_path)
        encoded = base64.b64encode(plot_path.read_bytes()).decode("ascii")
        image = f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(title)} plot">'
        table = format_table(grouped, backends, group == "combined", unit, unit_scale)
        sections.append(
            f"<details open><summary><h2>{html.escape(title)}</h2></summary>{image}"
            f'<div class="table-wrap">{table}</div></details>'
        )

    iteration_groups = (
        ("Dense", {"Dense static", "Dense dynamic"}, True),
        ("Sparse", {"Sparse"}, False),
        ("Bundle adjustment + robust pose", {"Bundle adjustment", "Robust pose"}, True),
    )
    iteration_sections = []
    for title, categories, include_category in iteration_groups:
        grouped = [item for item in iterations if item["category"] in categories]
        if grouped:
            iteration_sections.append(
                f"<details open><summary><h2>{html.escape(title)}</h2></summary>"
                f'<div class="table-wrap">{format_iteration_table(grouped, backends, include_category)}'
                "</div></details>"
            )
    iteration_sections.append(
        "<details><summary><h2>Stopping criteria by problem</h2></summary>"
        f'<div class="table-wrap">{format_stopping_criteria_table(iterations, backends)}'
        "</div></details>"
    )
    report = output_dir / "benchmark-report.html"
    report.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>Tinyopt benchmark report</title><style>"
        "body{font:15px system-ui,sans-serif;color:#20242a;max-width:1440px;margin:32px auto;"
        "padding:0 22px;background:#f6f8fa}h1,h2{color:#172b4d}details{background:white;"
        "padding:18px 22px;margin:16px 0;border:1px solid #d8dee4;border-radius:10px;"
        "box-shadow:0 2px 8px #172b4d0d}summary{cursor:pointer;list-style:none}summary::-webkit-"
        "details-marker{display:none}summary:before{content:'▸';display:inline-block;margin-right:"
        "8px;transition:transform .15s}details[open]>summary:before{transform:rotate(90deg)}"
        "summary h1,summary h2{display:inline}details>details{box-shadow:none;margin:12px 0;"
        "background:#fbfcfd}.table-wrap{overflow-x:auto}table{border-collapse:"
        "collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px 12px;"
        "border-bottom:1px solid #e6e9ed;text-align:right;white-space:nowrap}th{background:"
        "#f0f4f8;color:#334155}th:first-child,td:first-child{text-align:left}caption{"
        "caption-side:top;text-align:left;font-weight:600;padding:8px 0}img{width:100%;"
        "height:auto;border-radius:6px}tr.average-row{border-top:2px solid #9aa7b5}"
        "table.criteria td{text-align:left;white-space:normal;min-width:190px}"
        "dl{display:grid;"
        "grid-template-columns:max-content 1fr;gap:6px 16px}dt{font-weight:600}dd{margin:0;"
        "overflow-wrap:anywhere}</style></head><body><h1>Tinyopt benchmark report</h1>"
        "<details open><summary><h1>Timings</h1></summary>"
        "<p>Times are means; bold marks the fastest result and italics mark the runner-up. "
        "Third-party speedups compare against Tinyopt.</p>"
        + "".join(sections)
        + "</details><details open><summary><h1>Iterations</h1></summary>"
        + "".join(iteration_sections)
        + "</details><details><summary><h2>System and build configuration</h2></summary>"
        + build_configuration(build_dir)
        + "</details></body></html>",
        encoding="utf-8",
    )
    return report


def print_iteration_summary(iterations: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for item in iterations:
        key = (str(item["category"]), str(item["problem"]))
        grouped.setdefault(key, []).append(item)
    for (category, problem), methods in grouped.items():
        descriptions = [
            f"{item['backend']} {item['iterations']} {'ok' if item['converged'] else 'no'}"
            for item in methods
        ]
        converged = [item for item in methods if item["converged"]]
        fastest = min(converged, key=lambda item: int(item["iterations"])) if converged else None
        print(
            f"Iterations {category} {problem}: {', '.join(descriptions)}; "
            f"fastest {fastest['backend'] if fastest else 'none converged'}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build-bench"))
    parser.add_argument("--output-dir", default="tmp/benchmark-report",
                        help="Save plots, raw XML/CSV, and the HTML report to this folder.")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Plot existing backend XML files instead of running benchmarks.")
    parser.add_argument("--plot", action="store_true", help="Display plots and wait for them to close.")
    parser.add_argument("--show", action="store_true", help="Open the HTML report in the default browser.")
    parser.add_argument("--only", nargs="+", choices=BACKENDS, default=None,
                        help="Run only the selected backend(s); defaults to all five.")
    parser.add_argument("--samples", type=int, default=10, help="Catch2 samples per case.")
    parser.add_argument("--warmup-seconds", type=float, default=0.1)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup_seconds < 0:
        parser.error("--samples must be positive and --warmup-seconds cannot be negative")

    build_dir = (ROOT / args.build_dir).resolve() if not args.build_dir.is_absolute() else args.build_dir
    requested_dir = Path(args.output_dir)
    output_dir = (ROOT / requested_dir).resolve() if not requested_dir.is_absolute() else requested_dir
    results_dir = output_dir / "benchmark-results"
    results_dir.mkdir(parents=True, exist_ok=True)
    selected = args.only or list(BACKENDS)
    records: list[dict[str, object]] = []
    iterations: list[dict[str, object]] = []
    if args.input_dir is None:
        for backend in selected:
            backend_records, backend_iterations = run_backend(
                backend, build_dir, results_dir, args.samples, args.warmup_seconds
            )
            records.extend(backend_records)
            iterations.extend(backend_iterations)
    else:
        input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
        for backend in selected:
            xml_path = input_dir / f"{backend}.xml"
            xml_text = xml_path.read_text(encoding="utf-8")
            records.extend(parse_results(xml_text, backend))
            iterations.extend(parse_iterations(xml_text))
            (results_dir / xml_path.name).write_text(xml_text, encoding="utf-8")

    save_csv(records, output_dir / "benchmark-results.csv")
    save_iteration_csv(iterations, output_dir / "benchmark-iterations.csv")
    report = write_html_report(records, selected, output_dir, build_dir, iterations)
    print_iteration_summary(iterations)
    print(f"Saved benchmark report and plots under {output_dir} ({report.name})")
    if args.plot:
        plt.show(block=True)
    plt.close("all")
    if args.show:
        webbrowser.open(report.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
