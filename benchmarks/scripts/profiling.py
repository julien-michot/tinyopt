#!/usr/bin/env python3
"""Profile dedicated Tinyopt workloads with Linux perf or macOS sample and emit HTML reports."""

from __future__ import annotations

import argparse
import base64
import datetime
import html
import io
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = ROOT / "build-profilings"
WORKLOADS = {
    "1d": "tinyopt_profiling_1d",
    "2d": "tinyopt_profiling_2d",
    "3d-dyn": "tinyopt_profiling_3d",
    "6d": "tinyopt_profiling_6d",
    "block-sparse-ba": "tinyopt_profiling_block_sparse_ba",
    "sparse-ba": "tinyopt_profiling_sparse_ba",
}
# The micro workloads finish in a few milliseconds with their default run count,
# which leaves the sampler with too few samples to attribute. Give them enough
# work to run for a couple of seconds instead.
PROFILE_ARGS = {
    "1d": ["--runs", "200000000", "--samples", "1"],
    "2d": ["--runs", "20000000", "--samples", "1"],
    "6d": ["--runs", "5000000", "--samples", "1"],
}
HOTSPOT_LINE = re.compile(r"^\s*(?P<percent>\d+(?:\.\d+)?)%\s+(?P<tail>.+?)\s*$")
SYMBOL_MARKER = re.compile(r"\s+\[[^\]]+\]\s+(?P<symbol>.+?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workload",
        nargs="?",
        choices=(*WORKLOADS, "all"),
        default="all",
        help="workload to profile (default: all)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build-bench"),
        help="build directory containing profiling targets (default: build-bench)",
    )
    parser.add_argument("--frequency", type=int, default=999, help="perf sampling frequency in Hz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/profiles"),
        help="directory for perf data and reports (default: tmp/profiles)",
    )
    arguments = parser.parse_args()
    if arguments.frequency < 1:
        parser.error("--frequency must be positive")
    return arguments


def parse_hotspots(report: str, limit: int = 50) -> list[tuple[str, float]]:
    totals: dict[str, float] = {}
    for line in report.splitlines():
        match = HOTSPOT_LINE.match(line)
        if match is None:
            continue
        tail = match.group("tail")
        marker = SYMBOL_MARKER.search(tail)
        symbol = (marker.group("symbol") if marker else tail).strip()
        if symbol:
            totals[symbol] = totals.get(symbol, 0.0) + float(match.group("percent"))
    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:limit]


def format_hotspot_table(hotspots: list[tuple[str, float]]) -> str:
    if not hotspots:
        return "No symbol rows were parsed from perf report output."
    name_width = max(len("Function"), min(100, max(len(name) for name, _ in hotspots)))
    rows = [f"{'Rank':>4}  {'Overhead':>9}  {'Function':<{name_width}}", "-" * (17 + name_width)]
    rows.extend(
        f"{rank:>4}  {percent:>8.2f}%  {name[:name_width]}"
        for rank, (name, percent) in enumerate(hotspots, start=1)
    )
    return "\n".join(rows)


def render_chart(hotspots: list[tuple[str, float]]) -> str:
    chart_items = hotspots[:20]
    figure, axis = plt.subplots(figsize=(11, max(4, len(chart_items) * 0.32)))
    labels = [name if len(name) <= 90 else f"{name[:87]}..." for name, _ in reversed(chart_items)]
    values = [percent for _, percent in reversed(chart_items)]
    axis.barh(labels, values, color="#16815e")
    axis.set_xlabel("Sampled overhead (%)")
    axis.set_title("Top function hotspots")
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    axis.tick_params(axis="y", labelsize=7)
    figure.subplots_adjust(left=0.42, right=0.98, top=0.92, bottom=0.12)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, facecolor="white")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_html_report(hotspots: list[tuple[str, float]], workload: str,
                       profile_path: Path) -> str:
    peak = max((percent for _, percent in hotspots), default=1.0) or 1.0
    rows = []
    for rank, (symbol, percent) in enumerate(hotspots, start=1):
        width = min(100.0, percent / peak * 100.0)
        rows.append(
            "<tr>"
            f"<td class=\"rank\">{rank}</td>"
            f"<td class=\"percent\">{percent:.2f}%</td>"
            f"<td><div class=\"bar-track\"><div class=\"bar\" style=\"width:{width:.2f}%\"></div>"
            f"<span>{html.escape(symbol)}</span></div></td>"
            "</tr>"
        )
    table_rows = "\n".join(rows) or '<tr><td colspan="3">No function rows parsed.</td></tr>'
    chart = render_chart(hotspots) if hotspots else ""
    chart_section = f'<img class="chart" src="{chart}" alt="Function hotspot chart">' if chart else ""
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tinyopt profile: {html.escape(workload)}</title>
<style>
:root {{ color-scheme: light; font: 14px/1.45 system-ui, sans-serif; color: #172522; background: #f3f6f4; }}
body {{ margin: 0; padding: 32px; }}
main {{ max-width: 1200px; margin: auto; }}
h1 {{ margin: 0 0 6px; font-size: 24px; }}
h2 {{ margin: 28px 0 12px; font-size: 18px; }}
.meta {{ color: #52645e; margin-bottom: 22px; overflow-wrap: anywhere; }}
.chart {{ display: block; width: 100%; height: auto; background: white; }}
table {{ width: 100%; border-collapse: collapse; background: white; }}
th, td {{ padding: 8px 12px; border-bottom: 1px solid #e1e8e4; text-align: left; }}
th {{ color: #52645e; font-size: 12px; text-transform: uppercase; }}
.rank, .percent {{ width: 80px; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.bar-track {{ position: relative; min-height: 25px; display: flex; align-items: center; overflow: hidden; }}
.bar {{ position: absolute; inset: 3px auto 3px 0; background: #a6d9c0; border-left: 3px solid #16815e; }}
.bar-track span {{ position: relative; padding: 2px 6px; overflow-wrap: anywhere; }}
@media (max-width: 650px) {{ body {{ padding: 14px; }} th, td {{ padding: 7px; }} }}
</style>
<main>
<h1>Function Hotspots</h1>
<div class="meta">Workload: {html.escape(workload)} · Profile: {html.escape(str(profile_path))} · Top {len(hotspots)} symbols by sampled overhead</div>
<h2>Sampled overhead</h2>
{chart_section}
<h2>Function details</h2>
<table><thead><tr><th>Rank</th><th>Overhead</th><th>Function</th></tr></thead>
<tbody>{table_rows}</tbody></table>
</main>
</html>
"""


def parse_sample_hotspots(report: str, limit: int = 50) -> list[tuple[str, float]]:
    """Parse 'Sort by top of stack' section or call graph from macOS sample output."""
    totals: dict[str, float] = {}
    total_samples = 0

    # First check for the "Sort by top of stack, same collapsed" section
    in_summary = False
    for line in report.splitlines():
        if "Sort by top of stack" in line:
            in_summary = True
            continue
        if in_summary:
            if not line.strip() or line.startswith("Binary images"):
                break
            # Format: symbol  (in dylib)   count
            match = re.match(r"^\s*(.+?)\s+(\d+)\s*$", line)
            if match:
                symbol = match.group(1).strip()
                count = int(match.group(2))
                totals[symbol] = count
                total_samples += count

    # If no summary section found, extract leaf nodes from Call graph:
    if not totals:
        for line in report.splitlines():
            # Lines look like: + ! : 5 dyld4::RuntimeState::notify... (in dyld) + 432 [0x18f...]
            match = re.search(r"(\d+)\s+([A-Za-z0-9_::~<>,.*&]+.*?\s+\(in [^)]+\))", line)
            if match:
                count = int(match.group(1))
                symbol = match.group(2).strip()
                totals[symbol] = totals.get(symbol, 0) + count
                total_samples = max(total_samples, count)

    if total_samples <= 0:
        total_samples = sum(totals.values()) or 1

    percentages = [
        (symbol, (count / total_samples) * 100.0)
        for symbol, count in totals.items()
    ]
    return sorted(percentages, key=lambda item: (-item[1], item[0]))[:limit]


def profile_workload(profiler: str, profiler_type: str, workload: str, executable: Path,
                     output_dir: Path, frequency: int, timestamp: str) -> int:
    if not executable.is_file():
        print(f"Missing profiling executable: {executable}", file=sys.stderr)
        print("Run `pixi run build-bench` or `pixi run build-profilings` first.", file=sys.stderr)
        return 1

    report_path = output_dir / f"{timestamp}-{workload}.txt"
    html_path = output_dir / f"{timestamp}-{workload}.html"

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    if profiler_type == "sample":
        # macOS sample profiler
        # Run workload in background, sample it for duration
        print(f"Profiling {workload} on macOS with sample: {executable}", flush=True)
        workload_args = PROFILE_ARGS.get(workload, [])
        proc = subprocess.Popen([str(executable), *workload_args], env=env)
        time.sleep(0.02)
        # sample interval in ms = 1000 / frequency
        interval_ms = max(1, int(1000 / frequency))
        duration_s = "10"
        sample_cmd = [
            profiler,
            str(proc.pid),
            duration_s,
            str(interval_ms),
            "-file",
            str(report_path),
        ]
        sampled = subprocess.run(sample_cmd, cwd=ROOT, text=True, capture_output=True, check=False)
        proc.wait()
        if not report_path.is_file() or report_path.stat().st_size == 0:
            print(f"sample failed or workload ended before sampling: {sampled.stderr}", file=sys.stderr)
            return 1
        raw_report = report_path.read_text(encoding="utf-8", errors="replace")
        hotspots = parse_sample_hotspots(raw_report)
        data_path = report_path
    else:
        # Linux perf profiler
        data_path = output_dir / f"{timestamp}-{workload}.data"
        record_command = [
            profiler, "record", "--all-user", "--call-graph", "dwarf", "--freq", str(frequency),
            "--output", str(data_path), "--", str(executable),
            *PROFILE_ARGS.get(workload, []),
        ]
        print(f"Profiling {workload} on Linux with perf: {executable}", flush=True)
        recorded = subprocess.run(record_command, cwd=ROOT, text=True, capture_output=True, check=False, env=env)
        if recorded.stdout:
            print(recorded.stdout, end="")
        if recorded.stderr:
            print(recorded.stderr, end="", file=sys.stderr)
            print('Try this fix: sudo sysctl -w kernel.perf_event_paranoid=0  # or grant CAP_PERFMON to the executable', file=sys.stderr)
        if recorded.returncode != 0:
            if data_path.is_file() and data_path.stat().st_size == 0:
                data_path.unlink()
            print(
                "perf could not access CPU sampling events. Set kernel.perf_event_paranoid to 0 "
                "(or grant CAP_PERFMON), then retry; no function profile was generated.",
                file=sys.stderr,
            )
            return recorded.returncode

        report = subprocess.run(
            [profiler, "report", "--stdio", "--no-children", "--call-graph", "none",
             "--column-widths=12,200", "--sort", "symbol", "--percent-limit", "0",
             "--input", str(data_path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if report.returncode != 0:
            if report.stderr:
                print(report.stderr, end="", file=sys.stderr)
            print(f"Could not render perf report; raw data remains at {data_path}", file=sys.stderr)
            return report.returncode
        raw_report = report.stdout
        report_path.write_text(raw_report, encoding="utf-8")
        hotspots = parse_hotspots(raw_report)

    if not hotspots:
        print(f"Could not parse function rows; raw report saved to {report_path}", file=sys.stderr)
        return 1
    html_path.write_text(render_html_report(hotspots, workload, data_path), encoding="utf-8")
    print("\nTop function hotspots by sampled overhead:")
    print(format_hotspot_table(hotspots))
    print(f"HTML report with hotspot graph saved to {html_path}")
    print(f"Raw profile / call graph saved to {report_path}")
    return 0


def main() -> int:
    arguments = parse_args()

    profiler_type = None
    profiler = None
    if sys.platform == "darwin":
        sample = shutil.which("sample") or "/usr/bin/sample"
        if Path(sample).is_file():
            profiler = str(sample)
            profiler_type = "sample"
    if not profiler:
        perf = shutil.which("perf")
        if perf:
            profiler = str(perf)
            profiler_type = "perf"

    if profiler is None:
        print("No supported profiler found (perf on Linux, /usr/bin/sample on macOS).", file=sys.stderr)
        return 1

    output_dir = arguments.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir = arguments.build_dir
    if not build_dir.is_absolute():
        build_dir = ROOT / build_dir
    workloads = WORKLOADS.items() if arguments.workload == "all" else (
        (arguments.workload, WORKLOADS[arguments.workload]),
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return max(
        profile_workload(
            profiler,
            profiler_type,
            name,
            build_dir / "benchmarks" / "tinyopt" / "profiling" / executable,
            output_dir,
            arguments.frequency,
            timestamp,
        )
        for name, executable in workloads
    )


if __name__ == "__main__":
    raise SystemExit(main())
