#!/usr/bin/env python3
"""Profile dedicated Tinyopt workloads with Linux perf and emit HTML reports."""

from __future__ import annotations

import argparse
import base64
import datetime
import html
import io
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = ROOT / "build-profilings"
WORKLOADS = {
    "1d": "tinyopt_profiling_1d",
    "2d": "tinyopt_profiling_2d",
    "sparse-ba": "tinyopt_profiling_sparse_ba",
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


def profile_workload(perf: str, workload: str, executable: Path, output_dir: Path,
                     frequency: int, timestamp: str) -> int:
    if not executable.is_file():
        print(f"Missing profiling executable: {executable}", file=sys.stderr)
        print("Run `pixi run build-profilings` first.", file=sys.stderr)
        return 1

    data_path = output_dir / f"{timestamp}-{workload}.data"
    report_path = output_dir / f"{timestamp}-{workload}.txt"
    html_path = output_dir / f"{timestamp}-{workload}.html"
    record_command = [
        perf, "record", "--all-user", "--call-graph", "dwarf", "--freq", str(frequency),
        "--output", str(data_path), "--", str(executable),
    ]
    print(f"Profiling {workload}: {executable}", flush=True)
    recorded = subprocess.run(record_command, cwd=ROOT, text=True, capture_output=True, check=False)
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
        [perf, "report", "--stdio", "--no-children", "--call-graph", "none",
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

    hotspots = parse_hotspots(report.stdout)
    report_path.write_text(report.stdout, encoding="utf-8")
    if not hotspots:
        print(f"Could not parse function rows; raw report saved to {report_path}", file=sys.stderr)
        return 1
    html_path.write_text(render_html_report(hotspots, workload, data_path), encoding="utf-8")
    print("\nTop function hotspots by sampled overhead:")
    print(format_hotspot_table(hotspots))
    print(f"HTML report with hotspot graph saved to {html_path}")
    print(f"Raw profile saved to {data_path}")
    return 0


def main() -> int:
    arguments = parse_args()
    perf = shutil.which("perf")
    if perf is None:
        print("Linux perf was not found on PATH.", file=sys.stderr)
        return 1

    output_dir = arguments.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    workloads = WORKLOADS.items() if arguments.workload == "all" else (
        (arguments.workload, WORKLOADS[arguments.workload]),
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return max(
        profile_workload(
            perf,
            name,
            BUILD_DIR / "benchmarks" / "tinyopt" / "profiling" / executable,
            output_dir,
            arguments.frequency,
            timestamp,
        )
        for name, executable in workloads
    )


if __name__ == "__main__":
    raise SystemExit(main())
