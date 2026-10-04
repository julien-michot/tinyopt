#!/usr/bin/env python3
"""Plot test compilation and benchmark timing history."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TIMINGS_DIR = ROOT / "tmp" / "timings"


def load_history() -> list[dict[str, object]]:
    history = []
    for path in TIMINGS_DIR.glob("*.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            metrics = result["metrics"]
            timestamp = result.get("commit_datetime") or result["recorded_at"]
            if "test_compile_seconds" not in metrics or "benchmark_mean_sum_seconds" not in metrics:
                continue
            history.append(
                {
                    "commit": result.get("commit", path.stem),
                    "timestamp": datetime.fromisoformat(timestamp.replace("Z", "+00:00")),
                    "test_compile_seconds": float(metrics["test_compile_seconds"]),
                    "benchmark_mean_sum_seconds": float(metrics[
                        "benchmark_mean_sum_seconds"
                    ]),
                }
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"Skipping invalid timing file {path.name}: {error}")

    return sorted(history, key=lambda item: item["timestamp"])


def plot_history(history: list[dict[str, object]], output: Path) -> None:
    timestamps = [item["timestamp"] for item in history]
    commits = [str(item["commit"]) for item in history]
    metrics = (
        ("test_compile_seconds", "Clean Test Compilation", "#167d72"),
        ("benchmark_mean_sum_seconds", "Tinyopt Benchmark Mean Sum", "#d86b42"),
    )

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    figure.suptitle("Tinyopt Performance History", fontsize=16, fontweight="bold")

    for axis, (key, title, color) in zip(axes, metrics):
        values = [float(item[key]) for item in history]
        axis.plot(timestamps, values, color=color, linewidth=2, marker="o", markersize=6)
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel("Seconds")
        axis.grid(axis="y", color="#d7dedb", linewidth=0.8, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        for timestamp, value, commit in zip(timestamps, values, commits):
            axis.annotate(
                commit,
                (timestamp, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#34413e",
            )

    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=timezone.utc)
    )
    axes[-1].set_xlabel("Commit time (UTC)")
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=TIMINGS_DIR / "evolution.png",
        help="plot output path (default: tmp/timings/evolution.png)",
    )
    arguments = parser.parse_args()

    history = load_history()
    if not history:
        parser.error("no XML-based timing records found in tmp/timings/")

    output = arguments.output if arguments.output.is_absolute() else ROOT / arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_history(history, output)
    print(f"Saved timing history for {len(history)} commits to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())