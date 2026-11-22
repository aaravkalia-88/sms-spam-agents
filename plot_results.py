"""Render the benchmark figures from the checked-in aggregate reports."""

import argparse
import json
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

MODELS = (
    ("word", "Word agent", "#176963", "o"),
    ("character", "Character agent", "#5f4da6", "D"),
    ("coordinator", "Coordinator", "#a84921", "s"),
)


def plot_splits(reports, output):
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    fig.suptitle("Each split favored a different model", fontsize=16, fontweight="bold")
    for position, (key, label, color, marker) in enumerate(MODELS):
        offset = (position - 1) * 0.14
        ax.scatter(
            [i + offset for i in range(len(reports))],
            [report["models"][key]["f1"] for report in reports],
            label=label,
            color=color,
            marker=marker,
            s=72,
            zorder=3,
        )
    ax.set_xticks(range(len(reports)), [f"Seed {report['seed']}" for report in reports])
    ax.set(xlim=(-0.38, len(reports) - 0.62), ylabel="Spam F1")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color="#dce3ed")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3, frameon=False)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_primary_split(report, output):
    fig, (errors, intervals) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    fig.suptitle(f"Held-out comparison · seed {report['seed']}", fontsize=16, fontweight="bold")
    for position, (key, label, color, _) in enumerate(MODELS):
        metrics = report["models"][key]
        offset = (position - 1) * 0.23
        bars = errors.bar(
            [offset, 1 + offset],
            [metrics["fp"], metrics["fn"]],
            width=0.21,
            label=label,
            color=color,
            zorder=3,
        )
        errors.bar_label(bars, padding=2)
    errors.set(title="Classification errors", ylabel="Messages", ylim=(0, errors.get_ylim()[1] + 1))
    errors.set_xticks([0, 1], ["Legitimate flagged", "Spam missed"])
    errors.yaxis.set_major_locator(MaxNLocator(integer=True))
    errors.grid(axis="y", color="#dce3ed")
    errors.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=8)

    intervals.axvline(0, color="#626974", linestyle="--", linewidth=1)
    for row, (baseline, _, color, marker) in enumerate(reversed(MODELS[:2])):
        data = report["paired_f1"][baseline]
        low, high = (100 * value for value in data["ci95"])
        intervals.hlines(row, low, high, color=color, linewidth=2)
        intervals.scatter(100 * data["delta_f1"], row, color=color, marker=marker, s=55, zorder=3)
    intervals.set_yticks([0, 1], ["vs. characters", "vs. words"])
    intervals.set(
        title="Coordinator F1 difference, 95% intervals",
        xlabel="Percentage points",
        ylim=(-0.55, 1.55),
    )
    intervals.grid(axis="x", color="#dce3ed")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    plt.switch_backend("Agg")
    reports = sorted(
        (
            json.loads(path.read_text(encoding="utf-8"))
            for path in args.results_dir.glob("seed-*.json")
        ),
        key=lambda report: report["seed"],
    )
    primary = next((report for report in reports if report["seed"] == 42), None)
    if primary is None:
        parser.error("The results directory must include the seed 42 report.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_splits(reports, args.output_dir / "f1-by-split.png")
    plot_primary_split(primary, args.output_dir / "seed-42-tradeoff.png")


if __name__ == "__main__":
    main()
