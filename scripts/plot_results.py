#!/usr/bin/env python3
"""
Publication-quality plotting for optimizer benchmark results.

Generates:
  1. Convergence curves (validation accuracy & loss vs. epochs) with ± std shading
  2. Bar charts (final accuracy, peak memory, throughput)
  3. Pareto frontiers (accuracy vs. time, accuracy vs. memory)
  4. Summary tables (LaTeX-ready)
  5. Performance profiles (Dolan-Moré style)

All plots exported as PDF for IEEE paper embedding.

Usage:
    uv run python scripts/plot_results.py --results-dir results/
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Publication-quality style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Color palette for optimizers
OPTIMIZER_COLORS = {
    "adam": "#2196F3",      # Blue
    "adamuon": "#FF5722",   # Deep Orange
    "kfac": "#4CAF50",      # Green
}
OPTIMIZER_LABELS = {
    "adam": "Adam",
    "adamuon": "AdaMuon",
    "kfac": "K-FAC",
}


def load_all_results(results_dir: str) -> dict:
    """Load all metrics.json files into a structured dict."""
    results = defaultdict(lambda: defaultdict(list))
    base = Path(results_dir)

    for metrics_file in sorted(base.rglob("metrics.json")):
        parts = metrics_file.relative_to(base).parts
        if len(parts) >= 3:
            dataset = parts[0]
            optimizer = parts[1]
            seed_dir = parts[2]

            with open(metrics_file) as f:
                data = json.load(f)
            results[dataset][optimizer].append(data)

    return dict(results)


def plot_convergence_curves(results: dict, output_dir: Path):
    """Plot validation accuracy and loss vs. epochs for each dataset."""
    for dataset, opt_results in results.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for opt_name, runs in opt_results.items():
            color = OPTIMIZER_COLORS.get(opt_name, "#999999")
            label = OPTIMIZER_LABELS.get(opt_name, opt_name)

            # Collect per-epoch data across seeds
            all_val_acc = []
            all_val_loss = []

            for run in runs:
                epochs = run["metrics"]["epoch_metrics"]
                all_val_acc.append([e["val_accuracy"] for e in epochs])
                all_val_loss.append([e["val_loss"] for e in epochs])

            # Align lengths
            min_len = min(len(a) for a in all_val_acc)
            all_val_acc = np.array([a[:min_len] for a in all_val_acc])
            all_val_loss = np.array([l[:min_len] for l in all_val_loss])

            epochs_range = np.arange(1, min_len + 1)
            mean_acc = all_val_acc.mean(axis=0)
            std_acc = all_val_acc.std(axis=0)
            mean_loss = all_val_loss.mean(axis=0)
            std_loss = all_val_loss.std(axis=0)

            # Accuracy
            axes[0].plot(epochs_range, mean_acc, color=color, label=label, linewidth=1.5)
            axes[0].fill_between(epochs_range, mean_acc - std_acc, mean_acc + std_acc,
                                color=color, alpha=0.15)

            # Loss
            axes[1].plot(epochs_range, mean_loss, color=color, label=label, linewidth=1.5)
            axes[1].fill_between(epochs_range, mean_loss - std_loss, mean_loss + std_loss,
                                color=color, alpha=0.15)

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Validation Accuracy")
        axes[0].set_title(f"{dataset} — Validation Accuracy")
        axes[0].legend()

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation Loss")
        axes[1].set_title(f"{dataset} — Validation Loss")
        axes[1].legend()

        plt.tight_layout()
        fig.savefig(output_dir / f"convergence_{dataset}.pdf")
        fig.savefig(output_dir / f"convergence_{dataset}.svg")
        plt.close(fig)
        print(f"  Saved convergence_{dataset}.pdf")


def plot_bar_charts(results: dict, output_dir: Path):
    """Plot bar charts for final accuracy, peak memory, and throughput."""
    for dataset, opt_results in results.items():
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        opt_names = []
        final_accs = []
        final_acc_stds = []
        peak_mems = []
        peak_mem_stds = []
        throughputs = []
        throughput_stds = []

        for opt_name in ["adam", "adamuon", "kfac"]:
            if opt_name not in opt_results:
                continue
            runs = opt_results[opt_name]
            opt_names.append(OPTIMIZER_LABELS.get(opt_name, opt_name))

            accs = [r["metrics"]["summary"]["best_val_accuracy"] for r in runs]
            mems = [r["metrics"]["summary"]["peak_memory_mb"] for r in runs]
            thrs = [r["metrics"]["summary"]["avg_throughput_samples_sec"] for r in runs]

            final_accs.append(np.mean(accs))
            final_acc_stds.append(np.std(accs))
            peak_mems.append(np.mean(mems))
            peak_mem_stds.append(np.std(mems))
            throughputs.append(np.mean(thrs))
            throughput_stds.append(np.std(thrs))

        x = np.arange(len(opt_names))
        colors = [OPTIMIZER_COLORS.get(k, "#999") for k in ["adam", "adamuon", "kfac"] if k in opt_results]

        # Final Accuracy
        axes[0].bar(x, final_accs, yerr=final_acc_stds, color=colors, capsize=4, edgecolor="black", linewidth=0.5)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(opt_names)
        axes[0].set_ylabel("Best Val Accuracy")
        axes[0].set_title(f"{dataset}")

        # Peak Memory
        axes[1].bar(x, [m / 1000 for m in peak_mems],
                    yerr=[s / 1000 for s in peak_mem_stds],
                    color=colors, capsize=4, edgecolor="black", linewidth=0.5)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(opt_names)
        axes[1].set_ylabel("Peak GPU Memory (GB)")
        axes[1].set_title(f"{dataset}")

        # Throughput
        axes[2].bar(x, throughputs, yerr=throughput_stds, color=colors, capsize=4, edgecolor="black", linewidth=0.5)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(opt_names)
        axes[2].set_ylabel("Throughput (samples/sec)")
        axes[2].set_title(f"{dataset}")

        plt.tight_layout()
        fig.savefig(output_dir / f"bars_{dataset}.pdf")
        fig.savefig(output_dir / f"bars_{dataset}.svg")
        plt.close(fig)
        print(f"  Saved bars_{dataset}.pdf")


def plot_pareto_frontiers(results: dict, output_dir: Path):
    """Plot accuracy vs. time and accuracy vs. memory Pareto frontiers."""
    for dataset, opt_results in results.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        for opt_name, runs in opt_results.items():
            color = OPTIMIZER_COLORS.get(opt_name, "#999999")
            label = OPTIMIZER_LABELS.get(opt_name, opt_name)

            accs = [r["metrics"]["summary"]["best_val_accuracy"] for r in runs]
            times = [r["metrics"]["summary"]["total_time_sec"] for r in runs]
            mems = [r["metrics"]["summary"]["peak_memory_mb"] / 1000 for r in runs]  # GB

            axes[0].scatter(times, accs, color=color, label=label, s=50, zorder=5, edgecolors="black", linewidths=0.5)
            axes[1].scatter(mems, accs, color=color, label=label, s=50, zorder=5, edgecolors="black", linewidths=0.5)

        axes[0].set_xlabel("Total Training Time (s)")
        axes[0].set_ylabel("Best Val Accuracy")
        axes[0].set_title(f"{dataset} — Accuracy vs. Time")
        axes[0].legend()

        axes[1].set_xlabel("Peak GPU Memory (GB)")
        axes[1].set_ylabel("Best Val Accuracy")
        axes[1].set_title(f"{dataset} — Accuracy vs. Memory")
        axes[1].legend()

        plt.tight_layout()
        fig.savefig(output_dir / f"pareto_{dataset}.pdf")
        fig.savefig(output_dir / f"pareto_{dataset}.svg")
        plt.close(fig)
        print(f"  Saved pareto_{dataset}.pdf")


def plot_time_per_epoch(results: dict, output_dir: Path):
    """Plot average time per epoch as grouped bar chart."""
    datasets = []
    opt_times = defaultdict(list)
    opt_stds = defaultdict(list)

    for dataset, opt_results in results.items():
        datasets.append(dataset)
        for opt_name in ["adam", "adamuon", "kfac"]:
            if opt_name in opt_results:
                times = [r["metrics"]["summary"]["avg_epoch_time_sec"] for r in opt_results[opt_name]]
                opt_times[opt_name].append(np.mean(times))
                opt_stds[opt_name].append(np.std(times))
            else:
                opt_times[opt_name].append(0)
                opt_stds[opt_name].append(0)

    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, opt_name in enumerate(["adam", "adamuon", "kfac"]):
        ax.bar(x + i * width, opt_times[opt_name], width,
               yerr=opt_stds[opt_name],
               label=OPTIMIZER_LABELS[opt_name],
               color=OPTIMIZER_COLORS[opt_name],
               capsize=3, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Dataset × Model")
    ax.set_ylabel("Avg Time per Epoch (s)")
    ax.set_title("Average Epoch Time Across Workloads")
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "epoch_time_comparison.pdf")
    fig.savefig(output_dir / "epoch_time_comparison.svg")
    plt.close(fig)
    print("  Saved epoch_time_comparison.pdf")


def generate_latex_table(results: dict, output_dir: Path):
    """Generate LaTeX tables with mean±std for all metrics."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\caption{Summary of Benchmark Results (mean $\pm$ std over 5 seeds)}")
    lines.append(r"\centering\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{|l|l|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Workload} & \textbf{Optimizer} & \textbf{Best Val Acc} & \textbf{Time (s)} & \textbf{Memory (GB)} & \textbf{Throughput} \\")
    lines.append(r"\hline")

    for dataset, opt_results in results.items():
        for opt_name in ["adam", "adamuon", "kfac"]:
            if opt_name not in opt_results:
                continue
            runs = opt_results[opt_name]
            accs = [r["metrics"]["summary"]["best_val_accuracy"] for r in runs]
            times = [r["metrics"]["summary"]["total_time_sec"] for r in runs]
            mems = [r["metrics"]["summary"]["peak_memory_mb"] / 1000 for r in runs]
            thrs = [r["metrics"]["summary"]["avg_throughput_samples_sec"] for r in runs]

            label = OPTIMIZER_LABELS[opt_name]
            lines.append(
                f"{dataset} & {label} & "
                f"${np.mean(accs):.4f} \\pm {np.std(accs):.4f}$ & "
                f"${np.mean(times):.1f} \\pm {np.std(times):.1f}$ & "
                f"${np.mean(mems):.2f} \\pm {np.std(mems):.2f}$ & "
                f"${np.mean(thrs):.0f} \\pm {np.std(thrs):.0f}$ \\\\"
            )
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\label{tab:benchmark_results}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)
    with open(output_dir / "summary_table.tex", "w") as f:
        f.write(latex_str)
    print("  Saved summary_table.tex")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality benchmark plots")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory with experiment results")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory for output plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_all_results(args.results_dir)

    if not results:
        print(f"No results found in {args.results_dir}/")
        return

    print(f"Found {len(results)} datasets:")
    for ds, opts in results.items():
        print(f"  {ds}: {list(opts.keys())} ({sum(len(v) for v in opts.values())} runs)")

    print("\nGenerating plots...")
    plot_convergence_curves(results, output_dir)
    plot_bar_charts(results, output_dir)
    plot_pareto_frontiers(results, output_dir)
    plot_time_per_epoch(results, output_dir)
    generate_latex_table(results, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
