#!/usr/bin/env python3
"""
scripts/plot_byzantine_sweep.py
--------------------------------
Runs the FLTrust vs FedAvg vs Krum Byzantine-ratio sweep using the
instrumented run_experiment() in benchmark_byzantine.py, and produces:

  reports/byzantine_results.csv   one row per (strategy, byzantine_ratio)
  reports/byzantine_sweep.png     two panels:
                                    1) Byzantine-client detection accuracy
                                    2) Final global model divergence from
                                       a clean (0% Byzantine) baseline

Usage:
    python scripts/plot_byzantine_sweep.py
    python scripts/plot_byzantine_sweep.py --ratios 0.1 0.2 0.3 0.4 --num-clients 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from benchmark_byzantine import run_experiment  # instrumented version


def l2_divergence(arrays_a, arrays_b):
    """Euclidean distance between two flattened weight-array lists."""
    flat_a = np.concatenate([a.ravel() for a in arrays_a])
    flat_b = np.concatenate([b.ravel() for b in arrays_b])
    return float(np.linalg.norm(flat_a - flat_b))


def run_sweep(strategies, ratios, num_clients=10, seed=42):
    results = []
    baselines = {}

    # Clean baseline per strategy: 0% Byzantine, fixed seed. Any divergence
    # at higher ratios is then attributable to the attack + how well the
    # strategy resists it, not to run-to-run randomness.
    for strat in strategies:
        clean = run_experiment(strat, num_clients, byzantine_ratio=0.0, seed=seed)
        baselines[strat] = clean["final_arrays"]
        clean["divergence"] = 0.0
        results.append(clean)

    for strat in strategies:
        for ratio in ratios:
            res = run_experiment(strat, num_clients, byzantine_ratio=ratio, seed=seed)
            res["divergence"] = l2_divergence(res["final_arrays"], baselines[strat])
            results.append(res)

    return results


def export_and_plot(results, reports_dir):
    reports_dir.mkdir(exist_ok=True)

    # Drop the large weight arrays / list columns before writing to CSV
    rows = [
        {k: v for k, v in r.items() if k not in ("final_arrays", "roles", "flagged_indices")}
        for r in results
    ]
    df = pd.DataFrame(rows).sort_values(["strategy", "byzantine_ratio"])
    csv_path = reports_dir / "byzantine_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"FedAvg": "crimson", "Krum": "darkorange", "FLTrust": "steelblue"}

    for strat in df["strategy"].unique():
        sub = df[df["strategy"] == strat].sort_values("byzantine_ratio")
        color = colors.get(strat)
        axes[0].plot(sub["byzantine_ratio"] * 100, sub["balanced_accuracy"],
                     marker="o", label=strat, color=color)
        axes[1].plot(sub["byzantine_ratio"] * 100, sub["divergence"],
                     marker="o", label=strat, color=color)

    axes[0].set_xlabel("Byzantine Client Ratio (%)")
    axes[0].set_ylabel("Balanced Detection Accuracy")
    axes[0].set_title("Byzantine-Client Detection Accuracy")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()

    axes[1].set_xlabel("Byzantine Client Ratio (%)")
    axes[1].set_ylabel("L2 Divergence from Clean Baseline")
    axes[1].set_title("Global Model Divergence vs Clean Model")
    axes[1].legend()

    plt.tight_layout()
    png_path = reports_dir / "byzantine_sweep.png"
    plt.savefig(png_path, dpi=150)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", default=["FedAvg", "Krum", "FLTrust"])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_sweep(args.strategies, args.ratios, args.num_clients, args.seed)
    export_and_plot(results, PROJECT_ROOT / "reports")
