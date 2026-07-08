#!/usr/bin/env python3
"""
scripts/ablation_sweep.py
--------------------------
Threshold-based visualizations for the AURA ablation study (Modes A-D
from scripts/benchmark_ablation.py). Split into two kinds of sweep
because "threshold" doesn't mean the same thing for every mode:

  Modes A & B (Autoencoder Only / GraphSAGE Only) each produce ONE
  continuous decision score per node, so we get an EXACT sweep --
  ROC curve, PR curve, and a residual-distribution plot -- computed
  directly from a single run's saved scores. No re-running needed,
  and nothing here is approximate.

  Modes C & D (cascades) gate nodes before scoring -- a node the AE
  (or EMA tracker) doesn't flag never reaches GraphSAGE, so there is
  no single continuous score across all nodes (this is exactly why
  benchmark_ablation.py marks their AUC as approximate). Their real
  "threshold" is the upstream gate parameter instead:
    - Mode C : --ae-percentile      (AE cutoff deciding who reaches GraphSAGE)
    - Mode D : --ema-sigma / --k-consecutive (EMA persistence gate)
  For these, the cascade is actually re-run once per parameter value
  and the resulting Precision/Recall/F1/FPR are recorded -- a genuine
  sensitivity sweep, not a post-hoc slider on one run's output.

Exports (to reports/)
----------------------
  ablation_ab_scores.npz     raw y_true/y_score for Modes A & B
  ablation_ab_sweep.png      ROC + PR + residual distribution, A & B
  ablation_c_sweep.csv/.png  Mode C metrics vs --ae-percentile
  ablation_d_sigma_sweep.*   Mode D metrics vs --ema-sigma (k fixed)
  ablation_d_k_sweep.*       Mode D metrics vs --k-consecutive (sigma fixed)

Usage
-----
  python scripts/ablation_sweep.py
  python scripts/ablation_sweep.py --bundle saved_models/aura_bundle.pth --load-fraction 0.5
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import config as cfg
from aura.data_loader import CICIDSDataLoader
from aura.models import AURAModelBundle

from benchmark_ablation import (
    derive_node_labels,
    compute_ae_node_residual,
    calibrate_gnn_platt,
    compute_metrics,
    collect_test_windows,
    load_shared_scaler,
    run_mode_a,
    run_mode_b,
    run_mode_c,
    run_mode_d,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_calibration_residuals(ae, calibration_windows, device):
    """Raw benign-node AE residuals from calibration windows, computed ONCE
    so every percentile threshold below is just a np.percentile() call
    instead of a fresh pass over the calibration set."""
    all_residuals = []
    for graph, labels in calibration_windows:
        edge_attr = graph["edge_attr"].to(device)
        edge_index = graph["edge_index"].to(device)
        num_nodes = graph["x"].shape[0]
        node_res = compute_ae_node_residual(ae, edge_attr, edge_index, num_nodes, device)
        node_labels = derive_node_labels(labels, graph["edge_index"], num_nodes, device)
        benign_mask = node_labels == 0
        if benign_mask.sum() > 0:
            all_residuals.append(node_res[benign_mask].cpu().numpy())
    if not all_residuals:
        logger.warning("No benign calibration nodes found -- using fallback residual array.")
        return np.array([0.05])
    return np.concatenate(all_residuals)


def plot_ab_sweep(y_true_a, y_score_a, y_true_b, y_score_b, calib_residuals, reports_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for name, y_true, y_score, color in [
        ("Mode A (AE)", y_true_a, y_score_a, "steelblue"),
        ("Mode B (GraphSAGE)", y_true_b, y_score_b, "darkorange"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})", color=color)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        axes[1].plot(rec, prec, label=name, color=color)

    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC — Modes A & B (exact)"); axes[0].legend()

    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall — Modes A & B (exact)"); axes[1].legend()

    ax = axes[2]
    benign_res = y_score_a[y_true_a == 0]
    attack_res = y_score_a[y_true_a == 1]
    ax.hist(benign_res, bins=50, alpha=0.5, label="Benign", color="steelblue", density=True)
    ax.hist(attack_res, bins=50, alpha=0.5, label="Attack", color="crimson", density=True)
    for pct in (90, 95, 99):
        val = float(np.percentile(calib_residuals, pct))
        ax.axvline(val, linestyle="--", label=f"P{pct}={val:.4f}")
    ax.set_xlabel("AE Reconstruction Error (test set)"); ax.set_ylabel("Density")
    ax.set_title("Mode A Residual Distribution"); ax.legend(fontsize=7)

    plt.tight_layout()
    save_path = reports_dir / "ablation_ab_sweep.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


def plot_metric_sweep(df, x_col, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for metric, color in [("Precision", "steelblue"), ("Recall", "crimson"),
                           ("F1", "green"), ("FPR", "darkorange")]:
        ax.plot(df[x_col], df[metric], marker="o", label=metric, color=color)
    ax.set_xlabel(x_col)
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description="AURA ablation threshold/sensitivity sweeps")
    parser.add_argument("--bundle", type=str, default=str(cfg.MODELS_DIR / "aura_bundle.pth"))
    parser.add_argument("--load-fraction", type=float, default=cfg.DATA_LOAD_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=cfg.TEST_SPLIT_FRACTION)
    parser.add_argument("--gnn-threshold", type=float, default=0.5)
    parser.add_argument("--ae-percentiles", type=float, nargs="+", default=[85, 90, 95, 97, 99])
    parser.add_argument("--ema-sigmas", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0])
    parser.add_argument("--k-consecutive-values", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    bundle_path = Path(args.bundle)
    bundle = AURAModelBundle()
    if bundle_path.exists():
        state = torch.load(bundle_path, map_location=device, weights_only=True)
        bundle.load_state_dict(state)
        logger.info(f"Loaded bundle from {bundle_path}")
    else:
        logger.warning(f"Bundle not found at {bundle_path} -- results will not be meaningful.")

    ae = bundle.autoencoder.to(device).eval()
    stgnn = bundle.stgnn.to(device).eval()

    loader = CICIDSDataLoader(load_fraction=args.load_fraction)
    scaler = load_shared_scaler(loader)
    calibration_windows, test_windows = collect_test_windows(
        loader, scaler, test_fraction=args.test_fraction
    )
    if not test_windows:
        logger.error("Test split is empty. Increase --load-fraction or check the dataset.")
        sys.exit(1)

    residuals = get_calibration_residuals(ae, calibration_windows, device)
    ema_mean, ema_std = float(residuals.mean()), float(residuals.std())
    platt_scaler = calibrate_gnn_platt(stgnn, calibration_windows, device)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    # ── Modes A & B: single run each, EXACT continuous sweep ────────────────
    logger.info("Running Mode A (single pass) …")
    base_threshold = float(np.percentile(residuals, 95.0))
    y_true_a, _, y_score_a = run_mode_a(ae, test_windows, base_threshold, device)

    logger.info("Running Mode B (single pass) …")
    y_true_b, _, y_score_b = run_mode_b(stgnn, platt_scaler, test_windows, device, args.gnn_threshold)

    np.savez(
        reports_dir / "ablation_ab_scores.npz",
        y_true_a=y_true_a, y_score_a=y_score_a,
        y_true_b=y_true_b, y_score_b=y_score_b,
    )
    print(f"Saved {reports_dir / 'ablation_ab_scores.npz'}")
    plot_ab_sweep(y_true_a, y_score_a, y_true_b, y_score_b, residuals, reports_dir)

    # ── Mode C: real re-run at each AE gate percentile ──────────────────────
    c_rows = []
    for pct in args.ae_percentiles:
        t = float(np.percentile(residuals, pct))
        logger.info(f"Running Mode C at AE percentile={pct} (threshold={t:.6f}) …")
        y_true, y_pred, y_score = run_mode_c(
            ae, stgnn, platt_scaler, test_windows, t, device, args.gnn_threshold
        )
        m = compute_metrics(y_true, y_pred, y_score, auc_is_approximate=True)
        m["ae_percentile"] = pct
        m["threshold"] = t
        c_rows.append(m)
    df_c = pd.DataFrame(c_rows)
    df_c.to_csv(reports_dir / "ablation_c_sweep.csv", index=False)
    print(f"Saved {reports_dir / 'ablation_c_sweep.csv'}")
    plot_metric_sweep(
        df_c, "ae_percentile", "Mode C: Metrics vs AE Gate Percentile",
        reports_dir / "ablation_c_sweep.png",
    )

    # ── Mode D: real re-run sweeping ema_sigma (k fixed), then k (sigma fixed) ──
    default_k = args.k_consecutive_values[len(args.k_consecutive_values) // 2]
    d_sigma_rows = []
    for sigma in args.ema_sigmas:
        logger.info(f"Running Mode D at ema_sigma={sigma}, k={default_k} …")
        y_true, y_pred, y_score = run_mode_d(
            ae, stgnn, platt_scaler, test_windows, ema_mean, ema_std, device,
            args.gnn_threshold, ema_sigma_mult=sigma, k_consecutive=default_k,
        )
        m = compute_metrics(y_true, y_pred, y_score, auc_is_approximate=True)
        m["ema_sigma"] = sigma
        m["k_consecutive"] = default_k
        d_sigma_rows.append(m)
    df_d_sigma = pd.DataFrame(d_sigma_rows)
    df_d_sigma.to_csv(reports_dir / "ablation_d_sigma_sweep.csv", index=False)
    print(f"Saved {reports_dir / 'ablation_d_sigma_sweep.csv'}")
    plot_metric_sweep(
        df_d_sigma, "ema_sigma", f"Mode D: Metrics vs EMA Sigma (k={default_k})",
        reports_dir / "ablation_d_sigma_sweep.png",
    )

    default_sigma = args.ema_sigmas[len(args.ema_sigmas) // 2]
    d_k_rows = []
    for k in args.k_consecutive_values:
        logger.info(f"Running Mode D at ema_sigma={default_sigma}, k={k} …")
        y_true, y_pred, y_score = run_mode_d(
            ae, stgnn, platt_scaler, test_windows, ema_mean, ema_std, device,
            args.gnn_threshold, ema_sigma_mult=default_sigma, k_consecutive=k,
        )
        m = compute_metrics(y_true, y_pred, y_score, auc_is_approximate=True)
        m["ema_sigma"] = default_sigma
        m["k_consecutive"] = k
        d_k_rows.append(m)
    df_d_k = pd.DataFrame(d_k_rows)
    df_d_k.to_csv(reports_dir / "ablation_d_k_sweep.csv", index=False)
    print(f"Saved {reports_dir / 'ablation_d_k_sweep.csv'}")
    plot_metric_sweep(
        df_d_k, "k_consecutive", f"Mode D: Metrics vs K-Consecutive (sigma={default_sigma})",
        reports_dir / "ablation_d_k_sweep.png",
    )

    print("\nAll sweeps complete. See reports/ for CSVs and PNGs.")


if __name__ == "__main__":
    main()
