#!/usr/bin/env python3
"""
scripts/benchmark_ablation.py — AURA Ablation Study & Metrics Benchmark
=========================================================================

Implements the Tier 1 Ablation Study (Doc §5.1.4) and the metric set
required by Doc §5.1.2, evaluated on the final 20% test split of
NF-UNSW-NB15-v3 (native real attack traffic, no synthetic injection).

Doc Config -> Mode mapping
--------------------------
  A  Autoencoder Only          — Statistical tripwire in isolation
  B  GraphSAGE Only            — Structural validator in isolation
  C  AE + GraphSAGE (no EMA)   — Sequential cascade (the production pipeline)
  D  AE + GraphSAGE + EMA      — Cascade + EMA temporal persistence gate

  Config E ("Full AURA pipeline") is intentionally NOT implemented here.
  E additionally requires FLTrust federated aggregation and the HITL
  response engine, neither of which changes per-node detection scores —
  they operate on the *training* process and the *response* layer,
  respectively. Faking a distinct E result without those components
  would misrepresent what was actually measured, so it is omitted.
  See the printed report footer for this caveat.

Corrections made vs. the previous version of this script
----------------------------------------------------------
  1. AE residual now uses MSE (mean squared error), matching
     FlowAutoencoder.anomaly_score() in aura/models.py and the doc's
     stated "Anomaly score: Reconstruction error (MSE per flow)" —
     the previous version silently used MAE, putting its calibrated
     threshold on a different scale than the rest of the codebase.
  2. Removed the bogus INPUT_DIM = 43 constant. Feature width is read
     once from cfg.FEATURE_DIM (47) and cross-checked against the
     loaded model.
  3. Removed the untrained "Parallel Fusion" mode. It required a
     learned projection layer that was never actually trained in the
     prior script (eval()/no_grad() throughout), making its output
     statistically meaningless. Replaced with Mode D (EMA persistence),
     which is implementable from existing config (cfg.EMA_ALPHA,
     cfg.EMA_SIGMA_MULTIPLIER, cfg.EMA_WARMUP_BATCHES,
     cfg.K_CONSECUTIVE_READINGS) and is what the doc actually lists as
     Config D.
  4. Scaler consistency: if saved_models/scaler.joblib exists (the one
     train.py actually fit the model against), it is loaded and reused
     instead of being re-fit from scratch on a possibly different
     --load-fraction slice of the CSV.
  5. Metrics now include ROC-AUC and PR-AUC per Doc §5.1.2, which calls
     out PR-AUC as the metric to emphasize given class imbalance.
     Cascade/persistence modes (C, D) do not have a single natural
     continuous decision function (gated nodes never receive a GNN
     score), so their AUC values are explicitly marked approximate —
     see compute_metrics() docstring.

Exports
-------
  reports/ablation_results.csv
  reports/ablation_results.json

Usage
-----
  python scripts/benchmark_ablation.py
  python scripts/benchmark_ablation.py --load-fraction 0.5
  python scripts/benchmark_ablation.py --bundle saved_models/aura_bundle.pth
"""

import argparse
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Project imports ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from aura.data_loader import CICIDSDataLoader, CSV_FILES
from aura.models import AURAModelBundle, AuraSTGNN, FlowAutoencoder

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — labels and AE residual (MSE, matching aura/models.py)
# ─────────────────────────────────────────────────────────────────────────────

def derive_node_labels(edge_labels: torch.Tensor, edge_index: torch.Tensor,
                        num_nodes: int, device: torch.device) -> torch.Tensor:
    """
    Convert edge-level binary labels to node-level labels.

    Convention (matches train_stgnn() in train.py):
        A node is labelled 1 (attack) if ANY incident edge carries label 1.
    """
    node_labels = torch.zeros(num_nodes, dtype=torch.long, device=device)
    if edge_labels.sum() > 0:
        attack_mask = edge_labels.bool()
        src = edge_index[0][attack_mask]
        dst = edge_index[1][attack_mask]
        node_labels[src] = 1
        node_labels[dst] = 1
    return node_labels


def compute_ae_node_residual(
    autoencoder: FlowAutoencoder,
    edge_attr: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Per-node mean-squared AE reconstruction error.

    Uses MSE per edge (matching FlowAutoencoder.anomaly_score's own
    convention), then scatter-means the edge residuals onto incident nodes.

    Returns: [N] tensor of per-node mean AE error.
    """
    edge_attr = edge_attr.to(device)
    with torch.no_grad():
        x_hat, _ = autoencoder(edge_attr)
        edge_residual = ((edge_attr - x_hat) ** 2).mean(dim=1)  # [E], MSE per edge

    node_error = torch.zeros(num_nodes, device=device)
    node_count = torch.zeros(num_nodes, device=device)

    src, dst = edge_index[0].to(device), edge_index[1].to(device)
    node_error.scatter_add_(0, src, edge_residual)
    node_error.scatter_add_(0, dst, edge_residual)
    node_count.scatter_add_(0, src, torch.ones_like(edge_residual))
    node_count.scatter_add_(0, dst, torch.ones_like(edge_residual))

    node_count = node_count.clamp(min=1.0)
    return node_error / node_count  # [N]


def calibrate_ae_threshold(
    autoencoder: FlowAutoencoder,
    calibration_windows: list,
    device: torch.device,
    percentile: float = 95.0,
) -> tuple:
    """
    Estimate the AE anomaly threshold from benign-dominated calibration windows.

    Returns (threshold, mean, std) — mean/std are additionally needed to
    seed the EMA tracker used by Mode D.
    """
    all_residuals = []
    for graph, labels in calibration_windows:
        edge_attr = graph["edge_attr"].to(device)
        edge_index = graph["edge_index"].to(device)
        num_nodes = graph["x"].shape[0]

        node_res = compute_ae_node_residual(
            autoencoder, edge_attr, edge_index, num_nodes, device
        )
        node_labels = derive_node_labels(labels, graph["edge_index"], num_nodes, device)
        benign_mask = node_labels == 0
        if benign_mask.sum() > 0:
            all_residuals.append(node_res[benign_mask].cpu().numpy())

    if not all_residuals:
        logger.warning("No benign nodes found for calibration — using fallback threshold 0.05")
        return 0.05, 0.0, 0.05

    residuals = np.concatenate(all_residuals)
    threshold = float(np.percentile(residuals, percentile))
    mean = float(residuals.mean())
    std = float(residuals.std())
    logger.info(
        f"AE threshold calibrated: {percentile}th percentile = {threshold:.6f}  "
        f"(mean={mean:.6f}, std={std:.6f}, from {len(residuals)} benign node samples)"
    )
    return threshold, mean, std


# ─────────────────────────────────────────────────────────────────────────────
# EMA Temporal Persistence Tracker (Mode D)
# ─────────────────────────────────────────────────────────────────────────────

class NodeEMATracker:
    """
    Per-node EMA mean/std of AE residual, with consecutive-alert counting.

    Implements the doc's described mechanism (config.py: EMA_ALPHA,
    EMA_SIGMA_MULTIPLIER, EMA_WARMUP_BATCHES, K_CONSECUTIVE_READINGS),
    approximated at window granularity (the doc's TEMPORAL_WINDOW_SECONDS
    cannot be applied directly since this script only has window-index
    ordering, not wall-clock timestamps per window).

    A node "fires" (single-window alert) if its residual exceeds
    EMA_mean + EMA_SIGMA_MULTIPLIER * EMA_std. A node only counts as a
    final positive prediction once it has fired in K_CONSECUTIVE_READINGS
    consecutive windows it has appeared in — this is the persistence
    mechanism meant to catch low-and-slow attacks while not raising
    single-spike false alarms (Doc H3).
    """

    def __init__(self, init_mean: float, init_std: float,
                 alpha: float = cfg.EMA_ALPHA,
                 sigma_mult: float = cfg.EMA_SIGMA_MULTIPLIER,
                 warmup: int = cfg.EMA_WARMUP_BATCHES,
                 k_consecutive: int = cfg.K_CONSECUTIVE_READINGS):
        self.alpha = alpha
        self.sigma_mult = sigma_mult
        self.warmup = warmup
        self.k_consecutive = k_consecutive

        self._mean: dict = {}
        self._var: dict = {}
        self._streak: dict = {}
        self._seen: dict = {}
        self._init_mean = init_mean
        self._init_std = init_std

    def update_and_check(self, node_id: int, residual: float) -> bool:
        """
        Update this node's EMA state with a new residual reading and return
        True if the node should be predicted POSITIVE this window (i.e. it
        has met the K-consecutive-fire persistence requirement).
        """
        seen = self._seen.get(node_id, 0)
        mean = self._mean.get(node_id, self._init_mean)
        var = self._var.get(node_id, self._init_std ** 2)

        fired = False
        if seen >= self.warmup:
            std = max(var ** 0.5, 1e-8)
            fired = residual > (mean + self.sigma_mult * std)

        # Update EMA mean/variance (always — even during warm-up)
        delta = residual - mean
        new_mean = mean + self.alpha * delta
        new_var = (1 - self.alpha) * (var + self.alpha * delta * delta)
        self._mean[node_id] = new_mean
        self._var[node_id] = new_var
        self._seen[node_id] = seen + 1

        streak = self._streak.get(node_id, 0)
        streak = streak + 1 if fired else 0
        self._streak[node_id] = streak

        return streak >= self.k_consecutive


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_score: np.ndarray = None,
                     auc_is_approximate: bool = False) -> dict:
    """
    Precision, Recall, F1 (binary, positive class), Macro-F1, and FPR from
    the confusion matrix, plus ROC-AUC / PR-AUC if a continuous score is
    supplied (Doc §5.1.2 — PR-AUC should be emphasised given class imbalance).

    For cascade/persistence modes there is no single natural continuous
    decision function (gated-out nodes never receive a model score), so
    callers pass auc_is_approximate=True and the returned dict flags this.
    """
    precision = precision_score(y_true, y_pred, zero_division=0.0)
    recall = recall_score(y_true, y_pred, zero_division=0.0)
    f1_binary = f1_score(y_true, y_pred, zero_division=0.0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0.0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "Precision": round(precision, 6),
        "Recall": round(recall, 6),
        "F1": round(f1_binary, 6),
        "Macro-F1": round(macro_f1, 6),
        "FPR": round(fpr, 6),
    }

    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["ROC-AUC"] = round(float(roc_auc_score(y_true, y_score)), 6)
            metrics["PR-AUC"] = round(float(average_precision_score(y_true, y_score)), 6)
        except ValueError:
            metrics["ROC-AUC"] = None
            metrics["PR-AUC"] = None
    else:
        metrics["ROC-AUC"] = None
        metrics["PR-AUC"] = None

    metrics["AUC_Approximate"] = bool(auc_is_approximate)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Data Collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_test_windows(
    loader: CICIDSDataLoader,
    scaler,
    test_fraction: float = 0.20,
) -> tuple:
    """
    Stream ALL windows in order, then return only the final `test_fraction`
    portion (preserving order — required by Mode D's EMA tracker), plus a
    small calibration set from the start of the training portion.
    """
    logger.info("Streaming all graph windows to isolate the test split …")

    all_windows = []
    for graph, labels in loader.stream_graphs(scaler):
        graph_copy = {
            "x": graph["x"].clone(),
            "edge_index": graph["edge_index"].clone(),
            "edge_attr": graph["edge_attr"].clone(),
            "window_id": graph["window_id"],
        }
        all_windows.append((graph_copy, labels.clone()))

    total = len(all_windows)
    if total == 0:
        raise RuntimeError(
            "No graph windows produced. Check that the dataset CSV exists at "
            f"{cfg.CSV_DIR} and DATA_LOAD_FRACTION ({cfg.DATA_LOAD_FRACTION}) is sufficient."
        )

    test_start = int(total * (1.0 - test_fraction))
    train_windows = all_windows[:test_start]
    test_windows = all_windows[test_start:]

    n_calib = max(5, int(len(train_windows) * 0.10))
    calibration_windows = train_windows[:n_calib]

    logger.info(
        f"Total windows: {total} | Train: {len(train_windows)} | "
        f"Calibration (from train): {len(calibration_windows)} | Test: {len(test_windows)}"
    )

    total_edges = sum(labels.numel() for _, labels in test_windows)
    total_attacks = sum(labels.sum().item() for _, labels in test_windows)
    logger.info(
        f"Test set attack ratio: {total_attacks}/{total_edges} "
        f"= {total_attacks / max(total_edges, 1):.2%}"
    )

    return calibration_windows, test_windows


# ─────────────────────────────────────────────────────────────────────────────
# Ablation Mode Runners
# ─────────────────────────────────────────────────────────────────────────────

def run_mode_a(ae: FlowAutoencoder, test_windows: list, threshold: float,
               device: torch.device) -> tuple:
    """Mode A — Autoencoder Only (Doc Config A)."""
    all_y_true, all_y_pred, all_y_score = [], [], []

    for graph, edge_labels in test_windows:
        edge_index = graph["edge_index"].to(device)
        edge_attr = graph["edge_attr"].to(device)
        num_nodes = graph["x"].shape[0]

        node_labels = derive_node_labels(edge_labels, graph["edge_index"], num_nodes, device)
        node_residual = compute_ae_node_residual(ae, edge_attr, edge_index, num_nodes, device)
        y_pred = (node_residual > threshold).long()

        all_y_true.append(node_labels.cpu().numpy())
        all_y_pred.append(y_pred.cpu().numpy())
        all_y_score.append(node_residual.cpu().numpy())

    return (np.concatenate(all_y_true), np.concatenate(all_y_pred),
            np.concatenate(all_y_score))


def run_mode_b(stgnn: AuraSTGNN, test_windows: list, device: torch.device,
               gnn_threshold: float = 0.5) -> tuple:
    """Mode B — GraphSAGE Only (Doc Config B). AE is bypassed entirely."""
    all_y_true, all_y_pred, all_y_score = [], [], []

    for graph, edge_labels in test_windows:
        x = graph["x"].to(device)
        edge_index = graph["edge_index"].to(device)
        num_nodes = x.shape[0]

        node_labels = derive_node_labels(edge_labels, graph["edge_index"], num_nodes, device)
        with torch.no_grad():
            scores, _ = stgnn(x, edge_index)
        y_pred = (scores > gnn_threshold).long()

        all_y_true.append(node_labels.cpu().numpy())
        all_y_pred.append(y_pred.cpu().numpy())
        all_y_score.append(scores.cpu().numpy())

    return (np.concatenate(all_y_true), np.concatenate(all_y_pred),
            np.concatenate(all_y_score))


def run_mode_c(ae: FlowAutoencoder, stgnn: AuraSTGNN, test_windows: list,
               threshold: float, device: torch.device,
               gnn_threshold: float = 0.5) -> tuple:
    """
    Mode C — Sequential Cascade (Doc Config C / current production pipeline).

    AE gates: nodes below threshold are predicted negative immediately and
    NEVER reach GraphSAGE — this is the exact "AE as gatekeeper" blind spot
    the doc calls out in §3.1.
    """
    all_y_true, all_y_pred, all_y_score = [], [], []

    for graph, edge_labels in test_windows:
        x = graph["x"].to(device)
        edge_index = graph["edge_index"].to(device)
        edge_attr = graph["edge_attr"].to(device)
        num_nodes = x.shape[0]

        node_labels = derive_node_labels(edge_labels, graph["edge_index"], num_nodes, device)
        node_residual = compute_ae_node_residual(ae, edge_attr, edge_index, num_nodes, device)
        ae_flagged = node_residual > threshold

        y_pred = torch.zeros(num_nodes, dtype=torch.long, device=device)
        # Score for AUC: residual for gated-out nodes (always < threshold,
        # so they rank as "more negative"), GNN score for evaluated nodes.
        # NOTE: these two value ranges are not directly comparable — see
        # compute_metrics()'s auc_is_approximate flag.
        y_score = node_residual.clone()

        if ae_flagged.sum() > 0:
            with torch.no_grad():
                gnn_scores, _ = stgnn(x, edge_index)
            gnn_decision = (gnn_scores > gnn_threshold).long()
            y_pred[ae_flagged] = gnn_decision[ae_flagged]
            y_score[ae_flagged] = gnn_scores[ae_flagged]

        all_y_true.append(node_labels.cpu().numpy())
        all_y_pred.append(y_pred.cpu().numpy())
        all_y_score.append(y_score.cpu().numpy())

    return (np.concatenate(all_y_true), np.concatenate(all_y_pred),
            np.concatenate(all_y_score))


def run_mode_d(ae: FlowAutoencoder, stgnn: AuraSTGNN, test_windows: list,
               ema_mean: float, ema_std: float, device: torch.device,
               gnn_threshold: float = 0.5) -> tuple:
    """
    Mode D — Sequential Cascade + EMA Temporal Persistence (Doc Config D).

    Same AE-gate-then-GraphSAGE cascade as Mode C, except the AE gate is now
    the EMA persistence tracker (NodeEMATracker) instead of a flat
    percentile threshold: a node must exceed EMA_mean + 3*EMA_std for
    K_CONSECUTIVE_READINGS consecutive windows before it triggers the
    GraphSAGE check. This is the mechanism the doc credits with detecting
    low-and-slow attacks (Hypothesis H3) without raising FPR on single-spike
    benign noise.

    Windows must be processed IN ORDER for the EMA state to be meaningful —
    collect_test_windows() preserves stream order, so this holds as long as
    test_windows is passed through unmodified.
    """
    tracker = NodeEMATracker(init_mean=ema_mean, init_std=ema_std)
    all_y_true, all_y_pred, all_y_score = [], [], []

    for graph, edge_labels in test_windows:
        x = graph["x"].to(device)
        edge_index = graph["edge_index"].to(device)
        edge_attr = graph["edge_attr"].to(device)
        num_nodes = x.shape[0]

        node_labels = derive_node_labels(edge_labels, graph["edge_index"], num_nodes, device)
        node_residual = compute_ae_node_residual(ae, edge_attr, edge_index, num_nodes, device)
        residual_np = node_residual.detach().cpu().numpy()

        ema_flagged = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        for node_id in range(num_nodes):
            if tracker.update_and_check(node_id, float(residual_np[node_id])):
                ema_flagged[node_id] = True

        y_pred = torch.zeros(num_nodes, dtype=torch.long, device=device)
        y_score = node_residual.clone()

        if ema_flagged.sum() > 0:
            with torch.no_grad():
                gnn_scores, _ = stgnn(x, edge_index)
            gnn_decision = (gnn_scores > gnn_threshold).long()
            y_pred[ema_flagged] = gnn_decision[ema_flagged]
            y_score[ema_flagged] = gnn_scores[ema_flagged]

        all_y_true.append(node_labels.cpu().numpy())
        all_y_pred.append(y_pred.cpu().numpy())
        all_y_score.append(y_score.cpu().numpy())

    return (np.concatenate(all_y_true), np.concatenate(all_y_pred),
            np.concatenate(all_y_score))


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_results(results: dict, reports_dir: Path) -> pd.DataFrame:
    reports_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "Mode"

    csv_path = reports_dir / "ablation_results.csv"
    json_path = reports_dir / "ablation_results.json"
    df.to_csv(csv_path)
    df.to_json(json_path, orient="index", indent=2)

    logger.info(f"Results exported → {csv_path}")
    logger.info(f"Results exported → {json_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def load_shared_scaler(loader: CICIDSDataLoader):
    """
    Prefer the scaler train.py actually fit the loaded bundle against
    (saved_models/scaler.joblib) over re-fitting a fresh one, which could
    silently differ if --load-fraction doesn't match the training run.
    """
    scaler_path = cfg.MODELS_DIR / "scaler.joblib"
    if scaler_path.exists():
        import joblib
        logger.info(f"Loading shared scaler from {scaler_path} (matches training run) …")
        scaler = joblib.load(scaler_path)
        # loader still needs _feature_cols populated for downstream calls
        loader._load_csv(CSV_FILES[0])
        return scaler
    logger.warning(
        "No saved_models/scaler.joblib found — fitting a fresh scaler from "
        "--load-fraction. This may not exactly match the bundle's training scaler."
    )
    return loader.fit_scaler()


def main():
    parser = argparse.ArgumentParser(
        description="AURA Ablation Study — Configs A-D evaluation on NF-UNSW-NB15-v3 test split"
    )
    parser.add_argument("--bundle", type=str, default=str(cfg.MODELS_DIR / "aura_bundle.pth"))
    parser.add_argument("--load-fraction", type=float, default=cfg.DATA_LOAD_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--ae-percentile", type=float, default=95.0)
    parser.add_argument("--gnn-threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── 1. Load pre-trained model bundle ─────────────────────────────────────
    bundle_path = Path(args.bundle)
    bundle = AURAModelBundle()

    if bundle_path.exists():
        logger.info(f"Loading pre-trained bundle from {bundle_path} …")
        state = torch.load(bundle_path, map_location=device, weights_only=True)
        bundle.load_state_dict(state)
        logger.info("✓ Model bundle loaded successfully.")
    else:
        logger.warning(
            f"Bundle not found at {bundle_path}. Running with randomly initialised "
            "weights — results will not be meaningful. Train first with: python train.py"
        )

    ae = bundle.autoencoder.to(device).eval()
    stgnn = bundle.stgnn.to(device).eval()

    logger.info(f"AE params: {ae.count_params():,}  |  STGNN params: {stgnn.count_params():,}")

    actual_feature_dim = ae.encoder[0].in_features
    if actual_feature_dim != cfg.FEATURE_DIM:
        logger.warning(
            f"Loaded model's feature dim ({actual_feature_dim}) does not match "
            f"cfg.FEATURE_DIM ({cfg.FEATURE_DIM}) — bundle may be stale."
        )
    logger.info(f"Model feature dimension: {actual_feature_dim}")

    # ── 2. Load data and isolate test split ──────────────────────────────────
    logger.info("Initialising data loader …")
    loader = CICIDSDataLoader(load_fraction=args.load_fraction)
    scaler = load_shared_scaler(loader)

    calibration_windows, test_windows = collect_test_windows(
        loader, scaler, test_fraction=args.test_fraction
    )
    if not test_windows:
        logger.error("Test split is empty. Increase --load-fraction or check the dataset.")
        sys.exit(1)

    # ── 3. Calibrate AE threshold + EMA seed stats ───────────────────────────
    ae_threshold, ema_mean, ema_std = calibrate_ae_threshold(
        ae, calibration_windows, device, percentile=args.ae_percentile
    )

    # ── 4. Run the 4 ablation modes ──────────────────────────────────────────
    results = OrderedDict()

    print(f"\n{'=' * 72}")
    print(f"  AURA Ablation Study — {len(test_windows)} test windows on {device}")
    print(f"  AE threshold (p{args.ae_percentile:.0f}): {ae_threshold:.6f}  "
          f"(EMA seed mean={ema_mean:.6f}, std={ema_std:.6f})")
    print(f"  GNN threshold: {args.gnn_threshold}")
    print(f"{'=' * 72}\n")

    def run_and_record(name, fn, auc_approx):
        t0 = time.time()
        logger.info(f"▶ Running {name} …")
        y_true, y_pred, y_score = fn()
        metrics = compute_metrics(y_true, y_pred, y_score, auc_is_approximate=auc_approx)
        metrics["Samples"] = len(y_true)
        metrics["Time_s"] = round(time.time() - t0, 2)
        results[name] = metrics
        logger.info(f"  {name} done in {metrics['Time_s']}s — {metrics}")

    run_and_record(
        "A: Autoencoder Only",
        lambda: run_mode_a(ae, test_windows, ae_threshold, device),
        auc_approx=False,
    )
    run_and_record(
        "B: GraphSAGE Only",
        lambda: run_mode_b(stgnn, test_windows, device, args.gnn_threshold),
        auc_approx=False,
    )
    run_and_record(
        "C: Sequential Cascade (AE+GraphSAGE, no EMA)",
        lambda: run_mode_c(ae, stgnn, test_windows, ae_threshold, device, args.gnn_threshold),
        auc_approx=True,
    )
    run_and_record(
        "D: Cascade + EMA Persistence",
        lambda: run_mode_d(ae, stgnn, test_windows, ema_mean, ema_std, device, args.gnn_threshold),
        auc_approx=True,
    )

    # ── 5. Export + print ────────────────────────────────────────────────────
    reports_dir = PROJECT_ROOT / "reports"
    df = export_results(results, reports_dir)

    print(f"\n{'=' * 72}")
    print("  ABLATION STUDY RESULTS")
    print(f"{'=' * 72}")
    print(df.to_string())
    print(f"{'=' * 72}")
    print(
        "\n  NOTE: Config E ('Full AURA pipeline') is not reported. It additionally\n"
        "  requires FLTrust federated aggregation and the HITL response engine,\n"
        "  neither of which alters per-node detection scores measured here — see\n"
        "  module docstring for details.\n"
        "  NOTE: ROC-AUC/PR-AUC for Modes C and D are approximate (flagged\n"
        "  AUC_Approximate=True): gated-out nodes never receive a GraphSAGE score,\n"
        "  so no single continuous decision function exists across all nodes.\n"
    )
    print(f"  ✓ Exported to:")
    print(f"    • {reports_dir / 'ablation_results.csv'}")
    print(f"    • {reports_dir / 'ablation_results.json'}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()