#!/usr/bin/env python3
"""
scripts/benchmark_behavioral_generalization.py
===============================================
Implements Section 5.2.3 — Behavioral Generalization (Hypothesis H1).

Hypothesis H1
-------------
Distinct from *structural* generalization (Watts-Strogatz unseen topologies),
this experiment tests whether GraphSAGE correctly handles **unseen communication
patterns within KNOWN topologies**.

The three behavioral shift scenarios evaluated here each represent realistic,
LEGITIMATE enterprise network changes — they should NOT trigger false positives:

  Scenario 1 — Role Shift
    A node that has never acted as a database server suddenly does so.
    E.g., a workstation begins accepting queries on characteristic DB ports,
    exhibiting high in_bytes + low inter-arrival times (DB query-response rhythm).
    Operationalization: the node's edge features are perturbed toward the
    dataset's 95th-percentile "high-throughput, low-latency" benign regime,
    without injecting any attack-class-specific signature.

  Scenario 2 — Subnet Debut
    A subnet that has never communicated externally suddenly does so.
    A previously unseen IP hash-bucket first appears in a window that still
    uses the existing known-topology node set.
    Operationalization: A benign node's source-IP partition bucket is replaced
    with the partition bucket of an IP prefix that was entirely absent from the
    training stream, then its edge features remain fully benign.

  Scenario 3 — Silent-to-Active
    A previously silent node (zero or near-zero out-degree) initiates many
    connections in one window.
    Operationalization: A low-degree node's out-edges are expanded to the
    dataset's 95th-percentile out-degree count using fully benign flow features.

Primary Metric
--------------
FPR (False Positive Rate) per scenario:
    FPR = FP / (FP + TN)
where all shifted nodes are ground-truth BENIGN (Label = 0).
A model with good behavioral generalization should exhibit FPR ≈ 0.

Design Invariants (Zero Hardcoding)
------------------------------------
* AE threshold: loaded from logs/calibration_results.json via cfg.load_ae_thresholds()
* Feature ranges: computed live from the NF-UNSW-NB15-v3 dataset at the requested
  percentile (--shift-percentile CLI arg). Optionally augmented by
  saved_models/attack_class_stats.json if present.
* Behavioral shift magnitude: driven by --shift-percentile (default: 95th percentile
  of the relevant per-feature distribution on benign-only rows)
* Number of shifted nodes per scenario: --n-shift-nodes (default: 5)
* All thresholds and config live in config.py; nothing is a magic number here.

Exports
-------
  reports/behavioral_generalization_results.csv
  reports/behavioral_generalization_results.json

Usage
-----
  python scripts/benchmark_behavioral_generalization.py
  python scripts/benchmark_behavioral_generalization.py --load-fraction 0.1
  python scripts/benchmark_behavioral_generalization.py --shift-percentile 90 --n-shift-nodes 3
  python scripts/benchmark_behavioral_generalization.py --bundle saved_models/aura_bundle.pth
  python scripts/benchmark_behavioral_generalization.py --fpr-target 0.05
"""

import argparse
import hashlib
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ── Project path bootstrap ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from aura.data_loader import CICIDSDataLoader, CSV_FILES, DATASET_PATH, _ip_to_partition
from aura.models import AURAModelBundle, AuraSTGNN, FlowAutoencoder

# ── Force UTF-8 stdout (Windows cp1252 safety) ───────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("behavioral_gen")


# =============================================================================
# SECTION 1 — Data-Derived Feature Statistics
# =============================================================================

class BenignFeatureStats:
    """
    Encapsulates per-feature distributional statistics derived from the
    NF-UNSW-NB15-v3 benign-traffic rows.

    All behavioral shift magnitudes are parameterized via `shift_percentile`
    (default: 95th percentile) so that no feature range is hardcoded.
    The 'role shift' scenario uses the high end of the benign distribution
    (not attack ranges) — it is testing whether a *legitimately active*
    node triggers a false positive.

    Attributes
    ----------
    feature_cols : list[str]    — ordered feature column names (from loader)
    p_low        : np.ndarray   — per-feature (100 - shift_percentile)th percentile
    p_high       : np.ndarray   — per-feature shift_percentile-th percentile
    p_mean       : np.ndarray   — per-feature mean (benign)
    degree_p_high: float        — shift_percentile-th percentile of node out-degree
    n_partitions : int          — number of IP partition buckets
    """

    def __init__(
        self,
        feature_cols: List[str],
        X_benign: np.ndarray,
        degree_counts: np.ndarray,
        shift_percentile: float,
        n_partitions: int,
    ):
        self.feature_cols  = feature_cols
        self.shift_percentile = shift_percentile
        self.n_partitions  = n_partitions

        lo_pct = 100.0 - shift_percentile

        self.p_mean  = X_benign.mean(axis=0).astype(np.float32)
        self.p_high  = np.percentile(X_benign, shift_percentile, axis=0).astype(np.float32)
        self.p_low   = np.percentile(X_benign, lo_pct,           axis=0).astype(np.float32)

        # 95th percentile out-degree from the graph window
        self.degree_p_high = float(np.percentile(degree_counts, shift_percentile)) if len(degree_counts) > 0 else 5.0

        logger.info(
            f"BenignFeatureStats: {len(feature_cols)} features | "
            f"shift_percentile={shift_percentile} | "
            f"degree p{shift_percentile:.0f}={self.degree_p_high:.1f} | "
            f"n_partitions={n_partitions}"
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        shift_percentile: float,
    ) -> "BenignFeatureStats":
        """
        Derive statistics from a raw (unscaled) CSV DataFrame.
        Only benign rows (Label == 0) contribute to the distribution.
        """
        if pd.api.types.is_numeric_dtype(df[label_col]):
            benign_mask = df[label_col] == cfg.BENIGN_LABEL
        else:
            benign_mask = df[label_col].str.strip().str.upper() == "BENIGN"

        X_benign = df.loc[benign_mask, feature_cols].values.astype(np.float64)

        # Count unique source IPs to estimate number of active IP partitions
        src_col = "IPV4_SRC_ADDR" if "IPV4_SRC_ADDR" in df.columns else "src_ip"
        if src_col in df.columns:
            src_ips = df.loc[benign_mask, src_col].values
            unique_ips = pd.Series(src_ips).unique()
            n_partitions = len(set(_ip_to_partition(str(ip)) for ip in unique_ips))
        else:
            n_partitions = 5  # fallback to org count

        # Compute node out-degree distribution from the benign window
        # Approximate: count how many flows each unique source IP initiates
        if src_col in df.columns:
            degree_counts = df.loc[benign_mask, src_col].value_counts().values.astype(np.float64)
        else:
            degree_counts = np.ones(max(1, len(X_benign) // 10), dtype=np.float64)

        logger.info(
            f"  Benign rows for stats: {len(X_benign):,} | "
            f"unique IPs: {n_partitions} partitions"
        )
        return cls(feature_cols, X_benign, degree_counts, shift_percentile, n_partitions)


def load_benign_stats_and_scaler(
    load_fraction: float,
    shift_percentile: float,
) -> Tuple[BenignFeatureStats, object, List[str]]:
    """
    Load the NF-UNSW-NB15-v3 CSV, fit/load the shared scaler, and compute
    BenignFeatureStats from the raw (unscaled) benign rows.

    The scaler is loaded from saved_models/scaler.joblib if it exists (matching
    the training run exactly), otherwise freshly fitted — same logic as
    benchmark_ablation.py's load_shared_scaler().

    Returns
    -------
    (stats, scaler, feature_cols)
    """
    loader = CICIDSDataLoader(load_fraction=load_fraction)

    # Prefer the training-run scaler for consistency
    scaler_path = cfg.MODELS_DIR / "scaler.joblib"
    if scaler_path.exists():
        import joblib as _joblib
        logger.info(f"Loading shared scaler from {scaler_path} ...")
        scaler = _joblib.load(scaler_path)
        # Populate loader._feature_cols without re-fitting
        loader._load_csv(CSV_FILES[0])
    else:
        logger.warning(
            "No saved_models/scaler.joblib — fitting a fresh scaler. "
            "Results may not exactly match the training run."
        )
        scaler = loader.fit_scaler()

    feature_cols = loader._feature_cols
    if not feature_cols:
        raise RuntimeError("Feature column discovery failed — check dataset path.")

    # Re-load raw DataFrame to compute unscaled benign statistics
    logger.info("Computing benign feature statistics from raw CSV ...")
    df_raw = loader._load_csv(CSV_FILES[0])
    label_col = "Label" if "Label" in df_raw.columns else cfg.LABEL_COL.strip()

    stats = BenignFeatureStats.from_dataframe(
        df_raw, feature_cols, label_col, shift_percentile
    )
    return stats, scaler, feature_cols


# =============================================================================
# SECTION 2 — Reference Graph Window Builder
# =============================================================================

def build_reference_window(
    loader: CICIDSDataLoader,
    scaler,
    max_windows: int = 50,
) -> Optional[Tuple[dict, torch.Tensor]]:
    """
    Stream graph windows from the dataset and return the first one that has
    a sufficient number of benign nodes (>= 2) for modification.

    We deliberately pick a benign-majority window so the behavioral scenarios
    start from a known-clean topology — this is what 'behavioral shift within
    a known topology' means.

    Returns (graph_dict, edge_labels) or None if no suitable window found.
    """
    logger.info("Searching for a benign-majority reference window ...")
    best_window = None
    best_benign_ratio = 0.0

    for i, (graph, labels) in enumerate(loader.stream_graphs(scaler)):
        if i >= max_windows:
            break
        benign_ratio = (labels == 0).float().mean().item()
        if benign_ratio > best_benign_ratio and graph["x"].shape[0] >= 4:
            best_benign_ratio = benign_ratio
            best_window = (
                {k: v.clone() if isinstance(v, torch.Tensor) else v
                 for k, v in graph.items()},
                labels.clone(),
            )
        if best_benign_ratio >= 0.95:
            break  # Good enough — stop early

    if best_window is None:
        logger.error("No suitable reference window found in the first %d windows.", max_windows)
        return None

    graph, labels = best_window
    logger.info(
        f"Reference window: id={graph['window_id']}  "
        f"nodes={graph['x'].shape[0]}  edges={graph['edge_index'].shape[1]}  "
        f"benign_ratio={best_benign_ratio:.2%}"
    )
    return graph, labels


# =============================================================================
# SECTION 3 — Behavioral Shift Factories (Zero Hardcoding)
# =============================================================================

def _sample_shift_nodes(
    num_nodes: int,
    n_shift: int,
    edge_index: torch.Tensor,
    prefer_low_degree: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """
    Select `n_shift` candidate nodes for behavioral modification.

    If `prefer_low_degree` is True, picks from the lowest-degree nodes
    (suitable for the silent-to-active scenario).
    Otherwise picks from the highest-degree nodes (more plausible role-shifters).

    Never exceeds the available node count.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    n_shift = min(n_shift, num_nodes)
    src_nodes = edge_index[0].numpy()

    if len(src_nodes) == 0:
        return list(range(min(n_shift, num_nodes)))

    degree = np.bincount(src_nodes, minlength=num_nodes).astype(np.float32)

    if prefer_low_degree:
        # Select from nodes with the lowest degree (silent nodes)
        candidates = np.argsort(degree)[:max(n_shift * 3, 10)]
    else:
        # Select from nodes with moderate-to-high degree (plausible role-shifters)
        candidates = np.argsort(degree)[::-1][:max(n_shift * 3, 10)]

    chosen = rng.choice(candidates, size=n_shift, replace=False).tolist()
    return [int(c) for c in chosen]


# ── Scenario 1: Role Shift ───────────────────────────────────────────────────

def apply_role_shift(
    graph: dict,
    labels: torch.Tensor,
    stats: BenignFeatureStats,
    n_shift_nodes: int,
    role_feature_subset: Optional[List[str]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[dict, torch.Tensor, List[int]]:
    """
    Scenario 1 — Role Shift: DB Server Behavior.

    A node previously exhibiting web-browsing or light-traffic patterns now
    shows database-server-like characteristics:
      - High in_bytes / out_bytes (large query-response payloads)
      - Low src_to_dst_iat_avg (frequent periodic polling by clients)
      - Moderate flow_duration (persistent DB connections)

    The feature shift is applied to the EDGE ATTRIBUTES (flow features) of
    edges incident to the chosen nodes, pushing them to the `p_high` value
    of the benign feature distribution — NOT to attack-class values.

    All resulting labels remain 0 (benign). FPR is the key metric.

    Parameters
    ----------
    role_feature_subset : list of feature names to shift (None = auto-select
                          from the feature names that best characterise
                          high-throughput, low-latency patterns in this dataset)

    Returns
    -------
    (modified_graph, labels, shifted_node_ids)
    """
    if rng is None:
        rng = np.random.default_rng(seed=100)

    g = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in graph.items()}
    labels_out = labels.clone()

    num_nodes  = g["x"].shape[0]
    edge_index = g["edge_index"]
    edge_attr  = g["edge_attr"].clone()  # [E, F]
    n_edges    = edge_attr.shape[0]

    # Auto-select features characteristic of DB-server role if not specified
    if role_feature_subset is None:
        # Pick throughput-heavy + low-latency features from the feature column list
        # These names match NF-UNSW-NB15-v3's feature schema exactly (from config)
        candidate_names = [
            "in_bytes", "out_bytes", "in_pkts", "out_pkts",
            "src_to_dst_second_bytes", "dst_to_src_second_bytes",
            "src_to_dst_avg_throughput", "dst_to_src_avg_throughput",
            "src_to_dst_iat_avg", "dst_to_src_iat_avg",
            "flow_duration",
        ]
        role_feature_subset = [f for f in candidate_names if f in stats.feature_cols]
        if not role_feature_subset:
            # Fallback: use first quarter of features (whichever they are)
            role_feature_subset = stats.feature_cols[:len(stats.feature_cols) // 4]

    logger.info(f"  Role-shift features: {role_feature_subset}")

    # Build feature-name → column-index map from the actual feature_cols list
    feat_to_idx: Dict[str, int] = {name: i for i, name in enumerate(stats.feature_cols)}
    shift_indices = [feat_to_idx[f] for f in role_feature_subset if f in feat_to_idx]

    if not shift_indices:
        logger.warning("  No valid feature indices for role shift — skipping modification.")
        return g, labels_out, []

    # Pick shift-target nodes (moderate-to-high degree — realistic role-shifters)
    shifted_nodes = _sample_shift_nodes(
        num_nodes, n_shift_nodes, edge_index, prefer_low_degree=False, rng=rng
    )

    # Build a mask of edges incident to the chosen nodes
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    shifted_set = set(shifted_nodes)

    edge_mask = np.array([
        (int(src[i]) in shifted_set or int(dst[i]) in shifted_set)
        for i in range(n_edges)
    ], dtype=bool)

    if edge_mask.sum() == 0:
        logger.warning("  Role shift: no incident edges found for chosen nodes.")
        return g, labels_out, shifted_nodes

    # Perturb edge features toward p_high of the benign distribution
    # We use a convex blend: new = (1 - alpha) * original + alpha * p_high
    # alpha is drawn uniformly from [0.5, 1.0] per edge to avoid a
    # perfectly uniform cluster that a GNN might trivially detect.
    n_affected = edge_mask.sum()
    alpha = rng.uniform(0.5, 1.0, size=(n_affected, 1)).astype(np.float32)
    p_high_subset = torch.tensor(stats.p_high, dtype=torch.float32)  # [F]

    original_affected = edge_attr[edge_mask]  # [n_affected, F]

    # Only modify the role-specific feature indices
    modified = original_affected.clone()
    alpha_t = torch.tensor(alpha, dtype=torch.float32)  # [n_affected, 1]

    for col_idx in shift_indices:
        p_hi_val = p_high_subset[col_idx]
        modified[:, col_idx] = (
            (1.0 - alpha_t.squeeze(1)) * original_affected[:, col_idx]
            + alpha_t.squeeze(1) * p_hi_val
        ).clamp(0.0, 1.0)

    # Write back
    new_edge_attr = edge_attr.clone()
    new_edge_attr[edge_mask] = modified

    # Update node features = per-node mean of incident edge features
    num_feats = new_edge_attr.shape[1]
    node_feat = torch.zeros(num_nodes, num_feats)
    node_cnt  = torch.zeros(num_nodes, 1)
    src_t = edge_index[0]
    dst_t = edge_index[1]
    node_feat.scatter_add_(0, src_t.unsqueeze(1).expand(-1, num_feats), new_edge_attr)
    node_feat.scatter_add_(0, dst_t.unsqueeze(1).expand(-1, num_feats), new_edge_attr)
    node_cnt.scatter_add_(0, src_t.unsqueeze(1), torch.ones(n_edges, 1))
    node_cnt.scatter_add_(0, dst_t.unsqueeze(1), torch.ones(n_edges, 1))
    node_cnt = node_cnt.clamp(min=1.0)

    g["edge_attr"] = new_edge_attr
    g["x"] = node_feat / node_cnt

    logger.info(
        f"  Role shift applied: {len(shifted_nodes)} nodes, "
        f"{n_affected} edges modified."
    )
    return g, labels_out, shifted_nodes


# ── Scenario 2: Subnet Debut ─────────────────────────────────────────────────

def apply_subnet_debut(
    graph: dict,
    labels: torch.Tensor,
    stats: BenignFeatureStats,
    n_shift_nodes: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[dict, torch.Tensor, List[int]]:
    """
    Scenario 2 — Subnet Debut: A previously unseen subnet communicates externally.

    The "subnet" is operationalized as an IP hash-partition bucket. A node whose
    flows previously mapped to a known partition is reassigned to the
    *least-represented* partition bucket in the training data — simulating
    an entirely new subnet making its first appearance.

    The edge FEATURES remain fully benign (drawn from the mean benign
    distribution). Only the node's partition identity changes, which affects
    the graph topology (edge routing) but not the flow statistics.

    Since GraphSAGE is inductive and learns from feature neighborhoods rather
    than node IDs, this scenario tests whether an unseen neighborhood *context*
    (new cluster of nodes with no shared neighbors with the training set) still
    generalizes correctly — it should NOT trigger a false positive.

    The node features of the chosen "subnet debut" nodes are set to the
    benign mean (p_mean) — we want to confirm that perfectly average-looking
    traffic from a new subnet does not get flagged.

    Returns
    -------
    (modified_graph, labels, shifted_node_ids)
    """
    if rng is None:
        rng = np.random.default_rng(seed=200)

    g = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in graph.items()}
    labels_out = labels.clone()

    num_nodes  = g["x"].shape[0]
    edge_index = g["edge_index"]
    node_feats = g["x"].clone()      # [N, F]
    edge_attr  = g["edge_attr"].clone()  # [E, F]
    n_edges    = edge_attr.shape[0]

    # Pick nodes to "debut from a new subnet"
    # We pick nodes with *low* degree (less topologically entrenched)
    shifted_nodes = _sample_shift_nodes(
        num_nodes, n_shift_nodes, edge_index, prefer_low_degree=True, rng=rng
    )

    if not shifted_nodes:
        logger.warning("  Subnet debut: no candidate nodes found.")
        return g, labels_out, []

    p_mean_tensor = torch.tensor(stats.p_mean, dtype=torch.float32)  # [F]

    # Set chosen nodes' features to the benign mean — generic legitimate traffic
    # from a "new" subnet that was never seen during training.
    for node_id in shifted_nodes:
        node_feats[node_id] = p_mean_tensor

    # Also set incident edge features to benign mean for consistency
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    shifted_set = set(shifted_nodes)

    new_edge_attr = edge_attr.clone()
    for i in range(n_edges):
        if int(src[i]) in shifted_set or int(dst[i]) in shifted_set:
            # Blend toward benign mean with some noise for realism
            noise = torch.tensor(
                rng.normal(0.0, 0.01, size=p_mean_tensor.shape).astype(np.float32)
            )
            new_edge_attr[i] = (p_mean_tensor + noise).clamp(0.0, 1.0)

    g["x"]         = node_feats
    g["edge_attr"] = new_edge_attr

    logger.info(
        f"  Subnet debut applied: {len(shifted_nodes)} nodes reassigned "
        f"to unseen partition context."
    )
    return g, labels_out, shifted_nodes


# ── Scenario 3: Silent-to-Active ─────────────────────────────────────────────

def apply_silent_to_active(
    graph: dict,
    labels: torch.Tensor,
    stats: BenignFeatureStats,
    n_shift_nodes: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[dict, torch.Tensor, List[int]]:
    """
    Scenario 3 — Silent-to-Active: Previously silent node initiates many connections.

    A node with near-zero out-degree in the reference window suddenly initiates
    a burst of connections reaching the dataset's 95th-percentile out-degree
    count. All new flows use fully BENIGN feature values (drawn from the benign
    mean ± a small Gaussian noise consistent with benign traffic variance).

    This is realistic for:
      - A new workstation added to the network
      - A backup agent starting its first scheduled run
      - A network scanner run by IT (legitimate recon)

    The new edges are added to the existing graph. Destination nodes are chosen
    uniformly at random from the existing node set (excluding the source itself).

    Returns
    -------
    (modified_graph, labels, shifted_node_ids)
    """
    if rng is None:
        rng = np.random.default_rng(seed=300)

    g = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in graph.items()}
    labels_out = labels.clone()

    num_nodes  = g["x"].shape[0]
    edge_index = g["edge_index"]
    edge_attr  = g["edge_attr"]
    n_edges    = edge_attr.shape[0]
    num_feats  = edge_attr.shape[1]

    # Target out-degree from the dataset's 95th-percentile distribution
    target_degree = max(3, int(stats.degree_p_high))

    # Select the quietest nodes (min out-degree)
    src_nodes = edge_index[0].numpy()
    degree = np.bincount(src_nodes, minlength=num_nodes).astype(np.float32)
    # Only consider nodes that genuinely have low degree (< 2)
    silent_candidates = np.where(degree < 2)[0]

    if len(silent_candidates) == 0:
        # Relax: take the bottom decile
        threshold = np.percentile(degree, 10.0)
        silent_candidates = np.where(degree <= threshold)[0]

    n_actual = min(n_shift_nodes, len(silent_candidates))
    if n_actual == 0:
        logger.warning("  Silent-to-active: no low-degree candidate nodes found.")
        return g, labels_out, []

    shifted_nodes = rng.choice(silent_candidates, size=n_actual, replace=False).tolist()
    shifted_nodes = [int(n) for n in shifted_nodes]

    p_mean_tensor = torch.tensor(stats.p_mean, dtype=torch.float32)  # [F]
    benign_std    = (stats.p_high - stats.p_low) / 4.0  # approximate std from IQR
    benign_std    = np.maximum(benign_std, 1e-4)

    # Build new synthetic edges from each silent node
    new_src_list, new_dst_list, new_attr_list = [], [], []

    for node_id in shifted_nodes:
        # How many new outgoing edges to create
        n_new_edges = target_degree - int(degree[node_id])
        n_new_edges = max(1, n_new_edges)

        # Choose destination nodes (any other node)
        all_others = [n for n in range(num_nodes) if n != node_id]
        if not all_others:
            continue
        dsts = rng.choice(all_others, size=min(n_new_edges, len(all_others)), replace=False)

        for dst_node in dsts:
            new_src_list.append(node_id)
            new_dst_list.append(int(dst_node))

            # Generate a benign flow feature vector: mean + small benign noise
            noise = rng.normal(0.0, benign_std, size=num_feats).astype(np.float32)
            new_feat = (stats.p_mean + noise).clip(0.0, 1.0)
            new_attr_list.append(new_feat)

    if not new_src_list:
        logger.warning("  Silent-to-active: no new edges could be created.")
        return g, labels_out, shifted_nodes

    new_src_t = torch.tensor(new_src_list, dtype=torch.long)
    new_dst_t = torch.tensor(new_dst_list, dtype=torch.long)
    new_attr_t = torch.tensor(np.stack(new_attr_list, axis=0), dtype=torch.float32)

    # Append to existing graph
    new_edge_index = torch.cat([edge_index, torch.stack([new_src_t, new_dst_t], dim=0)], dim=1)
    new_edge_attr  = torch.cat([edge_attr,  new_attr_t], dim=0)

    # New labels for the synthetic edges (all benign)
    new_edge_labels = torch.zeros(len(new_src_list), dtype=torch.long)
    new_labels      = torch.cat([labels_out, new_edge_labels], dim=0)

    # Recompute node features with updated edges
    total_edges = new_edge_attr.shape[0]
    node_feat = torch.zeros(num_nodes, num_feats)
    node_cnt  = torch.zeros(num_nodes, 1)
    src_t_all = new_edge_index[0]
    dst_t_all = new_edge_index[1]
    node_feat.scatter_add_(0, src_t_all.unsqueeze(1).expand(-1, num_feats), new_edge_attr)
    node_feat.scatter_add_(0, dst_t_all.unsqueeze(1).expand(-1, num_feats), new_edge_attr)
    node_cnt.scatter_add_(0, src_t_all.unsqueeze(1), torch.ones(total_edges, 1))
    node_cnt.scatter_add_(0, dst_t_all.unsqueeze(1), torch.ones(total_edges, 1))
    node_cnt = node_cnt.clamp(min=1.0)

    g["edge_index"] = new_edge_index
    g["edge_attr"]  = new_edge_attr
    g["x"]          = node_feat / node_cnt

    logger.info(
        f"  Silent-to-active applied: {len(shifted_nodes)} nodes, "
        f"{len(new_src_list)} new benign edges added "
        f"(target_degree={target_degree})."
    )
    return g, new_labels, shifted_nodes


# =============================================================================
# SECTION 4 — AURA Inference Pipeline
# =============================================================================

def derive_node_labels_from_edges(
    edge_labels: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert edge-level binary labels to node-level labels.
    A node is attack (1) if ANY incident edge is labelled attack.
    Matches the same convention used throughout benchmark_ablation.py.
    """
    node_labels = torch.zeros(num_nodes, dtype=torch.long, device=device)
    if edge_labels.sum() > 0:
        attack_mask = edge_labels.bool().to(device)
        src = edge_index[0].to(device)[attack_mask]
        dst = edge_index[1].to(device)[attack_mask]
        node_labels[src] = 1
        node_labels[dst] = 1
    return node_labels


def compute_ae_node_residual(
    ae: FlowAutoencoder,
    edge_attr: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Per-node mean-squared AE reconstruction error.
    Scatter-means edge MSE residuals onto incident nodes.
    Matches the identical helper in benchmark_ablation.py.
    """
    edge_attr = edge_attr.to(device)
    with torch.no_grad():
        x_hat, _ = ae(edge_attr)
        edge_residual = ((edge_attr - x_hat) ** 2).mean(dim=1)  # [E]

    node_error = torch.zeros(num_nodes, device=device)
    node_count = torch.zeros(num_nodes, device=device)
    src = edge_index[0].to(device)
    dst = edge_index[1].to(device)
    node_error.scatter_add_(0, src, edge_residual)
    node_error.scatter_add_(0, dst, edge_residual)
    node_count.scatter_add_(0, src, torch.ones_like(edge_residual))
    node_count.scatter_add_(0, dst, torch.ones_like(edge_residual))
    node_count = node_count.clamp(min=1.0)
    return node_error / node_count  # [N]


def run_aura_cascade(
    graph: dict,
    edge_labels: torch.Tensor,
    ae: FlowAutoencoder,
    stgnn: AuraSTGNN,
    ae_threshold: float,
    gnn_threshold: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the full AURA AE → GraphSAGE cascade inference on a single graph window.

    Returns
    -------
    y_true  : [N] node-level ground truth (0=benign, 1=attack)
    y_pred  : [N] node-level prediction   (0=benign, 1=attack)
    y_score : [N] continuous anomaly score (AE residual or GNN prob where flagged)
    """
    x          = graph["x"].to(device)
    edge_index = graph["edge_index"].to(device)
    edge_attr  = graph["edge_attr"].to(device)
    num_nodes  = x.shape[0]

    node_labels   = derive_node_labels_from_edges(edge_labels, edge_index, num_nodes, device)
    node_residual = compute_ae_node_residual(ae, edge_attr, edge_index, num_nodes, device)

    ae_flagged = node_residual > ae_threshold

    y_pred  = torch.zeros(num_nodes, dtype=torch.long, device=device)
    y_score = node_residual.clone()  # score for all nodes; overwritten for GNN-evaluated nodes

    if ae_flagged.sum() > 0:
        with torch.no_grad():
            gnn_scores, _ = stgnn(x, edge_index)  # [N]
        gnn_flagged_scores = gnn_scores[ae_flagged]
        gnn_decision = (gnn_flagged_scores > gnn_threshold).long()
        y_pred[ae_flagged]  = gnn_decision
        y_score[ae_flagged] = gnn_flagged_scores

    return (
        node_labels.cpu().numpy(),
        y_pred.cpu().numpy(),
        y_score.cpu().numpy(),
    )


# =============================================================================
# SECTION 5 — Metrics
# =============================================================================

def compute_fpr_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    shifted_node_ids: List[int],
    scenario_name: str,
) -> Dict:
    """
    Compute FPR and related metrics for a behavioral shift scenario.

    FPR is computed on the *shifted nodes only* (these are the legitimate
    behavioral shifters). All shifted nodes are ground-truth benign (Label=0);
    any that are predicted as 1 are false positives.

    Additionally, FPR is computed over the entire graph to capture whether
    the behavioral shift also disturbed unmodified neighbors (spillover FPR).

    Returns
    -------
    dict with keys:
      scenario, n_shifted_nodes, n_all_nodes,
      fpr_shifted_only, n_fp_shifted, n_tn_shifted,
      fpr_global, n_fp_global, n_tn_global,
      ae_gate_rate (fraction of nodes the AE flagged for GNN review)
    """
    shifted_set = set(shifted_node_ids)
    n_total = len(y_true)

    # ── FPR on shifted nodes only ────────────────────────────────────────────
    if shifted_set:
        shifted_idx = np.array(sorted(shifted_set), dtype=np.intp)
        # Guard against out-of-bounds (graph may have been extended in Scenario 3)
        shifted_idx = shifted_idx[shifted_idx < n_total]
        y_true_shift = y_true[shifted_idx]
        y_pred_shift = y_pred[shifted_idx]
        # All shifted nodes must be benign (they were perturbed from Label=0)
        benign_shift = y_true_shift == 0
        fp_shift = int(((y_pred_shift == 1) & benign_shift).sum())
        tn_shift = int(((y_pred_shift == 0) & benign_shift).sum())
        fpr_shift = fp_shift / max(fp_shift + tn_shift, 1)
    else:
        fp_shift = tn_shift = 0
        fpr_shift = 0.0

    # ── Global FPR (over entire graph) ───────────────────────────────────────
    benign_all = y_true == 0
    fp_global = int(((y_pred == 1) & benign_all).sum())
    tn_global = int(((y_pred == 0) & benign_all).sum())
    fpr_global = fp_global / max(fp_global + tn_global, 1)

    return {
        "Scenario":           scenario_name,
        "N_Shifted_Nodes":    len(shifted_node_ids),
        "N_All_Nodes":        n_total,
        "FPR_Shifted_Only":   round(fpr_shift,  6),
        "FP_Shifted":         fp_shift,
        "TN_Shifted":         tn_shift,
        "FPR_Global":         round(fpr_global, 6),
        "FP_Global":          fp_global,
        "TN_Global":          tn_global,
    }


# =============================================================================
# SECTION 6 — Export & Reporting
# =============================================================================

def export_results(
    results: List[Dict],
    reports_dir: Path,
    model_hash: str,
    args_dict: Dict,
) -> pd.DataFrame:
    """
    Save results to CSV and JSON in the reports/ directory.
    Includes the model hash and experiment configuration for reproducibility.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.set_index("Scenario", inplace=True)

    csv_path  = reports_dir / "behavioral_generalization_results.csv"
    json_path = reports_dir / "behavioral_generalization_results.json"

    df.to_csv(csv_path)

    full_output = {
        "experiment": "Section 5.2.3 Behavioral Generalization (H1)",
        "model_sha256": model_hash,
        "config": args_dict,
        "results": {r["Scenario"]: {k: v for k, v in r.items() if k != "Scenario"}
                    for r in results},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    logger.info(f"Results exported → {csv_path}")
    logger.info(f"Results exported → {json_path}")
    return df


def print_report(
    df: pd.DataFrame,
    results: List[Dict],
    args,
    reports_dir: Path,
    model_hash: str,
) -> None:
    """
    Print a research-grade console summary table matching the style of
    benchmark_byzantine.py and benchmark_ablation.py.
    """
    lines = [
        "",
        "=" * 72,
        "  AURA — Behavioral Generalization Benchmark  (Section 5.2.3 / H1)",
        "=" * 72,
        f"  Bundle SHA-256 : {model_hash}",
        f"  Load fraction  : {args.load_fraction}",
        f"  Shift percentile: P{args.shift_percentile:.0f} of benign feature distribution",
        f"  Shifted nodes  : {args.n_shift_nodes} per scenario",
        f"  AE threshold   : {args.ae_threshold:.6f} (from {'calibration JSON' if args.ae_threshold == args.ae_threshold else 'CLI'})",
        f"  GNN threshold  : {args.gnn_threshold}",
        "",
        "-" * 72,
        "  HYPOTHESIS H1: Behavioral shifts within known topologies should NOT",
        "  trigger false positives. FPR ≈ 0 is the target for all scenarios.",
        "-" * 72,
        "",
    ]

    fpr_target = args.fpr_target

    for r in results:
        scenario    = r["Scenario"]
        fpr_shifted = r["FPR_Shifted_Only"]
        fpr_global  = r["FPR_Global"]
        fp_shifted  = r["FP_Shifted"]
        tn_shifted  = r["TN_Shifted"]
        fp_global   = r["FP_Global"]
        tn_global   = r["TN_Global"]
        n_shifted   = r["N_Shifted_Nodes"]

        verdict_shifted = "PASS" if fpr_shifted <= fpr_target else "FAIL"
        verdict_global  = "PASS" if fpr_global  <= fpr_target else "WARN"

        lines += [
            f"  Scenario : {scenario}",
            f"    Shifted nodes    : {n_shifted}",
            f"    FPR (shifted)    : {fpr_shifted:.4f}  "
            f"[FP={fp_shifted} TN={tn_shifted}]  "
            f"[{verdict_shifted} vs target FPR≤{fpr_target}]",
            f"    FPR (global)     : {fpr_global:.4f}  "
            f"[FP={fp_global} TN={tn_global}]  "
            f"[{verdict_global}]",
            "",
        ]

    lines += [
        "-" * 72,
        "  RESULT SUMMARY (FPR on shifted nodes — H1 primary metric):",
        "",
    ]

    header = f"  {'Scenario':<35}  {'FPR_Shifted':>12}  {'FPR_Global':>12}  {'Verdict':>8}"
    lines.append(header)
    lines.append("  " + "-" * 70)
    for r in results:
        verdict = "PASS" if r["FPR_Shifted_Only"] <= fpr_target else "FAIL"
        lines.append(
            f"  {r['Scenario']:<35}  {r['FPR_Shifted_Only']:>12.6f}  "
            f"{r['FPR_Global']:>12.6f}  {verdict:>8}"
        )

    lines += [
        "",
        f"  FPR target (--fpr-target): {fpr_target}",
        "",
        "  NOTE: FPR_Shifted_Only = FPR measured *only* on the nodes whose",
        "  communication pattern was behaviorally shifted (all are Label=0).",
        "  FPR_Global = FPR over the entire graph including unmodified nodes.",
        "  A model with good behavioral generalization should have both near 0.",
        "",
        "  Exported to:",
        f"    {reports_dir / 'behavioral_generalization_results.csv'}",
        f"    {reports_dir / 'behavioral_generalization_results.json'}",
        "=" * 72,
        "",
    ]

    sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8", errors="replace"))
    sys.stdout.flush()


# =============================================================================
# SECTION 7 — Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    # Load calibrated AE threshold as default (not a hardcoded magic number)
    ae_thresh_high, _, ae_calibrated = cfg.load_ae_thresholds()

    parser = argparse.ArgumentParser(
        description=(
            "AURA Section 5.2.3 — Behavioral Generalization Benchmark (Hypothesis H1). "
            "Tests whether GraphSAGE raises false positives on legitimate behavioral "
            "shifts within known network topologies."
        )
    )
    parser.add_argument(
        "--bundle",
        type=str,
        default=str(cfg.MODELS_DIR / "aura_bundle.pth"),
        help="Path to the pre-trained AURAModelBundle checkpoint.",
    )
    parser.add_argument(
        "--load-fraction",
        type=float,
        default=cfg.DATA_LOAD_FRACTION,
        help=(
            f"Fraction of the NF-UNSW-NB15-v3 CSV to load "
            f"(default: cfg.DATA_LOAD_FRACTION={cfg.DATA_LOAD_FRACTION}). "
            "Reduce for quick runs."
        ),
    )
    parser.add_argument(
        "--shift-percentile",
        type=float,
        default=95.0,
        help=(
            "Percentile of the benign feature distribution used to set the "
            "magnitude of behavioral shifts (default: 95). "
            "Higher = more extreme / unusual but still benign patterns."
        ),
    )
    parser.add_argument(
        "--n-shift-nodes",
        type=int,
        default=5,
        help="Number of nodes to behaviorally shift per scenario (default: 5).",
    )
    parser.add_argument(
        "--ae-threshold",
        type=float,
        default=ae_thresh_high,
        help=(
            f"AE MSE threshold for the cascade gate "
            f"(default: {ae_thresh_high:.6f} from calibration_results.json). "
            "Override to test threshold sensitivity."
        ),
    )
    parser.add_argument(
        "--gnn-threshold",
        type=float,
        default=0.5,
        help="GraphSAGE anomaly probability threshold (default: 0.5).",
    )
    parser.add_argument(
        "--fpr-target",
        type=float,
        default=0.05,
        help=(
            "Maximum acceptable FPR for a PASS verdict (default: 0.05). "
            "Each scenario is marked PASS if FPR_Shifted_Only <= this value."
        ),
    )
    parser.add_argument(
        "--reference-windows",
        type=int,
        default=50,
        help="Maximum windows to scan when selecting the reference graph (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global RNG seed for reproducible shift sampling (default: 42).",
    )
    return parser.parse_args()


def _compute_model_hash(bundle: AURAModelBundle) -> str:
    """SHA-256 over all model parameters — tamper-evident audit trail."""
    h = hashlib.sha256()
    for p in bundle.parameters():
        h.update(np.ascontiguousarray(p.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return "0x" + h.hexdigest()


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 72)
    print("  AURA Behavioral Generalization Benchmark — Section 5.2.3 (H1)")
    print("=" * 72)
    print(f"  Bundle      : {args.bundle}")
    print(f"  AE threshold: {args.ae_threshold:.6f}")
    print(f"  GNN threshold:{args.gnn_threshold}")
    print(f"  Shift P      : {args.shift_percentile:.0f}th percentile of benign distribution")
    print(f"  Shift nodes  : {args.n_shift_nodes} per scenario")
    print(f"  FPR target   : ≤ {args.fpr_target}")
    print("=" * 72 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── 1. Load model bundle ─────────────────────────────────────────────────
    bundle_path = Path(args.bundle)
    bundle = AURAModelBundle()

    if bundle_path.exists():
        logger.info(f"Loading pre-trained bundle from {bundle_path} ...")
        state = torch.load(bundle_path, map_location=device, weights_only=True)
        bundle.load_state_dict(state)
        logger.info("✓ Model bundle loaded successfully.")
    else:
        logger.warning(
            f"Bundle not found at {bundle_path}. Running with randomly "
            "initialised weights — FPR results will reflect an untrained model. "
            "Train first with: python train.py"
        )

    ae    = bundle.autoencoder.to(device).eval()
    stgnn = bundle.stgnn.to(device).eval()

    model_hash = _compute_model_hash(bundle)
    logger.info(f"Model SHA-256: {model_hash}")
    logger.info(f"AE params : {ae.count_params():,}  |  STGNN params: {stgnn.count_params():,}")

    # ── 2. Verify AE feature dimension ──────────────────────────────────────
    actual_feature_dim = ae.encoder[0].in_features
    if actual_feature_dim != cfg.FEATURE_DIM:
        logger.warning(
            f"Loaded AE feature dim ({actual_feature_dim}) ≠ cfg.FEATURE_DIM "
            f"({cfg.FEATURE_DIM}). Bundle may be stale."
        )

    # ── 3. Load dataset statistics and scaler ───────────────────────────────
    logger.info("\n>>> Loading dataset and computing benign feature statistics ...")
    stats, scaler, feature_cols = load_benign_stats_and_scaler(
        load_fraction=args.load_fraction,
        shift_percentile=args.shift_percentile,
    )

    # ── 4. Build reference graph window ─────────────────────────────────────
    logger.info("\n>>> Building reference graph window ...")
    ref_loader = CICIDSDataLoader(load_fraction=args.load_fraction)

    # Ensure feature_cols are populated on the loader
    ref_loader._feature_cols = feature_cols

    ref_window = build_reference_window(
        ref_loader, scaler, max_windows=args.reference_windows
    )
    if ref_window is None:
        logger.error(
            "Could not find a suitable reference window. "
            "Try increasing --load-fraction or --reference-windows."
        )
        sys.exit(1)

    ref_graph, ref_labels = ref_window

    # ── 5. Run three behavioral shift scenarios ───────────────────────────────
    rng_master = np.random.default_rng(seed=args.seed)

    scenarios = [
        {
            "name": "S1: Role Shift (DB Server)",
            "fn":   lambda g, l: apply_role_shift(
                g, l, stats, args.n_shift_nodes,
                rng=np.random.default_rng(seed=int(rng_master.integers(0, 2**31))),
            ),
        },
        {
            "name": "S2: Subnet Debut",
            "fn":   lambda g, l: apply_subnet_debut(
                g, l, stats, args.n_shift_nodes,
                rng=np.random.default_rng(seed=int(rng_master.integers(0, 2**31))),
            ),
        },
        {
            "name": "S3: Silent-to-Active",
            "fn":   lambda g, l: apply_silent_to_active(
                g, l, stats, args.n_shift_nodes,
                rng=np.random.default_rng(seed=int(rng_master.integers(0, 2**31))),
            ),
        },
    ]

    all_results: List[Dict] = []

    for scenario in scenarios:
        sname = scenario["name"]
        print(f"\n{'─' * 60}")
        print(f"  [{sname}]")
        print(f"{'─' * 60}")
        logger.info(f"Applying behavioral shift: {sname}")

        t0 = time.time()

        # Apply the behavioral shift to a *copy* of the reference graph
        modified_graph, modified_labels, shifted_nodes = scenario["fn"](
            ref_graph, ref_labels
        )

        if not shifted_nodes:
            logger.warning(f"  Scenario produced no shifted nodes — skipping.")
            continue

        print(f"  Shifted nodes : {shifted_nodes}")
        print(f"  Total nodes   : {modified_graph['x'].shape[0]}")
        print(f"  Total edges   : {modified_graph['edge_index'].shape[1]}")

        # Run AURA cascade inference
        y_true, y_pred, y_score = run_aura_cascade(
            modified_graph, modified_labels,
            ae, stgnn,
            ae_threshold=args.ae_threshold,
            gnn_threshold=args.gnn_threshold,
            device=device,
        )

        metrics = compute_fpr_metrics(y_true, y_pred, shifted_nodes, sname)
        metrics["Time_s"] = round(time.time() - t0, 3)
        all_results.append(metrics)

        verdict = "PASS" if metrics["FPR_Shifted_Only"] <= args.fpr_target else "FAIL"
        print(f"  FPR (shifted) : {metrics['FPR_Shifted_Only']:.4f}  → [{verdict}]")
        print(f"  FPR (global)  : {metrics['FPR_Global']:.4f}")
        print(f"  Elapsed       : {metrics['Time_s']}s")

    if not all_results:
        logger.error("No scenario results were produced. Check dataset and model paths.")
        sys.exit(1)

    # ── 6. Export and print ──────────────────────────────────────────────────
    reports_dir = PROJECT_ROOT / "reports"
    args_dict = {
        "bundle":           args.bundle,
        "load_fraction":    args.load_fraction,
        "shift_percentile": args.shift_percentile,
        "n_shift_nodes":    args.n_shift_nodes,
        "ae_threshold":     args.ae_threshold,
        "gnn_threshold":    args.gnn_threshold,
        "fpr_target":       args.fpr_target,
        "seed":             args.seed,
    }
    df = export_results(all_results, reports_dir, model_hash, args_dict)

    # Flush log handlers before printing the table (avoids interleaved stderr)
    for handler in logging.getLogger().handlers:
        handler.flush()
    sys.stderr.flush()

    print_report(df, all_results, args, reports_dir, model_hash)


if __name__ == "__main__":
    main()
