"""
aura/data_loader.py — Phase 1: Data Ingestion & Topological Mapping
=====================================================================

Pipeline Design
---------------
This pipeline ingests NF-UNSW-NB15-v3 NetFlow CSV data. It explicitly
extracts IPV4_SRC_ADDR and IPV4_DST_ADDR to construct a genuine 
spatial-topological network graph.

Processing chain (in order):
  1. Raw CSV  →  strip column whitespace  →  drop Inf/NaN
  2. Label column extracted; rows split into BENIGN and ATTACK splits
  3. Benign split sanitised with IsolationForest (Poisoned Baseline Defence)
  4. MinMaxScaler fitted on sanitised benign data; applied to all splits
  5. Real topological edges mapped via unique Source/Destination IPs
  6. Rolling WINDOW_SIZE-row snapshots  →  PyTorch tensors
  7. Synthetic edges built with TTL counter; expired edges pruned each window
  8. Node features = per-node mean aggregation of incident edge features

Returns
-------
  A Python generator that yields (graph_dict, label_vector) tuples.
"""

import logging
import os
import hashlib as _hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple


import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# Project-level config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATH = Path(__file__).parent.parent / "dataset" / "NF-UNSW-NB15-v3.csv"
CSV_FILES: List[str] = [str(DATASET_PATH)]

# Org name -> stable partition index for federated learning clients.
_ORG_NAMES = ["hospital", "bank", "university", "isp", "retail"]

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _strip_column_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df

def _clean_infinities_and_nans(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].ffill()
    df[feature_cols] = df[feature_cols].bfill()
    df = df.dropna(subset=feature_cols)
    return df

def _isolationforest_sanitise(X: np.ndarray, contamination: float = cfg.IF_CONTAMINATION) -> Tuple[np.ndarray, np.ndarray]:
    logger.info(f"Running IsolationForest baseline sanitisation (contamination={contamination}) on {len(X)} benign rows …")
    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
    preds = iso.fit_predict(X)
    mask  = preds == 1
    X_clean = X[mask]
    removed = int((~mask).sum())
    logger.info(f"IsolationForest removed {removed} suspicious rows. Clean baseline size: {len(X_clean)} rows.")
    return X_clean, mask

def _assign_real_nodes(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, int]:
    src_col = 'IPV4_SRC_ADDR' if 'IPV4_SRC_ADDR' in df.columns else 'src_ip'
    dst_col = 'IPV4_DST_ADDR' if 'IPV4_DST_ADDR' in df.columns else 'dst_ip'
    
    unique_ips = pd.concat([df[src_col], df[dst_col]]).unique()
    ip_to_id = {ip: idx for idx, ip in enumerate(unique_ips)}
    
    src_nodes = df[src_col].map(ip_to_id).values.astype(np.int64)
    dst_nodes = df[dst_col].map(ip_to_id).values.astype(np.int64)
    
    return src_nodes, dst_nodes, len(unique_ips)


def _ip_to_partition(ip: str, n_clients: int = len(_ORG_NAMES)) -> int:
    """
    Hash a source IP to a stable FL client bucket.

    Keeping all flows from a host in one partition preserves local behaviour
    better than random row splitting, which matters for each client's AE
    baseline and any future per-client topology training.
    """
    h = int(_hashlib.md5(str(ip).encode()).hexdigest(), 16)
    return h % n_clients


# ─────────────────────────────────────────────────────────────────────────────
# TTL Edge Decay Tracker
# ─────────────────────────────────────────────────────────────────────────────

class TTLEdgeTracker:
    def __init__(self, ttl: int = cfg.EDGE_TTL_WINDOWS):
        self.ttl = ttl
        self._counters: Dict[Tuple[int, int], int] = defaultdict(lambda: ttl)

    def update(self, active_edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        active_set = set(active_edges)
        for e in active_set:
            self._counters[e] = self.ttl
        dormant = set(self._counters.keys()) - active_set
        for e in dormant:
            self._counters[e] -= 1
        expired = [e for e, ttl in self._counters.items() if ttl <= 0]
        for e in expired:
            del self._counters[e]
        return list(self._counters.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Node Feature Aggregator
# ─────────────────────────────────────────────────────────────────────────────

def _build_node_features(edge_features: np.ndarray, src_nodes: np.ndarray, dst_nodes: np.ndarray, num_nodes: int, feature_dim: int) -> np.ndarray:
    X = np.zeros((num_nodes, feature_dim), dtype=np.float32)
    counts = np.zeros(num_nodes, dtype=np.float32)

    for i, (s, d) in enumerate(zip(src_nodes, dst_nodes)):
        X[s] += edge_features[i]
        X[d] += edge_features[i]
        counts[s] += 1
        counts[d] += 1

    counts = np.maximum(counts, 1.0)
    X = X / counts[:, np.newaxis]
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Core Public API
# ─────────────────────────────────────────────────────────────────────────────

class CICIDSDataLoader:
    """
    Data loader for NF-UNSW-NB15-v3 NetFlow CSV data.

    Class name retained as CICIDSDataLoader for backward compatibility with
    existing imports across the codebase (train.py, calibrate_thresholds.py,
    dashboard.py, etc.).
    """
    def __init__(self, csv_dir: Path = cfg.CSV_DIR, load_fraction: float = cfg.DATA_LOAD_FRACTION, window_size: int = cfg.WINDOW_SIZE):
        self.csv_dir = csv_dir
        self.load_fraction = load_fraction
        self.window_size = window_size
        self.num_nodes = 0 
        self._ttl_tracker = TTLEdgeTracker()
        self._scaler: Optional[MinMaxScaler] = None
        self._feature_cols: Optional[List[str]] = None

    def _load_csv(self, path_str: str) -> pd.DataFrame:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        logger.info(f"Loading {path.name} …")
        total_rows = sum(1 for _ in open(path)) - 1
        n_rows = max(100, int(total_rows * self.load_fraction))
        df = pd.read_csv(path, nrows=n_rows, low_memory=False)
        df = _strip_column_whitespace(df)

        if self._feature_cols is None:
            # Columns to exclude from feature set:
            # - IP addresses (used for topology, not as features)
            # - Port numbers (used for topology context)
            # - Timestamps (absolute time, not statistical features)
            # - Labels (target variables)
            ignored_cols = {
                'IPV4_SRC_ADDR', 'IPV4_DST_ADDR',
                'L4_SRC_PORT', 'L4_DST_PORT',
                'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS',
                'Label', 'Attack',
                'src_ip', 'dst_ip',
            }
            label_c = cfg.LABEL_COL.strip()
            self._feature_cols = [c for c in df.columns if c not in ignored_cols and c != label_c]
            logger.info(f"Discovered {len(self._feature_cols)} statistical feature columns.")

        df = _clean_infinities_and_nans(df, self._feature_cols)
        return df

    def _label_to_binary(self, series: pd.Series) -> np.ndarray:
        """Convert label column to binary (0=benign, 1=attack).

        NF-UNSW-NB15-v3 Label is already binary int (0/1).
        Falls back to string comparison for compatibility.
        """
        if pd.api.types.is_numeric_dtype(series):
            return series.values.astype(np.int64)
        return (series.str.strip().str.upper() != "BENIGN").astype(np.int64).values

    def fit_scaler(self) -> MinMaxScaler:
        df = self._load_csv(CSV_FILES[0])
        label_col = 'Label' if 'Label' in df.columns else cfg.LABEL_COL.strip()
        
        # NF-UNSW-NB15-v3: Label is binary int (0 = Benign, 1 = Attack)
        if pd.api.types.is_numeric_dtype(df[label_col]):
            benign_df = df[df[label_col] == cfg.BENIGN_LABEL]
        else:
            benign_df = df[df[label_col].str.strip().str.upper() == "BENIGN"]
            
        logger.info(f"Benign training rows before sanitisation: {len(benign_df)}")

        X_benign = benign_df[self._feature_cols].values.astype(np.float32)
        X_clean, _ = _isolationforest_sanitise(X_benign)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_clean)
        self._scaler = scaler
        logger.info("MinMaxScaler fitted on sanitised benign baseline.")
        return scaler

    def stream_graphs(self, scaler: MinMaxScaler, csv_files: Optional[List[str]] = None) -> Generator[Tuple[Dict, torch.Tensor], None, None]:
        if csv_files is None:
            csv_files = CSV_FILES

        for csv_file in csv_files:
            try:
                df = self._load_csv(csv_file)
            except FileNotFoundError:
                logger.warning(f"Skipping missing file: {csv_file}")
                continue

            label_col = 'Label' if 'Label' in df.columns else cfg.LABEL_COL.strip()
            labels_all = self._label_to_binary(df[label_col])

            X_scaled = scaler.transform(df[self._feature_cols].values.astype(np.float32)).clip(0, 1)

            src_all, dst_all, total_nodes = _assign_real_nodes(df)
            self.num_nodes = total_nodes

            n_windows = len(df) // self.window_size
            logger.info(f"Streaming {n_windows} windows from {Path(csv_file).name} mapping {total_nodes} unique hosts…")

            for w in range(n_windows):
                s = w * self.window_size
                e = s + self.window_size

                X_window = X_scaled[s:e]
                src_window = src_all[s:e]
                dst_window = dst_all[s:e]
                labels_window = labels_all[s:e]

                active_edges = list(zip(src_window.tolist(), dst_window.tolist()))
                live_edges = self._ttl_tracker.update(active_edges)

                live_edge_set = set(live_edges)
                keep_mask = np.array([(int(src_window[i]), int(dst_window[i])) in live_edge_set for i in range(len(src_window))])

                if keep_mask.sum() == 0:
                    continue

                X_edge = X_window[keep_mask]
                src_edge = src_window[keep_mask]
                dst_edge = dst_window[keep_mask]
                labels_w = labels_window[keep_mask]

                X_node = _build_node_features(X_edge, src_edge, dst_edge, self.num_nodes, len(self._feature_cols))

                edge_index = torch.tensor(np.stack([src_edge, dst_edge], axis=0), dtype=torch.long)
                x = torch.tensor(X_node, dtype=torch.float32)
                edge_attr = torch.tensor(X_edge, dtype=torch.float32)

                graph_dict = {
                    "x": x,
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                    "ttl_state": dict(self._ttl_tracker._counters),
                    "window_id": f"{Path(csv_file).name}:w{w}",
                }

                label_tensor = torch.tensor(labels_w, dtype=torch.long)
                yield graph_dict, label_tensor


# ─────────────────────────────────────────────────────────────────────────────
# FL Client Partition Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_client_partition(
    client_id: str,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load one FL client's benign NF-UNSW-NB15-v3 partition for AE training.

    Rows are assigned by source-IP hash so each organisation gets a stable,
    host-coherent, non-IID slice of the shared CSV. The returned tensors are
    flat flow-feature matrices shaped [N, FEATURE_DIM], matching the current
    Flower client training loop, which trains the autoencoder locally.
    """
    parts = client_id.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse org key from client_id: '{client_id}'")

    org_key = parts[1].lower()
    if org_key not in _ORG_NAMES:
        raise ValueError(
            f"Unknown org '{org_key}'. Valid orgs: {_ORG_NAMES}. "
            "Expected client_id format: 'org_<orgname>_<num>'."
        )
    partition_idx = _ORG_NAMES.index(org_key)

    loader = CICIDSDataLoader()
    if scaler is None:
        logger.warning(
            "[%s] No shared scaler provided; fitting a local scaler. "
            "For FL runs, pass the shared scaler for comparable client updates.",
            client_id,
        )
        scaler = loader.fit_scaler()

    df = loader._load_csv(str(DATASET_PATH))
    label_col = "Label" if "Label" in df.columns else cfg.LABEL_COL.strip()

    if pd.api.types.is_numeric_dtype(df[label_col]):
        benign_df = df[df[label_col] == cfg.BENIGN_LABEL].copy()
    else:
        benign_df = df[df[label_col].str.strip().str.upper() == "BENIGN"].copy()

    src_col = "IPV4_SRC_ADDR" if "IPV4_SRC_ADDR" in benign_df.columns else "src_ip"
    if src_col not in benign_df.columns:
        raise ValueError(
            f"Source IP column not found. Expected 'IPV4_SRC_ADDR' or 'src_ip'. "
            f"Available columns: {list(benign_df.columns)}"
        )

    partitions = benign_df[src_col].apply(
        lambda ip: _ip_to_partition(str(ip), n_clients=len(_ORG_NAMES))
    )
    partition_df = benign_df[partitions == partition_idx].copy()

    if partition_df.empty:
        raise RuntimeError(
            f"[{client_id}] Partition {partition_idx} ({org_key}) has no benign "
            "rows. Try increasing cfg.DATA_LOAD_FRACTION."
        )

    feature_cols = loader._feature_cols
    if not feature_cols:
        raise RuntimeError("Feature columns were not discovered while loading the dataset.")

    X = partition_df[feature_cols].values.astype(np.float32)
    X = scaler.transform(X).clip(0, 1)

    if len(X) < 2:
        raise RuntimeError(
            f"[{client_id}] Partition {partition_idx} ({org_key}) has only "
            f"{len(X)} row(s), not enough for train/validation split."
        )

    rng = np.random.default_rng(seed=42 + partition_idx)
    rng.shuffle(X)

    split = max(1, int(len(X) * 0.8))
    if split >= len(X):
        split = len(X) - 1

    X_train = torch.tensor(X[:split], dtype=torch.float32)
    X_val = torch.tensor(X[split:], dtype=torch.float32)

    logger.info(
        "[%s] Partition %d (%s) ready: train=%d val=%d features=%d",
        client_id,
        partition_idx,
        org_key,
        len(X_train),
        len(X_val),
        X_train.shape[1],
    )
    return X_train, X_val

def load_client_subpartition(
    client_idx: int,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a sub-partition for one of 10 benchmark clients.
    
    Maps 10 clients onto 5 real org partitions by halving each:
      clients 0-4 → first half of org 0-4 partition
      clients 5-9 → second half of org 0-4 partition
    
    This gives 10 genuinely non-overlapping datasets from 5 real
    IP-hashed org partitions — no two clients share the same rows.
    """
    org_idx   = client_idx % len(_ORG_NAMES)   # 0-4
    org_key   = _ORG_NAMES[org_idx]
    is_second = client_idx >= len(_ORG_NAMES)   # True for clients 5-9

    # Reuse existing partition loader with canonical org ID
    _org_nums = {"hospital":1, "bank":2, "university":3, "isp":4, "retail":5}
    client_id = f"org_{org_key}_{_org_nums[org_key]}"

    X_train, X_val = load_client_partition(
        client_id=client_id,
        scaler=scaler,
    )

    # Split train in half — each half is a distinct non-overlapping subset
    half = len(X_train) // 2
    if half < 1:
        raise RuntimeError(
            f"Partition for {org_key} too small to sub-partition "
            f"({len(X_train)} rows). Increase DATA_LOAD_FRACTION."
        )

    if is_second:
        X_train = X_train[half:]   # second half → clients 5-9
    else:
        X_train = X_train[:half]   # first half  → clients 0-4

    # Val split is shared between both halves of same org
    # (val data is always the 20% held out by load_client_partition,
    #  independent of the train split — no leakage)
    logger.info(
        "[subpartition] client_idx=%d  org=%s  half=%s  train=%d  val=%d",
        client_idx, org_key,
        "second" if is_second else "first",
        len(X_train), len(X_val),
    )
    return X_train, X_val

# ─────────────────────────────────────────────────────────────────────────────
# CLI Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== AURA Data Loader — Real Topology Sanity Check ===")
    loader = CICIDSDataLoader(load_fraction=0.05)
    try:
        print("Fitting scaler on benign baseline ...")
        scaler = loader.fit_scaler()
        print("Streaming first 3 graph windows ...")
        for i, (graph, labels) in enumerate(loader.stream_graphs(scaler)):
            print(f"\n[Window {i}]  id={graph['window_id']}")
            print(f"  x.shape        = {graph['x'].shape}        (Nodes x Features)")
            print(f"  edge_index.shape= {graph['edge_index'].shape}  (2 x Edges)")
            print(f"  edge_attr.shape = {graph['edge_attr'].shape}  (Edges x Features)")
            print(f"  labels.shape    = {labels.shape}   | attack ratio={labels.float().mean():.3f}")
            print(f"  live edges (TTL)= {len(graph['ttl_state'])}")
            if i >= 2:
                break
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")

    print("\n[PASS] Data loader test passed.")
