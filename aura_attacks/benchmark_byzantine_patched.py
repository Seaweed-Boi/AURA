#!/usr/bin/env python3
"""
scripts/benchmark_byzantine.py
==============================
Implements the 5.1.5 FLTrust Byzantine Benchmark (Hypothesis mapping: H2).

Runs a comparative evaluation of:
  - FedAvg (no defense)
  - Krum (distance-based exclusion)
  - FLTrust (cosine similarity against server root dataset)

Under various Byzantine attack ratios (10%, 20%, 30%, 40%).
Also includes the rare-client contribution preservation experiment.

NOTE: Runs in-process (no Ray/Flower simulation daemon required).
"""

import sys
import logging
import hashlib
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
import torch
import torch.nn as nn
import numpy as np

from aura.fl_server import fltrust_aggregate, krum_select, krum_aggregate, _build_root_dataset
from aura.fl_client import AURAFlowerClient
from aura.data_loader import CICIDSDataLoader, load_client_partition
from aura.attack_injector import _benign_profile
from aura.models import AURAModelBundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("byz_bench")

# Force stdout to UTF-8 so Unicode characters render correctly on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# -----------------------------------------------------------------------------
# Initialize global scaler once for the entire benchmark run
# -----------------------------------------------------------------------------
_loader = CICIDSDataLoader()
try:
    _shared_scaler = _loader.fit_scaler()
    logger.info("Global dataset scaler initialized successfully.")
except Exception as e:
    logger.warning(f"Could not fit scaler on CSV dataset: {e}. Falling back to synthetic profiles.")
    _shared_scaler = None


def generate_client_data(
    client_idx: int,
    is_byzantine: bool,
    is_rare: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate local training data for a client.

    - Benign:    real NF-UNSW-NB15-v3 partition (or synthetic benign fallback)
    - Rare:      benign data with a +0.15 global shift (honest but unusual traffic)
    - Byzantine: 80% of rows poisoned with DDoS statistical profile from config
    """
    n_samples   = 200
    feature_dim = cfg.FEATURE_DIM
    client_id_str = f"org_test_{client_idx}"

    train_data, val_data = None, None
    if _shared_scaler is not None:
        try:
            train_data, val_data = load_client_partition(
                client_id=client_id_str, scaler=_shared_scaler
            )
            if len(train_data) > n_samples:
                train_data = train_data[:n_samples]
            if len(val_data) > max(1, n_samples // 5):
                val_data = val_data[:n_samples // 5]
        except Exception:
            pass

    # Fallback: realistic synthetic benign profile
    if train_data is None:
        _train_np = _benign_profile(n_samples, feature_dim)
        _val_np   = _benign_profile(max(1, n_samples // 5), feature_dim)
        train_data = torch.tensor(_train_np, dtype=torch.float32)
        val_data   = torch.tensor(_val_np,   dtype=torch.float32)

    if is_rare:
        # Legitimate but distribution-shifted client (e.g., hospital with rare traffic).
        # +0.15 shift simulates higher baseline volume -- still benign direction.
        train_data = train_data + 0.15

    if is_byzantine:
        # Poison 80% of local batch using real DDoS feature ranges from NF-UNSW-NB15-v3
        ddos_profile = cfg.ATTACK_CORRUPTION_PROFILES.get("ddos", {})
        feat_map     = cfg.FEATURE_INDEX_MAP
        n_attack     = int(len(train_data) * 0.8)
        attack_rows  = train_data[:n_attack].clone()
        for feat_name, (lo, hi) in ddos_profile.items():
            if feat_name in feat_map:
                col_idx = feat_map[feat_name]
                attack_rows[:, col_idx] = torch.rand(n_attack) * (hi - lo) + lo
        train_data[:n_attack] = attack_rows

    return train_data, val_data


def _run_local_training(
    client:        AURAFlowerClient,
    global_arrays: List[np.ndarray],
) -> Tuple[List[np.ndarray], float]:
    """
    One FL round on a single client (in-process, no gRPC needed):
      1. Load global weights into client's local model.
      2. Run FL_LOCAL_EPOCHS of unsupervised AE training.
      3. Return updated weight arrays + training loss.
    """
    with torch.no_grad():
        for p, arr in zip(client.model.parameters(), global_arrays):
            p.copy_(torch.tensor(arr, dtype=torch.float32))

    num_examples, train_loss = client._local_train()
    updated_arrays = [p.detach().cpu().numpy() for p in client.model.parameters()]
    return updated_arrays, train_loss


def run_experiment(
    strategy_name:   str,
    num_clients:     int,
    byzantine_ratio: float,
    rare_client:     bool = False,
    seed:            int = None,
):
    """
    Run one complete federated learning simulation.

    Parameters
    ----------
    strategy_name   : "FedAvg", "Krum", or "FLTrust"
    num_clients     : Total clients in the federation (10)
    byzantine_ratio : Fraction of clients that are adversarial (0.1 to 0.4)
    rare_client     : If True, last client gets shifted-but-benign distribution
    seed            : If set, fixes torch/numpy RNG so runs are comparable
                       across strategies/ratios (needed for divergence metric).

    Returns
    -------
    dict with keys: strategy, byzantine_ratio, num_byzantine, roles,
    flagged_indices (clients this strategy excluded/flagged, empty for
    FedAvg since it has no defense), tp/fp/fn/tn and balanced_accuracy for
    Byzantine-client detection, final_arrays (the resulting global model
    weights, for computing divergence against a clean baseline), model_hash.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    logger.info("\n" + "=" * 60)
    logger.info(
        f"Running {strategy_name} | Byzantine Ratio: {byzantine_ratio*100:.0f}% "
        f"| Rare Client: {rare_client}"
    )
    logger.info("=" * 60)

    num_byzantine = int(num_clients * byzantine_ratio)
    roles         = ["benign"] * num_clients
    for i in range(num_byzantine):
        roles[i] = "byzantine"
    if rare_client and "benign" in roles:
        roles[-1] = "rare"

    logger.info(f"Client Roles: {roles}")

    # Build clients
    clients: List[AURAFlowerClient] = []
    for idx in range(num_clients):
        role = roles[idx]
        train_data, val_data = generate_client_data(
            idx,
            is_byzantine=(role == "byzantine"),
            is_rare=(role == "rare"),
        )
        clients.append(AURAFlowerClient(f"client_{idx}", train_data, val_data))

    # Shared global model
    global_model  = AURAModelBundle()
    global_arrays = [p.detach().cpu().numpy() for p in global_model.parameters()]

    # FLTrust server root dataset (benign reference -- built once per experiment)
    root_data = _build_root_dataset()

    # Federated rounds
    num_rounds = 2
    for rnd in range(1, num_rounds + 1):
        print("\n" + "-" * 60)
        print(f"  [{strategy_name} | {byzantine_ratio*100:.0f}% Byzantine] Round {rnd}/{num_rounds}")
        print("-" * 60)

        # Each client trains locally
        client_updates: List[List[np.ndarray]] = []
        for idx, client in enumerate(clients):
            updated_arrays, train_loss = _run_local_training(client, global_arrays)
            role_tag = roles[idx]
            print(f"  Client {idx:2d} [{role_tag:10s}]  train_loss={train_loss:.4f}")
            client_updates.append(updated_arrays)

        # Aggregation strategy
        if strategy_name == "FedAvg":
            # Plain arithmetic mean -- no defense
            new_arrays = [
                np.mean([upd[i] for upd in client_updates], axis=0).astype(np.float32)
                for i in range(len(client_updates[0]))
            ]
            print(f"\n  [FedAvg] Plain mean aggregation applied.")
            print(
                f"  [FedAvg] All {num_clients} clients contributed equally -- "
                f"including {num_byzantine} Byzantine (NO FILTER)."
            )
            flagged_indices = []  # FedAvg has no defense -- never flags anyone

        elif strategy_name == "Krum":
            # Distance-based Krum selection
            num_select       = max(1, num_clients - num_byzantine - 2)
            selected_indices = krum_select(client_updates, num_to_select=num_select)
            selected_updates = [client_updates[i] for i in selected_indices]
            new_arrays       = krum_aggregate(selected_updates)
            dropped          = [i for i in range(num_clients) if i not in selected_indices]
            flagged_indices  = dropped  # unify naming with FLTrust's flagged_indices
            print(f"\n  [Krum] Selected: {selected_indices} | Dropped: {dropped}")
            for i in dropped:
                print(f"  [Krum] WARNING: Client {i:2d} [{roles[i]}] DROPPED (high Euclidean distance from cluster)")
            if rare_client:
                rare_idx = num_clients - 1
                if rare_idx in dropped:
                    print(
                        f"\n  [Krum] *** FALSE POSITIVE: Rare client {rare_idx} DROPPED "
                        f"(legitimate but geometrically distant) ***"
                    )
                else:
                    print(f"\n  [Krum] OK: Rare client {rare_idx} correctly kept.")

        else:
            # FLTrust cosine-trust aggregation
            with torch.no_grad():
                for p, arr in zip(global_model.parameters(), global_arrays):
                    p.copy_(torch.tensor(arr, dtype=torch.float32))

            new_arrays, trust_scores, flagged_indices = fltrust_aggregate(
                global_model   = global_model,
                client_updates = client_updates,
                root_data      = root_data,
                server_lr      = cfg.FLTRUST_SERVER_LR,
                min_trust      = cfg.FLTRUST_MIN_TRUST_SCORE,
            )
            print(f"\n  [FLTrust] Per-Client Trust Scores (cosine vs server root gradient):")
            for idx, trust in enumerate(trust_scores):
                flag = "[BYZANTINE SUSPECT]" if idx in flagged_indices else "[trusted         ]"
                print(
                    f"  [FLTrust] Client {idx:2d} [{roles[idx]:10s}]  "
                    f"trust={trust:.4f}  {flag}"
                )
            print(
                f"\n  [FLTrust] Flagged Byzantine: {flagged_indices}  "
                f"(expected adversarial clients: {list(range(num_byzantine))})"
            )
            if rare_client:
                rare_idx = num_clients - 1
                if rare_idx in flagged_indices:
                    print(
                        f"  [FLTrust] *** FALSE POSITIVE: Rare client {rare_idx} incorrectly flagged ***"
                    )
                else:
                    print(
                        f"  [FLTrust] *** TRUE NEGATIVE:  Rare client {rare_idx} correctly PRESERVED "
                        f"(direction-aligned despite distribution shift) ***"
                    )

        # Update shared global model
        global_arrays = new_arrays
        with torch.no_grad():
            for p, arr in zip(global_model.parameters(), global_arrays):
                p.copy_(torch.tensor(arr, dtype=torch.float32))

    # Final SHA-256 hash (tamper-evident audit trail)
    h = hashlib.sha256()
    for arr in global_arrays:
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    model_hash = "0x" + h.hexdigest()
    print(f"\n  [{strategy_name} | {byzantine_ratio*100:.0f}% Byzantine] Final Model SHA-256: {model_hash}")
    logger.info(f"Finished {strategy_name} | {byzantine_ratio*100:.0f}% Byzantine simulation.")

    # ── Byzantine-detection accuracy ─────────────────────────────────────
    # TP: byzantine client correctly flagged/dropped
    # FP: benign (or rare) client incorrectly flagged/dropped
    # FN: byzantine client missed
    # TN: benign client correctly left alone
    true_byzantine = set(i for i in range(num_clients) if roles[i] == "byzantine")
    flagged        = set(flagged_indices)
    non_byzantine  = set(range(num_clients)) - true_byzantine

    tp = len(true_byzantine & flagged)
    fn = len(true_byzantine - flagged)
    fp = len(non_byzantine & flagged)
    tn = len(non_byzantine - flagged)

    tpr = tp / len(true_byzantine) if true_byzantine else float("nan")
    tnr = tn / len(non_byzantine) if non_byzantine else float("nan")
    balanced_accuracy = (
        (tpr + tnr) / 2 if not (np.isnan(tpr) or np.isnan(tnr)) else float("nan")
    )

    return {
        "strategy": strategy_name,
        "byzantine_ratio": byzantine_ratio,
        "num_byzantine": num_byzantine,
        "roles": roles,
        "flagged_indices": sorted(flagged),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tpr": tpr, "tnr": tnr,
        "balanced_accuracy": balanced_accuracy,
        "final_arrays": global_arrays,
        "model_hash": model_hash,
    }


def main():
    print("\n" + "=" * 70)
    print("  AURA Byzantine Benchmark  --  Section 5.1.5  (Hypothesis H2)")
    print("=" * 70)
    num_clients = 10

    # Experiment 1: Byzantine Ratio Sweep
    print("\n>>> EXPERIMENT 1: Byzantine Ratio Sweep (FedAvg vs FLTrust)")
    print("    Tests how each strategy degrades as attacker fraction grows.")
    for ratio in [0.1, 0.2, 0.3, 0.4]:
        run_experiment("FedAvg",  num_clients, ratio)
        run_experiment("FLTrust", num_clients, ratio)

    # Experiment 2: Rare Client Preservation
    print("\n" + "=" * 70)
    print(">>> EXPERIMENT 2: Rare Client Preservation")
    print("    Krum (geometric) vs FLTrust (directional) on honest outlier client.")
    print("=" * 70)
    run_experiment("Krum",    num_clients, byzantine_ratio=0.1, rare_client=True)
    run_experiment("FLTrust", num_clients, byzantine_ratio=0.1, rare_client=True)

    print("\n" + "=" * 70)
    print("  Byzantine Benchmark Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
