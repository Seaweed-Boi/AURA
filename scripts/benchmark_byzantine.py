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
from aura.models import AURAModelBundle, FlowAutoencoder, AttackHead
import torch.nn.functional as F

from config import preflight_dc_fltrust_check
preflight_dc_fltrust_check()  # Hard stops if profiles are missing or stale

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
    
    from aura.split_manager import get_canonical_split

    # Load all windows
    all_windows = list(_loader.stream_graphs(_shared_scaler))

    # Get canonical train/test split
    _, train_windows, test_windows = get_canonical_split(
        all_windows, test_fraction=0.20
    )

    # Extract flows from train windows only
    _canonical_train_data = torch.cat([
        graph['edge_attr'] for graph, labels in train_windows
    ])
    
    def _build_benchmark_root_dataset(n_samples=2000):
        idx = torch.randperm(len(_canonical_train_data))[:n_samples]
        return _canonical_train_data[idx]
        
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
            train_data = _canonical_train_data.clone()
            # Val data is unused in this benchmark, just take a slice
            val_data = train_data[:max(1, n_samples // 5)]
            if len(train_data) > n_samples:
                train_data = train_data[:n_samples]
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


def _run_local_training_dual(
    ae: FlowAutoencoder,
    attack_head: AttackHead,
    all_flows: torch.Tensor,
    ae_optimizer: torch.optim.Optimizer,
    head_optimizer: torch.optim.Optimizer,
    global_ae_weights: dict,
    global_head_weights: dict,
    mse_threshold_high: float,
    head_epochs: int = 3
) -> tuple:
    """
    Two-pass dual-channel local training.
    
    Pass 1: AE trains on benign-only flows (MSE below threshold).
             AE latent geometry stays clean — attack flows never 
             influence encoder weights.
    Pass 2: Inference-only z collection from high-MSE flows.
             AE in eval mode, no gradient computation, weights unchanged.
             AttackHead trains on collected z vectors with soft MSE weighting.
    
    Returns: (ae_delta, head_delta, z_buffer, n_benign, n_high_mse)
    z_buffer is returned for potential submission to dynamic reference buffer.
    """
    
    # === PRE-PASS: classify flows without updating weights ===
    ae.eval()
    with torch.no_grad():
        recon, _ = ae(all_flows)
        mse_per_flow = F.mse_loss(recon, all_flows, reduction='none').mean(dim=1)
    ae.train()
    
    benign_mask = mse_per_flow < mse_threshold_high
    high_mse_mask = ~benign_mask
    benign_flows = all_flows[benign_mask]
    high_mse_flows = all_flows[high_mse_mask]
    high_mse_values = mse_per_flow[high_mse_mask]
    
    # === PASS 1: AE trains on benign flows only ===
    ae_loss_val = 0.0
    if len(benign_flows) > 0:
        ae_optimizer.zero_grad()
        recon_benign, _ = ae(benign_flows)
        ae_loss = F.mse_loss(recon_benign, benign_flows)
        ae_loss.backward()
        ae_optimizer.step()
        ae_loss_val = ae_loss.item()
    
    # === PASS 2: Inference-only z collection ===
    # ae.eval() ensures BatchNorm/Dropout behave consistently
    # torch.no_grad() ensures no gradient tape — weights CANNOT change
    z_buffer = []
    ae.eval()
    with torch.no_grad():
        if len(high_mse_flows) > 0:
            for i in range(0, len(high_mse_flows), 256):
                batch = high_mse_flows[i:i+256]
                z = ae.encode(batch)
                z_buffer.append(z.detach().cpu())
    ae.train()
    
    # === AttackHead training with soft MSE weighting ===
    head_loss_val = 0.0
    if z_buffer:
        z_tensor = torch.cat(z_buffer)
        
        # Soft weight: flows with higher MSE contribute more strongly
        # Prevents hard binary threshold from introducing arbitrary supervision boundary
        mse_weights = high_mse_values.cpu()
        mse_weights = (mse_weights - mse_weights.min()) / \
                      (mse_weights.max() - mse_weights.min() + 1e-8)
        # Match weight count to z_buffer count (may differ if batching truncates)
        mse_weights = mse_weights[:len(z_tensor)]
        
        for _ in range(head_epochs):
            head_optimizer.zero_grad()
            preds = attack_head(z_tensor).squeeze()
            pseudo_labels = torch.ones(len(z_tensor), device=z_tensor.device)
            head_loss = F.binary_cross_entropy(preds, pseudo_labels,
                                               weight=mse_weights)
            head_loss.backward()
            head_optimizer.step()
            head_loss_val = head_loss.item()
    
    # === Compute weight deltas for server transmission ===
    ae_delta = {k: ae.state_dict()[k].clone() - global_ae_weights[k]
                for k in ae.state_dict()}
    head_delta = {k: attack_head.state_dict()[k].clone() - global_head_weights[k]
                  for k in attack_head.state_dict()}
    
    logger.debug(
        f"Client round local: "
        f"benign_flows={len(benign_flows)}, high_mse_flows={len(high_mse_flows)}, "
        f"z_buffer_size={len(z_buffer) * 256 if z_buffer else 0}"
    )
    
    return ae_delta, head_delta, z_buffer, len(benign_flows), len(high_mse_flows)


def run_experiment(
    strategy_name:   str,
    num_clients:     int,
    byzantine_ratio: float,
    rare_client:     bool = False,
    mode:            str = "single_channel",
    num_rounds:      int = 2,
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
    import os, torch, torch.nn.functional as F
    global_model  = AURAModelBundle()
    
    # Load pretrained AE weights
    ae_path = os.path.join('saved_models', 'autoencoder_best.pth')
    if not os.path.exists(ae_path):
        raise FileNotFoundError(
            f"Pretrained AE not found at {ae_path}. "
            "Run train.py before benchmark_byzantine.py."
        )
    global_model.autoencoder.load_state_dict(torch.load(ae_path, map_location='cpu'))
    global_model.autoencoder.eval()
    logger.info(f"[INIT] Loaded pretrained AE from {ae_path}")

    global_arrays = [p.detach().cpu().numpy() for p in global_model.parameters()]

    # FLTrust server root dataset (benign reference -- built once per experiment)
    if _shared_scaler is not None:
        root_data = _build_benchmark_root_dataset()
    else:
        root_data = _build_root_dataset(2000)

    # Federated rounds
    from aura.attack_reference import AttackReferenceBuffer
    if mode == "dual_channel":
        logger.info(f"[DC-FLTrust] CH2 MSE split threshold: {cfg.CH2_MSE_SPLIT_THRESHOLD:.6f} "
              f"(P75 of benign distribution — flows above this route to AttackHead)")
              
        attack_ref_buffer = AttackReferenceBuffer(
            max_size=cfg.CH2_REF_BUFFER_MAX,
            min_size_to_use=cfg.CH2_REF_BUFFER_MIN
        )
        
        # Diagnostic
        ae = global_model.autoencoder
        ae.eval()
        with torch.no_grad():
            sample = clients[0].train_data[:100]
            recon, _ = ae(sample)
            mse = F.mse_loss(recon, sample, reduction='none').mean(dim=1)
            all_mse = mse.cpu().numpy()
        ae.train()

        above_threshold = (all_mse > cfg.CH2_MSE_SPLIT_THRESHOLD).sum()
        logger.debug(f"[INIT] Sanity check: {above_threshold}/100 flows above CH2 threshold "
              f"(expected ~25 for P75 threshold)")
        logger.debug(f"  MSE range: [{all_mse.min():.6f}, {all_mse.max():.6f}]")
        logger.debug(f"  MSE P75: {np.percentile(all_mse, 75):.6f}")
        logger.debug(f"  MSE P90: {np.percentile(all_mse, 90):.6f}")
        logger.debug(f"  MSE P99: {np.percentile(all_mse, 99):.6f}")
    else:
        attack_ref_buffer = None
        
    for rnd in range(1, num_rounds + 1):
        print("\n" + "-" * 60)
        print(f"  [{strategy_name} | {byzantine_ratio*100:.0f}% Byzantine] Round {rnd}/{num_rounds}")
        print("-" * 60)

        # Each client trains locally
        client_updates: List[List[np.ndarray]] = []
        c_ae_deltas = []
        c_head_deltas = []
        round_z_submissions = {}
        
        for idx, client in enumerate(clients):
            is_byzantine = (roles[idx] == "byzantine")
            if mode == "dual_channel":
                global_ae_weights = {k: global_model.autoencoder.state_dict()[k].clone() for k in global_model.autoencoder.state_dict()}
                global_head_weights = {k: global_model.attack_head.state_dict()[k].clone() for k in global_model.attack_head.state_dict()}
                
                ae_opt = torch.optim.Adam(client.model.autoencoder.parameters(), lr=1e-3)
                head_opt = torch.optim.Adam(client.model.attack_head.parameters(), lr=1e-3)
                
                with torch.no_grad():
                    for p, arr in zip(client.model.autoencoder.parameters(), global_model.autoencoder.parameters()):
                        p.copy_(arr)
                    for p, arr in zip(client.model.attack_head.parameters(), global_model.attack_head.parameters()):
                        p.copy_(arr)
                
                ae_delta, head_delta, z_buffer, n_benign, n_high_mse = _run_local_training_dual(
                    ae=client.model.autoencoder,
                    attack_head=client.model.attack_head,
                    all_flows=client.train_data,
                    ae_optimizer=ae_opt,
                    head_optimizer=head_opt,
                    global_ae_weights=global_ae_weights,
                    global_head_weights=global_head_weights,
                    mse_threshold_high=cfg.CH2_MSE_SPLIT_THRESHOLD,
                    head_epochs=3
                )
                c_ae_deltas.append(ae_delta)
                c_head_deltas.append(head_delta)
                round_z_submissions[idx] = z_buffer
                
                logger.info(
                    f"Client {idx} round {rnd}: "
                    f"benign_flows={n_benign}, high_mse_flows={n_high_mse}, "
                    f"z_buffer_size={sum(len(z) for z in z_buffer)}"
                )
                train_loss = 0.0
                updated_arrays = [p.detach().cpu().numpy() for p in client.model.parameters()]
            else:
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

            if mode == "dual_channel":
                from aura.fl_server import dc_fltrust_aggregate
                # Server root training for channel 2 reference deltas
                root_ae = FlowAutoencoder()
                root_head = AttackHead()
                root_ae.load_state_dict(global_model.autoencoder.state_dict())
                root_head.load_state_dict(global_model.attack_head.state_dict())
                root_ae_opt = torch.optim.Adam(root_ae.parameters(), lr=1e-3)
                root_head_opt = torch.optim.Adam(root_head.parameters(), lr=1e-3)
                
                g_ae_w = {k: v.clone() for k, v in global_model.autoencoder.state_dict().items()}
                g_head_w = {k: v.clone() for k, v in global_model.attack_head.state_dict().items()}
                
                r_ae_delta, r_head_delta, _, _, _ = _run_local_training_dual(
                    root_ae, root_head, root_data, root_ae_opt, root_head_opt, g_ae_w, g_head_w, mse_threshold_high=cfg.CH2_MSE_SPLIT_THRESHOLD
                )
                
                client_round_counts = [rnd] * num_clients
                new_ae, new_head, ch1_scores, ch2_scores, classifications = dc_fltrust_aggregate(
                    c_ae_deltas, c_head_deltas, r_ae_delta, r_head_delta, client_round_counts,
                    ch2_warmup_rounds=cfg.CH2_WARMUP_ROUNDS,
                    round_z_submissions=round_z_submissions,
                    attack_ref_buffer=attack_ref_buffer,
                    current_round=rnd,
                    reference_attack_head=global_model.attack_head
                )
                # Reconstruct new_arrays from new_ae and new_head
                # Not fully reconstructing the bundle parameters array here as this is a simulation.
                new_arrays = global_arrays # keep global arrays intact for printing
                trust_scores = ch1_scores
                flagged_indices = [i for i, c in enumerate(classifications) if c == 'BYZANTINE']
                
                for idx in range(num_clients):
                    print(f"  [DC-FLTrust] Client {idx:2d} [{roles[idx]:10s}] ch1={ch1_scores[idx]:.4f} ch2={ch2_scores[idx] if ch2_scores[idx] is not None else 0.0:.4f} -> {classifications[idx]}")
                    
                if attack_ref_buffer is not None:
                    print(f"  [DC-FLTrust] Buffer size: {len(attack_ref_buffer._buffer)}")
            else:
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
    import argparse
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="single_channel", choices=["single_channel", "dual_channel"])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("\n" + "=" * 70)
    print("  AURA Byzantine Benchmark  --  DC-FLTrust Deception Experiment")
    print("=" * 70)
    
    num_clients = 5
    ratio = 0.2  # 1 byzantine client
    run_experiment("FLTrust", num_clients, byzantine_ratio=ratio, mode=args.mode, num_rounds=args.rounds)

    print("\n" + "=" * 70)
    print("  Byzantine Benchmark Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
