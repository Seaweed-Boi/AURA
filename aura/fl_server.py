"""
aura/fl_server.py — Flower FL Server: FLTrust Aggregation + Straggler Policy
=============================================================================

This server implements two critical security properties:

1. BYZANTINE ROBUSTNESS (FLTrust Aggregation — Upgrade 6)
   --------------------------------------------------------
   Standard FedAvg is vulnerable to model poisoning. Legacy Krum (distance-based;
   see krum_select / krum_aggregate below) guards geometrically but has a critical flaw: it rejects
   clients whose updates are geometrically distant even if they are
   *honestly* trained on rare or skewed data distributions (e.g., a
   hospital with rare disease traffic).  This causes false-positive
   rejections of legitimate but outlier clients.

   FLTrust (Cao et al., 2020) fixes this:
     - The server holds a small clean root dataset (FLTRUST_ROOT_SAMPLES
       synthetic benign samples).
     - Each round, the server trains one optimisation step on the root
       dataset to obtain a reference gradient direction.
     - Each client update is scored by its cosine similarity with the
       server's gradient direction — ReLU ensures negative similarity
       (i.e., adversarial reversal) maps to zero trust.
     - Client updates are re-scaled to the server update's magnitude
       before weighted aggregation, preventing magnitude-based amplification.

   Key advantage: a hospital client with unusual-but-legitimate data has
   a gradient that still *points in the same direction* as improvement on
   normal traffic. Legacy Krum would drop it; FLTrust keeps it with proportional
   trust.

   **Active aggregation:** FLTrust in aggregate_fit. **Legacy fallback only:**
   krum_select / krum_aggregate (not used in the default path) retained for rollback or experiments.

2. STRAGGLER TIMEOUT POLICY
   --------------------------
   In synchronous FL, if a client disconnects mid-round, the server
   blocks indefinitely — causing a Denial-of-Service against the entire
   federation.

   AURA implements an explicit timeout policy:
     - After `round_timeout_sec` seconds, unreceived client updates
       are DROPPED from the aggregation round.
     - If fewer than `min_clients` responses arrive, the round is
       ABANDONED and the previous global model is preserved.
     - A warning is logged for operator review.

3. IMMUTABLE AUDIT LOG
   ----------------------
   After each successful aggregation, the server hashes the model weights
   (SHA-256) and writes the hash to the AURA blockchain module.
   This creates a tamper-evident chain of custody for all model updates.
"""

import hashlib
import io
import json
import logging
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
    EvaluateRes,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy import FedAvg

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from aura.models import AURAModelBundle

from config import preflight_dc_fltrust_check
preflight_dc_fltrust_check()

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy Krum aggregation (fallback only — not used by aggregate_fit; FLTrust is active)
# ─────────────────────────────────────────────────────────────────────────────

def krum_select(
    updates:       List[List[np.ndarray]],
    num_to_select: int = cfg.KRUM_NUM_TO_SELECT,
) -> List[int]:
    """
    Krum Selection Algorithm (Blanchard et al., 2017).

    Algorithm
    ---------
    Given n client weight updates {w_1, ..., w_n}:
    1. For each client i, flatten its weight update into a 1D vector.
    2. Compute pairwise squared Euclidean distances: d(i,j) = ||w_i - w_j||²
    3. For each client i, compute the Krum score:
         s(i) = Σ_{j ∈ N_k(i)} d(i, j)
       where N_k(i) = k nearest neighbours of i (k = n - num_to_select - 2)
    4. Select the num_to_select clients with the LOWEST Krum scores.

    Poisoned clients produce updates far from the cluster → high scores → dropped.

    Parameters
    ----------
    updates        : List of per-client parameters (each is a list of ndarrays)
    num_to_select  : How many clients to keep (KRUM_NUM_TO_SELECT in config)

    Returns
    -------
    Selected client indices (those with lowest Krum scores)
    """
    n = len(updates)
    if n <= num_to_select:
        logger.warning("Krum: fewer clients than num_to_select — accepting all.")
        return list(range(n))

    # Flatten each client's parameters to a single 1D vector
    flat = []
    for client_params in updates:
        flat.append(np.concatenate([p.flatten() for p in client_params]))

    # k = n - num_to_select - 2  (guaranteed Byzantine tolerance formula)
    k = max(1, n - num_to_select - 2)

    scores = []
    for i in range(n):
        # Squared Euclidean distances from client i to all others
        dists = sorted([
            float(np.sum((flat[i] - flat[j]) ** 2))
            for j in range(n) if j != i
        ])
        # Krum score = sum of k smallest distances
        scores.append(sum(dists[:k]))

    # Rank clients by score (ascending) and select the best num_to_select
    ranked = sorted(range(n), key=lambda idx: scores[idx])
    selected = ranked[:num_to_select]

    dropped = [i for i in range(n) if i not in selected]
    if dropped:
        logger.warning(
            f"[KRUM] Dropped client indices {dropped} as potential outliers.  "
            f"Scores: {[round(s, 2) for s in scores]}"
        )
    else:
        logger.info(f"[KRUM] All clients accepted.  Scores: {[round(s, 2) for s in scores]}")

    return selected


def krum_aggregate(
    selected_updates: List[List[np.ndarray]],
) -> List[np.ndarray]:
    """
    Aggregate selected (Krum-filtered) client updates by simple mean.

    By the time we reach this function, Byzantine clients have already been
    filtered by krum_select.  A simple mean of the remaining honest updates
    produces the new global model.

    Shape preservation:  each result array has the same shape as input arrays.
    """
    # Cast to float32: keeps dtype consistent with PyTorch model weights so
    # SHA-256 hashes computed before and after Flower serialization always match.
    return [
        np.mean([update[i] for update in selected_updates], axis=0).astype(np.float32)
        for i in range(len(selected_updates[0]))
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Root Dataset (server trusted data for FLTrust)
# ─────────────────────────────────────────────────────────────────────────────

def _build_root_dataset(n_samples: int = cfg.FLTRUST_ROOT_SAMPLES) -> torch.Tensor:
    """
    Build the FLTrust server root dataset (trusted benign reference).

    Strategy (controlled by cfg.FL_ROOT_DATA_SOURCE):
    -------------------------------------------------
    "real" (default for research paper):
      Loads a GLOBALLY REPRESENTATIVE benign sample from NF-UNSW-NB15-v3.
      It explicitly avoids using specific client partitions to ensure the 
      server gradient is unbiased, preventing false-positive Byzantine flags 
      on honest Non-IID clients.

    "synthetic" (fallback / dataset-unavailable):
      Gaussian samples in the 0.35–0.50 range.
    """
    if cfg.FL_ROOT_DATA_SOURCE == "real":
        try:
            import pandas as pd
            from aura.data_loader import CICIDSDataLoader, DATASET_PATH
            
            _loader = CICIDSDataLoader()
            _scaler = _loader.fit_scaler()
            
            # Load the global CSV directly
            df = _loader._load_csv(str(DATASET_PATH))
            label_col = 'Label' if 'Label' in df.columns else cfg.LABEL_COL.strip()

            # Isolate all benign traffic globally
            if pd.api.types.is_numeric_dtype(df[label_col]):
                benign_df = df[df[label_col] == 0]
            else:
                benign_df = df[df[label_col].str.strip().str.upper() == "BENIGN"]

            # Sample globally so all org behaviors are represented
            sampled_df = benign_df.sample(n=min(n_samples, len(benign_df)), random_state=42)
            
            X = sampled_df[_loader._feature_cols].values.astype(np.float32)
            X_scaled = _scaler.transform(X).clip(0, 1)
            
            root = torch.tensor(X_scaled, dtype=torch.float32)

            # Tile if the available dataset is smaller than requested
            if len(root) < n_samples:
                repeats = (n_samples // len(root)) + 1
                root = root.repeat(repeats, 1)[:n_samples]

            logger.info(
                f"[FLTrust] Global real root dataset loaded "
                f"({len(root)} unbiased samples)."
            )
            return root
            
        except Exception as e:
            raise RuntimeError(
                f"FATAL: Real root dataset unavailable. "
                f"Falling back to synthetic Gaussian noise invalidates FLTrust evaluation. "
                f"Fix the dataset path. Original error: {e}"
            )
# ─────────────────────────────────────────────────────────────────────────────
# FLTrust Aggregation (Upgrade 6 — active path in aggregate_fit; Krum is legacy fallback only)
# ─────────────────────────────────────────────────────────────────────────────

def _retrain_reference_head(head, z_tensor, epochs=3):
    """Retrain the server's reference AttackHead on the latest dynamic z buffer."""
    import torch.optim as optim
    import torch.nn.functional as F
    opt = optim.Adam(head.parameters(), lr=1e-3)
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        preds = head(z_tensor).squeeze()
        # Pseudo-label 1.0 because this is a confirmed attack buffer
        pseudo_labels = torch.ones_like(preds)
        loss = F.binary_cross_entropy(preds, pseudo_labels)
        loss.backward()
        opt.step()
    head.eval()

def ae_only_fltrust_aggregate(
    client_ae_deltas: list,
    root_ae_delta: dict,
) -> tuple:
    """
    AE-Only FLTrust (Mode A baseline).
    
    Only AE gradients are transmitted and evaluated.
    AttackHead does not exist in this mode.
    This is the pre-DC-FLTrust baseline — federate only the AE,
    ignore attack-pattern knowledge entirely.
    
    Returns: (aggregated_ae_weights, trust_scores)
    """
    def flatten(delta: dict) -> torch.Tensor:
        return torch.cat([v.flatten().float() for v in delta.values()])
    
    root_flat = flatten(root_ae_delta)
    trust_scores = []
    
    for ae_delta in client_ae_deltas:
        client_flat = flatten(ae_delta)
        sim = F.cosine_similarity(
            client_flat.unsqueeze(0),
            root_flat.unsqueeze(0)
        ).item()
        trust_scores.append(max(0.0, sim))  # ReLU
    
    # Normalize
    total = sum(trust_scores)
    if total > 0:
        normalized = [s / total for s in trust_scores]
    else:
        normalized = [1.0 / len(trust_scores)] * len(trust_scores)
    
    # Weighted aggregation
    agg_ae = {
        k: sum(w * d[k] for w, d in zip(normalized, client_ae_deltas))
        for k in client_ae_deltas[0]
    }
    
    return agg_ae, trust_scores


def joint_dual_fltrust_aggregate(
    client_ae_deltas: list,
    client_head_deltas: list,
    root_ae_delta: dict,
    root_head_delta: dict,
    client_round_counts: list,
    ch2_warmup_rounds: int = 10,
    ch1_weight: float = 0.7,
) -> tuple:
    """
    Joint Dual-Channel FLTrust (Mode B baseline).
    
    Both AE and AttackHead gradients evaluated independently,
    but COMBINED into a single trust weight before aggregation:
    trust = ch1_weight * ch1 + (1 - ch1_weight) * ch2
    
    A client with honest AE (ch1=0.8) but corrupted AttackHead (ch2=0.1)
    receives trust = 0.7*0.8 + 0.3*0.1 = 0.59 — reduced but NOT excluded.
    The corrupted AttackHead gradient still enters global aggregation
    at 59% weight rather than the 80% it would receive if AE-only.
    
    This is the key difference from DC-FLTrust (Mode C):
    Mode B reduces but does not eliminate corrupted gradient influence.
    Mode C detects BYZANTINE_FAKE_ATTACK and excludes the head gradient entirely.
    
    Returns: (aggregated_ae, aggregated_head, combined_trust_scores, ch1_scores, ch2_scores)
    """
    def flatten(delta: dict) -> torch.Tensor:
        return torch.cat([v.flatten().float() for v in delta.values()])
    
    root_ae_flat = flatten(root_ae_delta)
    root_head_flat = flatten(root_head_delta)
    
    ch1_scores = []
    ch2_scores = []
    combined_scores = []
    
    for ae_delta, head_delta, rounds in zip(
        client_ae_deltas, client_head_deltas, client_round_counts
    ):
        ch1 = max(0.0, F.cosine_similarity(
            flatten(ae_delta).unsqueeze(0),
            root_ae_flat.unsqueeze(0)
        ).item())
        
        if rounds >= ch2_warmup_rounds and head_delta is not None:
            ch2 = max(0.0, F.cosine_similarity(
                flatten(head_delta).unsqueeze(0),
                root_head_flat.unsqueeze(0)
            ).item())
        else:
            ch2 = 0.0  # warmup — treat as no attack signal
        
        combined = ch1_weight * ch1 + (1 - ch1_weight) * ch2
        
        ch1_scores.append(ch1)
        ch2_scores.append(ch2)
        combined_scores.append(combined)
    
    # Normalize combined scores
    total = sum(combined_scores)
    if total > 0:
        normalized = [s / total for s in combined_scores]
    else:
        normalized = [1.0 / len(combined_scores)] * len(combined_scores)
    
    # Both channels aggregated with same combined weight
    # This is the flaw: corrupted head gradients enter aggregation
    # at the same weight as honest AE gradients
    agg_ae = {
        k: sum(w * d[k] for w, d in zip(normalized, client_ae_deltas))
        for k in client_ae_deltas[0]
    }
    
    valid_heads = [(w, d) for w, d in zip(normalized, client_head_deltas)
                   if d is not None]
    if valid_heads:
        agg_head = {
            k: sum(w * d[k] for w, d in valid_heads)
            for k in valid_heads[0][1]
        }
    else:
        agg_head = None
    
    return agg_ae, agg_head, combined_scores, ch1_scores, ch2_scores


def dc_fltrust_aggregate(
    client_ae_deltas: list,
    client_head_deltas: list,
    root_ae_delta: dict,
    root_head_delta: dict,
    client_round_counts: list,
    ch2_warmup_rounds: int = 10,
    round_z_submissions: dict = None,
    attack_ref_buffer = None,
    current_round: int = 0,
    reference_attack_head = None
) -> tuple:
    """
    Dual-channel FLTrust aggregation.
    
    Channel 1: cosine similarity of AE delta vs root AE delta
    Channel 2: cosine similarity of AttackHead delta vs root head delta
               Only active after ch2_warmup_rounds per client
    
    Returns: (aggregated_ae_weights, aggregated_head_weights, 
               ch1_scores, ch2_scores, classifications)
    """
    
    def flatten(delta: dict) -> torch.Tensor:
        return torch.cat([v.flatten() for v in delta.values()])
    
    def cosine_trust(client_flat, root_flat) -> float:
        sim = F.cosine_similarity(
            client_flat.unsqueeze(0), 
            root_flat.unsqueeze(0)
        ).item()
        return max(0.0, sim)  # ReLU — negative similarity → zero trust
    
    root_ae_flat = flatten(root_ae_delta)
    root_head_flat = flatten(root_head_delta)
    
    ch1_scores = []
    ch2_scores = []
    classifications = []
    
    for i, (ae_delta, head_delta, rounds) in enumerate(
        zip(client_ae_deltas, client_head_deltas, client_round_counts)
    ):
        # Channel 1: always active
        ch1 = cosine_trust(flatten(ae_delta), root_ae_flat)
        
        # Channel 2: gated by warmup
        if rounds >= ch2_warmup_rounds:
            if head_delta is not None:
                ch2 = cosine_trust(flatten(head_delta), root_head_flat)
            else:
                ch2 = None  # No attack signal submitted
        else:
            ch2 = 'WARMUP'  # Special flag for warmup
        
        ch1_scores.append(ch1)
        ch2_scores.append(ch2 if isinstance(ch2, float) else 0.0)
        
        # Classify client based on dual-channel scores
        if ch2 == 'WARMUP':
            classification = 'WARMUP'
        elif ch2 is None:
            # Client past warmup but didn't submit a head (no attacks seen)
            if ch1 > 0.5:
                classification = 'HEALTHY'
            else:
                classification = 'BYZANTINE'
        else:
            # Client submitted a head delta
            if ch1 > 0.5 and ch2 > 0.5:
                classification = 'UNDER_ATTACK'
            elif ch1 > 0.5 and ch2 <= 0.5:
                classification = 'BYZANTINE_FAKE_ATTACK'
            elif ch1 <= 0.5 and ch2 > 0.5:
                classification = 'BYZANTINE_FAKE_ATTACK'
            else:
                classification = 'BYZANTINE'
        
        classifications.append(classification)
    
    # Aggregate AE weights — exclude Byzantine clients from ch1
    ch1_weights = torch.tensor([
        s if c not in ('BYZANTINE', 'BYZANTINE_FAKE_ATTACK') else 0.0
        for s, c in zip(ch1_scores, classifications)
    ])
    ch1_norm = ch1_weights.sum()
    if ch1_norm > 0:
        ch1_weights = ch1_weights / ch1_norm
    
    # === Attack Reference Buffer Update ===
    if round_z_submissions is not None and attack_ref_buffer is not None:
        for client_id, classification in enumerate(classifications):
            if classification == 'UNDER_ATTACK':
                if client_id < len(round_z_submissions) and round_z_submissions[client_id] is not None:
                    z_submission = round_z_submissions[client_id]
                    if len(z_submission) > 0:
                        if isinstance(z_submission, list):
                            z_submission = torch.cat(z_submission)
                        attack_ref_buffer.update(
                            z_submission, 
                            round_num=current_round
                        )
        
        if attack_ref_buffer.is_ready():
            new_reference = attack_ref_buffer.get_reference_tensor()
            if reference_attack_head is not None:
                _retrain_reference_head(reference_attack_head, new_reference)
                channel2_reference = reference_attack_head
            logger.info(f"Round {current_round}: ch2 reference updated from buffer "
                        f"({attack_ref_buffer.stats()['buffer_size']} vectors)")
        else:
            logger.info(f"Round {current_round}: buffer not ready "
                        f"({len(attack_ref_buffer._buffer)}/{attack_ref_buffer.min_size_to_use}), "
                        f"using static reference")
    
    # Aggregate head weights — only from clients with active ch2 and not Byzantine
    ch2_weights = torch.tensor([
        (s if s is not None and isinstance(s, float) else 0.0)
        if c not in ('BYZANTINE', 'BYZANTINE_FAKE_ATTACK', 'WARMUP') else 0.0
        for s, c in zip(ch2_scores, classifications)
    ])
    ch2_norm = ch2_weights.sum()
    if ch2_norm > 0:
        ch2_weights = ch2_weights / ch2_norm
    
    # Weighted aggregation
    agg_ae = {k: sum(w * d[k] for w, d in zip(ch1_weights, client_ae_deltas))
              for k in client_ae_deltas[0]}
    
    active_head_updates = [
        (w, d) for w, d, c in zip(ch2_weights, client_head_deltas, classifications)
        if c not in ('BYZANTINE', 'BYZANTINE_FAKE_ATTACK', 'WARMUP') and d is not None
    ]
    
    if active_head_updates:
        agg_head = {k: sum(w * d[k] for w, d in active_head_updates)
                    for k in active_head_updates[0][1]}
    else:
        agg_head = None  # no head update this round
    
    return agg_ae, agg_head, ch1_scores, ch2_scores, classifications


def fltrust_aggregate(
    global_model:    AURAModelBundle,
    client_updates:  List[List[np.ndarray]],   # per-client list-of-arrays
    root_data:       torch.Tensor,             # server's trusted benign dataset
    server_lr:       float = cfg.FLTRUST_SERVER_LR,
    min_trust:       float = cfg.FLTRUST_MIN_TRUST_SCORE,
) -> Tuple[List[np.ndarray], List[float], List[int]]:
    """
    FLTrust Aggregation (Cao et al., 2020).

    Algorithm
    ---------
    1. Compute server reference update:
         Clone the global model, train one step on root_data (benign),
         delta_server = new_params − old_params.
         Flatten to 1-D vector: server_vec.

    2. For each client i:
         delta_i    = client_params_i − global_params
         client_vec = flatten(delta_i)
         trust_i    = ReLU(cosine_similarity(client_vec, server_vec))
         # Positive = update points same direction as benign improvement
         # Negative = adversarial reversal → ReLU ⇒ zero trust, flagged.

    3. Normalise client update magnitude to server update magnitude:
         scale_i = ||server_vec|| / (||client_vec|| + ε)
         normalised_i = scale_i × delta_i

    4. Weighted aggregation:
         new_global = global + Σ(trust_i / Σtrust) × normalised_i

    Parameters
    ----------
    global_model   : The current global AURAModelBundle.
    client_updates : Each element is a list of np.ndarray (Flower format).
    root_data      : [N, F] benign tensor (server's trusted data).
    server_lr      : LR for the one-step server gradient computation.
    min_trust      : Trust scores at or below this are flagged Byzantine.

    Returns
    -------
    (new_arrays, trust_scores, flagged_indices)
      new_arrays    : Aggregated model as list[np.ndarray] (same layout as input)
      trust_scores  : Per-client cosine trust score ∈ [0, 1]
      flagged_indices: Indices of clients with trust ≤ min_trust (Byzantine suspects)
    """
    # ── Step 1: Compute server reference one-step update ───────────────────
    server_model = AURAModelBundle()
    # Load current global weights
    global_param_list = [p.detach().cpu() for p in global_model.parameters()]
    with torch.no_grad():
        for p, gp in zip(server_model.parameters(), global_param_list):
            p.copy_(gp)

    # One gradient step on root data (autoencoder reconstruction loss)
    optimiser = torch.optim.Adam(server_model.autoencoder.parameters(), lr=server_lr)
    server_model.autoencoder.train()
    optimiser.zero_grad()
    x_hat, _ = server_model.autoencoder(root_data)
    loss = nn.functional.mse_loss(x_hat, root_data)
    loss.backward()
    # Clip to match client training (preserves fairness in magnitude comparison)
    torch.nn.utils.clip_grad_norm_(server_model.autoencoder.parameters(), max_norm=1.0)
    optimiser.step()
    server_model.eval()

    # Server delta (new weights − old weights), flattened
    server_delta = [
        (p.detach().cpu() - gp)
        for p, gp in zip(server_model.parameters(), global_param_list)
    ]
    server_vec = torch.cat([d.flatten() for d in server_delta])   # [D]
    server_norm = server_vec.norm()                                # scalar

    # ── Step 2 & 3: Per-client trust scores + normalised deltas ───────────
    trust_scores: List[float] = []
    normalised_deltas: List[List[torch.Tensor]] = []   # per-client list of tensors

    global_arrays = [p.detach().cpu().numpy() for p in global_model.parameters()]

    for client_arrays in client_updates:
        # Build per-layer delta (client_params − global_params)
        client_delta = [
            torch.tensor(c_arr, dtype=torch.float32) - torch.tensor(g_arr, dtype=torch.float32)
            for c_arr, g_arr in zip(client_arrays, global_arrays)
        ]
        client_vec = torch.cat([d.flatten() for d in client_delta])   # [D]
        client_norm = client_vec.norm()

        # Cosine similarity with server gradient direction
        cos_sim = F.cosine_similarity(
            server_vec.unsqueeze(0), client_vec.unsqueeze(0)
        ).item()
        # ReLU: adversarial reversal (negative cos) ⇒ zero trust
        trust = max(0.0, cos_sim)
        trust_scores.append(trust)

        # Re-scale client delta to server update magnitude (prevents amplification)
        scale = float(server_norm) / (float(client_norm) + 1e-8)
        normalised_deltas.append([d * scale for d in client_delta])

    # ── Step 4: Weighted aggregation ────────────────────────────────────
    total_trust = sum(trust_scores) + 1e-8

    # Identify Byzantine suspects (zero or near-zero trust)
    flagged_indices = [
        i for i, t in enumerate(trust_scores) if t <= min_trust
    ]

    # Initialise new state as: global_params + weighted sum of normalised deltas
    new_arrays: List[np.ndarray] = []
    for layer_idx in range(len(global_arrays)):
        accumulated = torch.zeros_like(
            torch.tensor(global_arrays[layer_idx], dtype=torch.float32)
        )
        for i, (trust, deltas) in enumerate(zip(trust_scores, normalised_deltas)):
            accumulated += (trust / total_trust) * deltas[layer_idx]
        result = torch.tensor(global_arrays[layer_idx], dtype=torch.float32) + accumulated
        new_arrays.append(result.numpy().astype(np.float32))

    return new_arrays, trust_scores, flagged_indices


# ─────────────────────────────────────────────────────────────────────────────
# SHA-256 Model Hash
# ─────────────────────────────────────────────────────────────────────────────

def hash_model_weights(arrays: List[np.ndarray]) -> str:
    """
    Compute a SHA-256 hash over the concatenated model weight bytes.

    Normalises every array to C-contiguous float32 before hashing so the
    result is identical whether called on the server-side aggregated arrays
    or on the client-side after Flower's ndarrays_to_parameters round-trip.
    """
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return "0x" + h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Custom Flower Strategy: KrumFedAURA (FLTrust inside aggregate_fit; Krum helpers are legacy)
# ─────────────────────────────────────────────────────────────────────────────

class KrumFedAURA(FedAvg):
    """
    Custom Flower aggregation strategy extending FedAvg with:
      1. **FLTrust** Byzantine-robust aggregation (active — Upgrade 6; cosine trust vs server root).
      2. Straggler timeout + drop policy
      3. Per-round SHA-256 model hash logging
      4. Blockchain audit integration

    Class name remains **KrumFedAURA** for backward compatibility with dashboards and scripts.
    **krum_select / krum_aggregate** exist only as legacy fallback code paths, not used by aggregate_fit.

    Inheriting from FedAvg reuses Flower boilerplate (sampling, evaluation scheduling, etc.)
    while overriding `aggregate_fit` where FLTrust and audit logic live.
    """

    def __init__(
        self,
        fraction_fit:          float = 1.0,
        fraction_evaluate:     float = 1.0,
        min_fit_clients:       int   = cfg.FL_MIN_CLIENTS,
        min_available_clients: int   = cfg.FL_MIN_AVAILABLE,
        num_rounds:            int   = cfg.FL_NUM_ROUNDS,
        round_timeout_sec:     int   = cfg.FL_ROUND_TIMEOUT_SEC,
        blockchain_module=None,      # Optionally inject blockchain logger
    ):
        # Configure FedAvg base (we override aggregation but keep its scheduling)
        super().__init__(
            fraction_fit          = fraction_fit,
            fraction_evaluate     = fraction_evaluate,
            min_fit_clients       = min_fit_clients,
            min_available_clients = min_available_clients,
            # Round config function: tells clients how many local epochs to run
            on_fit_config_fn = lambda rnd: {
                "local_epochs": cfg.FL_LOCAL_EPOCHS,
                "round":        rnd,
                # Hint to client libraries; actual enforcement is server-side
                "timeout_sec":  round_timeout_sec,
            },
        )
        self.num_rounds        = num_rounds
        self.round_timeout_sec = round_timeout_sec
        self.blockchain        = blockchain_module
        self._model_version    = 0
        self._hash_history: List[dict] = []

        # FLTrust root dataset — generated once, reused every round.
        # This is the server's small trusted benign baseline.
        self._root_data: torch.Tensor = _build_root_dataset()
        # Global model reference — updated after each successful aggregation.
        # FLTrust needs the current global state to compute per-client deltas.
        self._global_model: AURAModelBundle = AURAModelBundle()

        # Per-round trust score history (for dashboard + Upgrade 3 detection log)
        self._trust_history: List[dict] = []

        # Clear the trusted registry at the start of each FL session so that
        # only the current session's final hash is present (1 hash per run).
        registry_path = Path(cfg.LOGS_DIR) / "hash_registry.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text("{}")
        logger.info("[REGISTRY] Cleared — fresh FL session starting.")

        logger.info(
            f"KrumFedAURA (FLTrust) strategy ready  |  "
            f"rounds={num_rounds}  timeout={round_timeout_sec}s  "
            f"root_samples={cfg.FLTRUST_ROOT_SAMPLES}  "
            f"server_lr={cfg.FLTRUST_SERVER_LR}"
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Override FedAvg's aggregate_fit to apply **FLTrust** (not plain FedAvg mean).

        Straggler Policy
        ----------------
        Flower already handles client timeouts internally via its gRPC layer
        (configured via flwr.server.ServerConfig(round_timeout=...)).
        Failed/timed-out clients arrive in the `failures` list, not in
        `results`.  We log them and proceed with whatever arrived on time.

        If fewer than min_fit_clients results arrive, we PRESERVE the previous
        global model (no aggregation) and log the stale round.
        """
        round_tag = f"[SERVER round={server_round}]"

        # ── Straggler Policy ────────────────────────────────────────────────
        n_received = len(results)
        n_failed   = len(failures)

        if n_failed > 0:
            logger.warning(
                f"{round_tag} {n_failed} client(s) timed out / failed "
                f"(straggler drop policy applied).  Proceeding with {n_received} responses."
            )

        # Use self.min_fit_clients so the quorum adapts when fewer orgs are
        # active (e.g., one quarantined).  cfg.FL_MIN_CLIENTS is only the
        # hard-coded default for standalone server runs; the dashboard always
        # passes len(active_orgs) explicitly at strategy instantiation time.
        min_needed = getattr(self, 'min_fit_clients', cfg.FL_MIN_CLIENTS)
        if n_received < min_needed:
            logger.error(
                f"{round_tag} Insufficient responses ({n_received} < "
                f"{min_needed}).  ABANDONING round — global model preserved."
            )
            return None, {"status": "abandoned", "received": n_received}

        # ── Extract weight arrays from Flower FitRes ─────────────────────────
        client_updates = []
        for client_proxy, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            client_updates.append(arrays)
            client_loss = fit_res.metrics.get("train_loss", "N/A")
            logger.info(f"{round_tag} Received update  |  "
                        f"num_examples={fit_res.num_examples}  loss={client_loss}")

        # ── FLTrust Aggregation ──────────────────────────────────────────────────────────────
        print(f"\n{round_tag} Running FLTrust aggregation on {n_received} updates …")
        aggregated, trust_scores, flagged_indices = fltrust_aggregate(
            global_model   = self._global_model,
            client_updates = client_updates,
            root_data      = self._root_data,
            server_lr      = cfg.FLTRUST_SERVER_LR,
            min_trust      = cfg.FLTRUST_MIN_TRUST_SCORE,
        )
        self._model_version += 1
        model_version_tag = f"v{self._model_version}.{server_round}"

        # ── Log per-client trust scores ──────────────────────────────────────────
        for idx, (trust, (cp, fr)) in enumerate(zip(trust_scores, results)):
            status = "BYZANTINE SUSPECT" if idx in flagged_indices else "trusted"
            print(
                f"{round_tag} [FLTrust] Client {idx} — "
                f"trust={trust:.4f}  [{status}]  "
                f"loss={fr.metrics.get('train_loss', 'N/A')}"
            )
            if idx in flagged_indices:
                logger.warning(
                    f"{round_tag} [FLTrust] Client {idx} flagged — "
                    f"trust score {trust:.4f} ≤ threshold {cfg.FLTRUST_MIN_TRUST_SCORE}. "
                    f"Client's gradient direction opposes benign improvement."
                )

        # Persist trust scores for Upgrade 3 detection log
        trust_record = {
            "round":          server_round,
            "trust_scores":   [round(t, 4) for t in trust_scores],
            "flagged_indices": flagged_indices,
            "timestamp":      time.time(),
        }
        self._trust_history.append(trust_record)
        self._write_trust_log(trust_record)

        # Update server's global model reference for next round
        with torch.no_grad():
            for p, arr in zip(self._global_model.parameters(), aggregated):
                p.copy_(torch.tensor(arr))

        # ── SHA-256 Hash (computed every round — clients verify weights) ──────
        model_hash = hash_model_weights(aggregated)
        print(f"{round_tag} Global Model {model_version_tag} aggregated.")
        print(f"{round_tag} SHA-256 hash: {model_hash}")

        # ── Blockchain Audit Log (FINAL ROUND ONLY) ──────────────────────────
        # Intermediate rounds converge the model; only the final aggregated
        # model is production-ready and gets minted on the blockchain ledger.
        is_final_round = (server_round == self.num_rounds)
        if is_final_round:
            final_version = f"final_v{self._model_version}"
            if self.blockchain is not None:
                try:
                    tx_hash = self.blockchain.log_model_update(
                        model_version=final_version,
                        model_hash=model_hash,
                    )
                    print(f"[BLOCKCHAIN] Final Model {final_version} minted. "
                          f"Hash {model_hash[:12]}… | TX: {str(tx_hash)[:16]}…")
                except Exception as e:
                    logger.warning(f"Blockchain log failed (fallback active): {e}")
            else:
                self._log_hash_local(final_version, model_hash, server_round)

            # Write trusted registry — only for the final converged model
            self._write_trusted_registry(final_version, model_hash)
            model_version_tag = final_version
        else:
            print(f"{round_tag} Intermediate round — hash not minted yet "
                  f"(blockchain mint on round {self.num_rounds} only).")

        # Expose which indices were selected so client_statuses can use it
        # Record in history (for dashboard display)
        # NOTE: selected_indices is now all clients with trust > 0
        selected_indices = [i for i, t in enumerate(trust_scores) if t > cfg.FLTRUST_MIN_TRUST_SCORE]
        self._hash_history.append({
            "round":   server_round,
            "version": model_version_tag,
            "hash":    model_hash,
            "clients_selected": selected_indices,
            "clients_dropped":  [i for i in range(n_received) if i not in selected_indices],
        })

        # Save the aggregated model to disk
        self._save_model(aggregated, model_version_tag)

        return ndarrays_to_parameters(aggregated), {
            "model_version":           model_version_tag,
            "model_hash":              model_hash,
            "krum_selected":           len(selected_indices),      # legacy dashboard compat
            "krum_selected_indices":   selected_indices,
            "krum_dropped":            len(flagged_indices),
            "trust_scores":            [round(t, 4) for t in trust_scores],
            "fltrust_flagged":         flagged_indices,
            "fltrust_trusted_indices": selected_indices,
            "fltrust_flagged_indices": flagged_indices,
        }

    def _log_hash_local(self, version: str, model_hash: str, rnd: int) -> None:
        """Fallback: write hash to local JSONL file if blockchain is unavailable."""
        record = {
            "timestamp": time.time(),
            "round":     rnd,
            "version":   version,
            "hash":      model_hash,
        }
        log_path = Path(cfg.LOGS_DIR) / "model_hashes.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(f"[LOCAL-HASH] {model_hash} written to {log_path}")

    def _write_trusted_registry(self, version: str, model_hash: str) -> None:
        """Write to the trusted hash registry (separate from the ledger).
        verify_chain.py reads this as the ground-truth reference.
        Corrupting the blockchain ledger won't affect this file.
        """
        registry_path = Path(cfg.LOGS_DIR) / "hash_registry.json"
        registry: dict = {}
        if registry_path.exists():
            try:
                registry = json.loads(registry_path.read_text())
            except Exception as e:
                raise RuntimeError(
                    f"FATAL: Hash registry read failed. "
                    f"Wiping registry silently breaks audit trail verification. "
                    f"Original error: {e}"
                )
        registry[version] = model_hash
        registry_path.write_text(json.dumps(registry, indent=2))
        logger.info(f"[REGISTRY] {version} written to trusted registry.")

    def _write_trust_log(self, record: dict) -> None:
        """
        Append per-round trust scoring record to logs/fltrust_trust_log.jsonl.
        Consumed by Upgrade 3 (Byzantine detection dashboard table).
        """
        try:
            log_path = Path(cfg.LOGS_DIR) / "fltrust_trust_log.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            logger.info(
                f"[FLTRUST] Round {record['round']} trust log written — "
                f"flagged={record['flagged_indices']}"
            )
        except Exception as e:
            logger.warning(f"[FLTRUST] Trust log write failed: {e}")

    def _save_model(self, arrays: List[np.ndarray], version_tag: str) -> None:
        """Save the aggregated global model weights to disk."""
        model = AURAModelBundle()
        with torch.no_grad():
            for p, arr in zip(model.parameters(), arrays):
                p.copy_(torch.tensor(arr))
        save_path = Path(cfg.MODELS_DIR) / f"global_model_{version_tag}.pth"
        torch.save(model.state_dict(), save_path)
        logger.info(f"Global model saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Server Launch
# ─────────────────────────────────────────────────────────────────────────────

def start_server(blockchain_module=None) -> None:
    """
    Start the AURA Flower federation server.

    The server blocks until all FL_NUM_ROUNDS are complete, then exits.
    Run in a separate process/thread from the dashboard.

    Straggler timeout is enforced via ServerConfig(round_timeout=...).
    Any client that doesn't respond within round_timeout_sec is treated
    as dropped for that round (Flower gRPC layer handles the socket close).
    """
    strategy = KrumFedAURA(blockchain_module=blockchain_module)

    server_config = fl.server.ServerConfig(
        num_rounds    = cfg.FL_NUM_ROUNDS,
        round_timeout = cfg.FL_ROUND_TIMEOUT_SEC,   # Straggler hard timeout
    )

    print(f"\n{'='*60}")
    print(f"  AURA Federation Server starting on {cfg.FL_SERVER_ADDRESS}")
    print(f"  Rounds: {cfg.FL_NUM_ROUNDS}  |  Timeout: {cfg.FL_ROUND_TIMEOUT_SEC}s")
    print(f"  Strategy: FLTrust (5-client Byzantine-robust aggregation)")
    print(f"{'='*60}\n")

    fl.server.start_server(
        server_address = cfg.FL_SERVER_ADDRESS,
        config         = server_config,
        strategy       = strategy,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simulation Mode (no real gRPC) — for demo without network setup
# ─────────────────────────────────────────────────────────────────────────────

def run_federation_simulation(blockchain_module=None, n_rounds: int = None,
                              active_orgs: list = None) -> List[dict]:
    """
    In-process federation simulation for the hackathon demo.

    Parameters
    ----------
    active_orgs : list of org keys that are ready, e.g. ["hospital","university"].
                  If None, defaults to all three.  Only ready orgs participate.
                  Byzantine client is randomly assigned among them each run.
    """
    from aura.fl_client import create_mock_clients

    if n_rounds is None:
        n_rounds = cfg.FL_NUM_ROUNDS

    if active_orgs is None:
        active_orgs = ["hospital", "bank", "university", "isp", "retail"]

    # Attack is tied to the bank org — only injected if bank is in active_orgs.
    # If bank is offline, all clients are honest (no meaningless FLTrust flag).
    if active_orgs:
        attack_arg = random.randint(0, len(active_orgs) - 1)
        print(f"[SERVER] 🎲 Randomly selected Byzantine client index: {attack_arg} ({active_orgs[attack_arg]})")
    else:
        attack_arg = -1

    from aura.data_loader import CICIDSDataLoader
    _shared_loader = CICIDSDataLoader()
    _shared_scaler = _shared_loader.fit_scaler()

    clients, attack_idx = create_mock_clients(
        n_clients     = len(active_orgs),
        n_samples     = 300,
        org_ids       = active_orgs,
        attack_client = attack_arg,
        shared_scaler = _shared_scaler,
    )
    strategy = KrumFedAURA(blockchain_module=blockchain_module,
                           num_rounds=n_rounds)

    # Initialise with random global model
    global_model = AURAModelBundle()
    global_params = [p.detach().cpu().numpy() for p in global_model.parameters()]

    round_results = []

    for rnd in range(1, n_rounds + 1):
        print(f"\n{'─'*55}")
        print(f"  FEDERATION ROUND {rnd}/{n_rounds}")
        print(f"{'─'*55}")

        fit_results = []
        for client in clients:
            # Build FitIns with current global params
            from flwr.common import FitIns, Config
            fit_ins = FitIns(
                parameters = ndarrays_to_parameters(global_params),
                config     = {"local_epochs": cfg.FL_LOCAL_EPOCHS, "round": rnd},
            )
            fit_res = client.fit(fit_ins)

            # Simulate per-client console output for demo effect
            print(f"  [CLIENT {client.client_id}] weights sent to server ✓")
            fit_results.append((None, fit_res))   # ClientProxy=None in simulation

        # Run server-side FLTrust aggregation
        new_params, metrics = strategy.aggregate_fit(
            server_round = rnd,
            results      = fit_results,
            failures     = [],
        )
        # Build per-client status for dashboard display
        # "Byzantine" = clients FLTrust flagged (low cosine trust vs server root)
        selected_idx = metrics.get("fltrust_trusted_indices", [])
        dropped_idx  = list(metrics.get("fltrust_flagged_indices", []))
        client_statuses = []
        for i, client in enumerate(clients):
            is_selected  = (selected_idx and i in selected_idx)
            is_byzantine = (i in dropped_idx)   # FLTrust-flagged = suspicious
            org_key = active_orgs[i] if i < len(active_orgs) else f"org_{i}"
            client_statuses.append({
                "client_id": client.client_id,
                "org_id":    org_key,
                "network":   cfg.ORG_NETWORK_MAP.get(org_key, "—"),
                "org":       org_key.capitalize(),
                "role":      "Byzantine" if is_byzantine else "Normal",
                "selected":  is_selected if selected_idx else (not is_byzantine),
                "round":     rnd,
            })
        if new_params is not None:
            global_params = parameters_to_ndarrays(new_params)
            model_version = metrics.get('model_version')
            server_hash   = metrics.get('model_hash')
            print(f"\n  [SERVER] Global Model {model_version} aggregated.")
            print(f"  [SERVER] SHA-256 minted on blockchain: {server_hash[:20]}...")
            print(f"  [SERVER] FLTrust trusted {len(selected_idx)} / "
                  f"{len(clients)} clients.")
            print()

            # ── Client-side hash verification ────────────────────────────────────
            # Hash verification only makes sense on the final round, because the
            # blockchain is only minted once (at the end of federation).
            # On intermediate rounds we print the computed hash for auditing only.
            is_final = model_version and model_version.startswith("final_")
            if is_final:
                client_received_hash = hash_model_weights(global_params)

                for client in clients:
                    # Fetch the hash the server minted for this version
                    bc = blockchain_module
                    if bc is not None:
                        on_chain_ok, _ = bc.verify_model(model_version, client_received_hash)
                    else:
                        # No blockchain — compare directly against server hash
                        on_chain_ok = (client_received_hash == server_hash)

                    if on_chain_ok:
                        print(f"  [CLIENT {client.client_id}] "
                              f"Received hash {client_received_hash[:16]}... "
                              f"== Blockchain hash {server_hash[:16]}... "
                              f"→ MATCH. Model deployed.")
                    else:
                        print(f"  [CLIENT {client.client_id}] "
                              f"Received hash {client_received_hash[:16]}... "
                              f"!= Blockchain hash {server_hash[:16]}... "
                              f"→ MISMATCH! Weights tampered in transit. REJECTING model.")
            else:
                print(f"  [CLIENTS] Intermediate round — blockchain not yet minted. "
                      f"Hash {server_hash[:20]}... recorded locally for auditing.")

        round_results.append({"round": rnd, "client_statuses": client_statuses, **metrics})

    print(f"\n{'='*55}")
    print(f"  Federation complete.  {n_rounds} rounds executed.")
    print(f"{'='*55}\n")

    return round_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AURA FL Server (FLTrust)")
    parser.add_argument(
        "--address", default=cfg.FL_SERVER_ADDRESS,
        help="gRPC bind address (default: %(default)s). "
             "Use 0.0.0.0:8080 to accept remote clients."
    )
    parser.add_argument(
        "--rounds", type=int, default=cfg.FL_NUM_ROUNDS,
        help="Number of FL rounds (default: %(default)s)"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run in-process simulation instead of gRPC server (legacy mode)"
    )
    args = parser.parse_args()

    if args.simulate:
        print("=== AURA Federation — In-Process Simulation Mode ===")
        results = run_federation_simulation(n_rounds=args.rounds)
        for r in results:
            print(f"  Round {r['round']}: {r.get('model_version')}  "
                  f"hash={r.get('model_hash', 'N/A')[:18]}…")
        print("✓ Federation simulation complete.")
    else:
        # TRUE NETWORKED MODE — waits for real gRPC client connections
        from aura.blockchain import AURABlockchainLogger
        bc = AURABlockchainLogger()

        strategy = KrumFedAURA(
            blockchain_module = bc,
            num_rounds        = args.rounds,
        )
        server_config = fl.server.ServerConfig(
            num_rounds    = args.rounds,
            round_timeout = cfg.FL_ROUND_TIMEOUT_SEC,
        )

        print(f"\n{'='*62}")
        print(f"  AURA Federation Server — NETWORKED MODE")
        print(f"  Binding on:  {args.address}")
        print(f"  Rounds:      {args.rounds}")
        print(f"  Strategy:    FLTrust (5-client Byzantine-robust aggregation)")
        print(f"  Waiting for {cfg.FL_MIN_AVAILABLE} clients to connect …")
        print(f"{'='*62}\n")

        fl.server.start_server(
            server_address = args.address,
            config         = server_config,
            strategy       = strategy,
        )

FLTrustServerAURA = KrumFedAURA
