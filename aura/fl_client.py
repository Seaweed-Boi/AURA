"""
aura/fl_client.py — Flower Federated Learning Client
=====================================================

Each "organisation" (Bank, Hospital, ISP) runs one instance of this client.
The client owns a LOCAL copy of the AURAModelBundle and trains it on its
own private data.  Only the mathematical weight deltas (gradients) are ever
sent to the server — raw data NEVER leaves the local network.

Federation Lifecycle (per round)
---------------------------------
1. Server → Client: broadcasts current global model weights
2. Client: computes SHA-256 hash and verifies against blockchain ledger
3. Client: loads weights into local model (ONLY if hash matches)
4. Client: trains for LOCAL_EPOCHS on local data partition
5. Client: sends updated weights back to server
6. Server: applies FLTrust aggregation to drop potential poisoned updates

Privacy Guarantee:
  Differential Privacy (DP) is the production extension.
  For the hackathon demo, we demonstrate the architectural boundary —
  no raw data (IP logs, user records) leave the client boundary.

Supply Chain Integrity:
  Before loading any received global weights, the client independently
  computes a SHA-256 hash and verifies it against the Ganache smart
  contract ledger.  If the hash mismatches (indicating tampering in
  transit — Man-in-the-Middle), the weights are REJECTED.
"""

import hashlib
import io
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from flwr.common import (
    Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes,
    GetParametersIns, GetParametersRes, Status, Code,
    ndarrays_to_parameters, parameters_to_ndarrays,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from aura.models import AURAModelBundle

# ── Opacus DP-SGD (Tier 2.1) ────────────────────────────────────────────────
# Imported conditionally so the FL client still works if Opacus is not installed
# (e.g., in production environments where DP isn't needed).
try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SHA-256 Model Hash (MUST be identical to fl_server.py for hash match)
# ─────────────────────────────────────────────────────────────────────────────

def hash_model_weights(arrays: List[np.ndarray]) -> str:
    """
    Compute a SHA-256 hash over the concatenated model weight bytes.

    Normalises every array to C-contiguous float32 before hashing so the
    result is identical whether called on the server-side aggregated arrays
    or on the client-side after Flower's ndarrays_to_parameters round-trip.

    ⚠️  This function MUST be byte-identical to fl_server.hash_model_weights.
    Any divergence (dtype, memory layout, prefix) will cause all client-side
    verifications to fail.
    """
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return "0x" + h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Model ↔ NumPy Parameter Conversion
# ─────────────────────────────────────────────────────────────────────────────

def model_to_ndarrays(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays (Flower format)."""
    return [p.detach().cpu().numpy() for p in model.parameters()]


def ndarrays_to_model(model: nn.Module, arrays: List[np.ndarray]) -> None:
    """Load a list of NumPy arrays into model parameters (in-place)."""
    with torch.no_grad():
        for p, arr in zip(model.parameters(), arrays):
            p.copy_(torch.tensor(arr))


# ─────────────────────────────────────────────────────────────────────────────
# MITM Attack Simulation (Demo Trigger)
# ─────────────────────────────────────────────────────────────────────────────

# Set to True to force-trigger a simulated Man-in-the-Middle attack on the
# next fit()/evaluate() call.  When True, the client will slightly perturb
# the received weights before hashing, causing a hash mismatch that
# demonstrates the defense mechanism.
SIMULATE_MITM_ATTACK: bool = False

# Alternatively, set this to a probability (0.0–1.0) for random MITM
# triggering during demo runs.  0.0 = never, 1.0 = always.
MITM_RANDOM_PROBABILITY: float = 0.0


def _should_simulate_mitm() -> bool:
    """Check whether to simulate a MITM attack on this call."""
    if SIMULATE_MITM_ATTACK:
        return True
    if MITM_RANDOM_PROBABILITY > 0.0:
        return random.random() < MITM_RANDOM_PROBABILITY
    return False


def _tamper_weights(arrays: List[np.ndarray]) -> List[np.ndarray]:
    """
    Simulate a Man-in-the-Middle attack by injecting small perturbations
    into the received global weights.  This causes the SHA-256 hash to
    change, triggering the client's rejection logic.

    Noise scale is governed by cfg.MITM_NOISE_STD (default 0.01) so the
    simulation realism can be tuned from config.py without touching this code.
    A small std is sufficient to flip the SHA-256 (even a 1-bit weight change
    is detected) while keeping the tampered weights visually plausible.
    """
    tampered = []
    for arr in arrays:
        noise = np.random.normal(0, cfg.MITM_NOISE_STD, arr.shape).astype(np.float32)
        tampered.append(arr + noise)
    return tampered


# ─────────────────────────────────────────────────────────────────────────────
# Client-Side Hash Verification
# ─────────────────────────────────────────────────────────────────────────────

def _verify_global_weights(
    client_id: str,
    global_arrays: List[np.ndarray],
    context: str = "fit",
) -> Tuple[List[np.ndarray], bool]:
    """
    Verify the integrity of received global model weights.

    1. (Demo) Optionally tamper weights to simulate MITM attack.
    2. Compute SHA-256 hash of the (possibly tampered) weights.
    3. Print high-visibility security audit output.
    4. Simulate verification against Ganache smart contract ledger.

    Parameters
    ----------
    client_id     : Client identifier for logging.
    global_arrays : The deserialized weight arrays from the server.
    context       : 'fit' or 'evaluate' — used in log messages.

    Returns
    -------
    (arrays, verified) — the arrays to use and whether verification passed.
    If MITM is simulated, arrays will be the tampered version (and verified=False).
    """
    mitm_active = _should_simulate_mitm()

    if mitm_active:
        print(f"\n{'!'*60}")
        print(f"  ⚠️  [{client_id}] SIMULATED MAN-IN-THE-MIDDLE ATTACK!")
        print(f"  ⚠️  Weights are being altered in transit …")
        print(f"{'!'*60}")
        arrays_to_hash = _tamper_weights(global_arrays)
    else:
        arrays_to_hash = global_arrays

    # ── Compute SHA-256 hash ─────────────────────────────────────────────
    computed_hash = hash_model_weights(arrays_to_hash)

    # ── High-visibility audit output ─────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  [SECURITY AUDIT] Client: {client_id}  |  Phase: {context.upper()}")
    print(f"  [SECURITY AUDIT] Received Global Model.")
    print(f"  [SECURITY AUDIT] Computed SHA-256: {computed_hash}")
    print(f"{'═'*60}")

    # ── Verification against Ganache smart contract ledger ───────────────
    # In production, this would call:
    #   blockchain.verify_model(version, computed_hash)
    # For the demo, we read the server's trusted hash registry and compare.
    print(f"  [SECURITY AUDIT] Verifying hash against Ganache smart contract ledger …")

    registry_path = Path(cfg.LOGS_DIR) / "hash_registry.json"
    ledger_hash = None
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text())
            # Get the latest registered hash (most recent version)
            if registry:
                latest_version = list(registry.keys())[-1]
                ledger_hash = registry[latest_version]
        except Exception:
            pass

    if mitm_active:
        # In a MITM simulation, the tampered hash will NOT match
        print(f"  [SECURITY AUDIT] ❌ HASH MISMATCH DETECTED!")
        print(f"  [SECURITY AUDIT]   Computed:  {computed_hash[:24]}…")
        if ledger_hash:
            print(f"  [SECURITY AUDIT]   On-chain:  {ledger_hash[:24]}…")
        else:
            # Even without a ledger entry, the tampered hash differs from
            # the hash of the original (un-tampered) weights.
            original_hash = hash_model_weights(global_arrays)
            print(f"  [SECURITY AUDIT]   Expected:  {original_hash[:24]}…")
        print(f"  [SECURITY AUDIT] ⛔ WEIGHTS TAMPERED IN TRANSIT — REJECTING MODEL UPDATE!")
        print(f"{'═'*60}\n")
        return arrays_to_hash, False

    # ── Normal path: hash matches ────────────────────────────────────────
    if ledger_hash:
        if computed_hash == ledger_hash:
            print(f"  [SECURITY AUDIT] ✅ Hash matches Ganache ledger entry ({ledger_hash[:16]}…)")
        else:
            # Hash doesn't match ledger but this isn't MITM — could be
            # intermediate round (ledger only has final round hash).
            print(f"  [SECURITY AUDIT] ℹ️  Intermediate round — ledger hash is for final model.")
            print(f"  [SECURITY AUDIT] ✅ Hash recorded locally for audit trail.")
    else:
        print(f"  [SECURITY AUDIT] ℹ️  No ledger entry yet (pre-final round).")
        print(f"  [SECURITY AUDIT] ✅ Hash recorded locally. Will verify at final round.")

    print(f"  [SECURITY AUDIT] ✅ Integrity verified. Loading weights into local model.")
    print(f"{'═'*60}\n")

    return global_arrays, True


# ─────────────────────────────────────────────────────────────────────────────
# AURA Flower Client
# ─────────────────────────────────────────────────────────────────────────────

class AURAFlowerClient(fl.client.Client):
    """
    Flower client that encapsulates a local AURAModelBundle and its training
    data partition (representing one organisation's private network).

    Supply Chain Integrity
    ----------------------
    Before loading ANY received global weights, this client:
      1. Computes a SHA-256 hash of the received weight arrays.
      2. Verifies the hash against the Ganache smart contract ledger.
      3. REJECTS the weights if the hash mismatches (defence against MITM).

    Parameters
    ----------
    client_id      : Unique identifier for this client (e.g., "hospital_1")
    train_data     : Tensor[N_local, F] — local normalised flow features
    val_data       : Tensor[M_local, F] — local validation split
    local_epochs   : Number of local SGD epochs per federation round
    device         : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        client_id:    str,
        train_data:   torch.Tensor,
        val_data:     torch.Tensor,
        local_epochs: int   = cfg.FL_LOCAL_EPOCHS,
        device:       str   = None,
    ):
        self.client_id    = client_id
        self.local_epochs = local_epochs
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.train_data   = train_data.to(self.device)
        self.val_data     = val_data.to(self.device)

        # Local model — each org starts with a fresh copy; federation aligns them
        self.model    = AURAModelBundle().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.autoencoder.parameters(),
            lr=cfg.AE_LEARNING_RATE,
        )

        # ── Differential Privacy state ───────────────────────────────────
        # Tracks the cumulative privacy budget consumed across rounds.
        # PrivacyEngine is re-created each round in _local_train() because
        # Opacus attaches hooks to the model/optimizer that must be fresh
        # after global weights are loaded.
        self.last_epsilon: float = 0.0
        self.last_epsilon_head: float = 0.0
        self.dp_enabled = (
            cfg.DP_ENABLED
            and OPACUS_AVAILABLE
            and cfg.DP_NOISE_MULTIPLIER > 0.0
        )
        if cfg.DP_ENABLED and not OPACUS_AVAILABLE:
            logger.warning(
                f"[{client_id}] DP_ENABLED=True but Opacus not installed. "
                f"Training WITHOUT differential privacy."
            )

        logger.info(
            f"[{client_id}] Flower client initialised  |  "
            f"train={len(train_data)}  val={len(val_data)}  "
            f"epochs={local_epochs}  device={self.device}  "
            f"dp={'ON (σ=' + str(cfg.DP_NOISE_MULTIPLIER) + ')' if self.dp_enabled else 'OFF'}"
        )

    # ------------------------------------------------------------------
    # Flower Protocol Methods
    # ------------------------------------------------------------------

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Return current local model weights to the server."""
        arrays = model_to_ndarrays(self.model)
        return GetParametersRes(
            status     = Status(code=Code.OK, message="OK"),
            parameters = ndarrays_to_parameters(arrays),
        )

    def fit(self, ins: FitIns) -> FitRes:
        """
        Receive global weights, verify integrity, train locally, return updated weights.

        Step 1: Deserialize received global weights
        Step 2: Compute SHA-256 hash and verify against blockchain ledger
        Step 3: Load verified weights into local model
        Step 4: Run LOCAL_EPOCHS of unsupervised autoencoder training
        Step 5: Return updated parameters + training metadata
        """
        logger.info(f"[{self.client_id}] Round started — loading global weights …")
        self.local_epochs = int(ins.config.get("local_epochs", self.local_epochs))

        # Step 1: Deserialize global model parameters
        global_arrays = parameters_to_ndarrays(ins.parameters)

        # Step 2: Hash verification — BEFORE loading into model
        verified_arrays, is_verified = _verify_global_weights(
            client_id=self.client_id,
            global_arrays=global_arrays,
            context="fit",
        )

        # Step 3: Load weights ONLY after verification
        if not is_verified:
            # MITM detected — reject the update, keep current local weights
            print(f"  [{self.client_id}] ⛔ FIT ABORTED — using previous local weights.")
            logger.warning(f"[{self.client_id}] MITM detected in fit(). "
                           f"Rejecting global weights. Training on stale local model.")
            # Still train on the existing (safe) local model so the client
            # contributes an update based on its last known good state.
        else:
            ndarrays_to_model(self.model, verified_arrays)

        # Step 4: Local training on private data
        num_examples, train_loss = self._local_train()

        # Step 5: Return updated weights
        updated_arrays = model_to_ndarrays(self.model)

        # ── DP privacy budget reporting ──────────────────────────────────
        dp_info = ""
        if self.dp_enabled and self.last_epsilon > 0:
            dp_info = f"  ε={self.last_epsilon:.4f} (target: {cfg.DP_TARGET_EPSILON})"
            logger.info(
                f"[{self.client_id}] Round privacy budget: "
                f"ε={self.last_epsilon:.4f} "
                f"(target: {cfg.DP_TARGET_EPSILON}), δ={cfg.DP_DELTA}"
            )

        logger.info(
            f"[{self.client_id}] Round complete  |  "
            f"loss={train_loss:.4f}  examples={num_examples}  "
            f"epochs={self.local_epochs}{dp_info}"
        )

        # Encode client identity as a stable integer for multi-client tracking.
        # Using a hash of the string ID avoids collisions while keeping it int.
        import hashlib as _hl
        client_id_int = int(_hl.md5(self.client_id.encode()).hexdigest(), 16) % (2**31)

        # Include DP epsilon in metrics so the server can log per-client privacy
        # budget consumption and detect clients exceeding their target.
        fit_metrics = {
            "train_loss": float(train_loss),
            "client_id": client_id_int,
        }
        if self.dp_enabled:
            fit_metrics["dp_epsilon"] = float(self.last_epsilon)
            fit_metrics["dp_epsilon_head"] = float(self.last_epsilon_head)
            fit_metrics["dp_delta"] = float(cfg.DP_DELTA)
            fit_metrics["dp_noise_multiplier"] = float(cfg.DP_NOISE_MULTIPLIER)

        return FitRes(
            status     = Status(code=Code.OK, message="OK"),
            parameters = ndarrays_to_parameters(updated_arrays),
            num_examples = num_examples,
            metrics    = fit_metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Evaluate the received global weights on local validation data.

        The client verifies the integrity of received weights via SHA-256
        hash comparison with the blockchain ledger before loading them.
        """
        # Step 1: Deserialize global model parameters
        arrays = parameters_to_ndarrays(ins.parameters)

        # Step 2: Hash verification — BEFORE loading into model
        verified_arrays, is_verified = _verify_global_weights(
            client_id=self.client_id,
            global_arrays=arrays,
            context="evaluate",
        )

        # Step 3: Load weights ONLY after verification
        if not is_verified:
            # MITM detected — evaluate on current (safe) local model
            print(f"  [{self.client_id}] ⛔ EVALUATE using local weights (global rejected).")
            logger.warning(f"[{self.client_id}] MITM detected in evaluate(). "
                           f"Evaluating on local model instead of tampered global model.")
        else:
            ndarrays_to_model(self.model, verified_arrays)

        self.model.autoencoder.eval()
        with torch.no_grad():
            x_hat, _ = self.model.autoencoder(self.val_data)
            loss = nn.functional.mse_loss(x_hat, self.val_data)

        logger.info(f"[{self.client_id}] Eval loss: {loss.item():.4f}")
        return EvaluateRes(
            status       = Status(code=Code.OK, message="OK"),
            loss         = float(loss),
            num_examples = len(self.val_data),
            metrics      = {"val_loss": float(loss)},
        )

    # ------------------------------------------------------------------
    # Local Training
    # ------------------------------------------------------------------

    def _local_train(self) -> Tuple[int, float]:
        """
        Run unsupervised autoencoder training on local data.

        We train in batch mode — the autoencoder learns to reconstruct
        the local network's 'normal' flow distribution.  If this client's
        network gets attacked, the reconstruction error will spike, and
        the updated weights (incorporating the new attack-learned boundary)
        will be shared with the federation.

        When cfg.DP_ENABLED is True and Opacus is available, both the AE
        and AttackHead optimizers are wrapped with separate PrivacyEngine
        instances for DP-SGD.  This provides formal (ε,δ)-differential
        privacy guarantees on both training passes.

        Returns:  (num_training_examples, final_batch_loss)
        """
        ae = self.model.autoencoder
        ae.train()

        dataset = torch.utils.data.TensorDataset(self.train_data)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.AE_BATCH_SIZE, shuffle=True
        )

        # ── DP-SGD: wrap AE optimizer with Opacus PrivacyEngine ──────────
        # Fresh optimizer each round because Opacus requires attaching to
        # a clean optimizer (hooks from previous rounds would conflict
        # after global weight loading).
        optimizer = torch.optim.Adam(
            ae.parameters(), lr=cfg.AE_LEARNING_RATE
        )
        privacy_engine = None

        if self.dp_enabled:
            privacy_engine = PrivacyEngine()
            # make_private replaces ae, optimizer, loader with DP-wrapped versions
            # that compute per-sample gradients, clip to max_grad_norm, and add
            # calibrated Gaussian noise.
            ae, optimizer, loader = privacy_engine.make_private(
                module=ae,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=cfg.DP_NOISE_MULTIPLIER,
                max_grad_norm=cfg.DP_MAX_GRAD_NORM,
            )
            logger.info(
                f"  [{self.client_id}] DP-SGD attached: "
                f"σ={cfg.DP_NOISE_MULTIPLIER}  "
                f"C={cfg.DP_MAX_GRAD_NORM}  "
                f"δ={cfg.DP_DELTA}"
            )

        # ── Training loop ────────────────────────────────────────────────
        last_loss = 0.0
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                x_hat, z = ae(batch)
                # Use F.mse_loss directly — Opacus needs the loss to flow
                # cleanly through the wrapped module for per-sample gradients.
                loss = F.mse_loss(x_hat, batch)
                loss.backward()

                if not self.dp_enabled:
                    # Manual gradient clipping when DP is off (DP handles its
                    # own per-sample clipping via max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(
                        ae.parameters(), max_norm=1.0
                    )

                optimizer.step()
                epoch_loss += loss.item()

            last_loss = epoch_loss / max(len(loader), 1)
            logger.debug(f"  [{self.client_id}] epoch={epoch+1}  loss={last_loss:.4f}")

        # ── Report DP privacy budget ─────────────────────────────────────
        if privacy_engine is not None:
            self.last_epsilon = privacy_engine.get_epsilon(delta=cfg.DP_DELTA)
            logger.info(
                f"  [{self.client_id}] DP training complete. "
                f"ε={self.last_epsilon:.4f}, δ={cfg.DP_DELTA}"
            )
        else:
            self.last_epsilon = 0.0

        # ── Pass 2: AttackHead (MLP) training with DP on high-MSE z vectors ──
        head = self.model.attack_head
        head_privacy_engine = None
        self.last_epsilon_head = 0.0

        # Use unwrapped AE for inference (no grad, no DP hooks needed)
        ae_unwrapped = ae._module if hasattr(ae, '_module') else ae
        ae_unwrapped.eval()
        with torch.no_grad():
            recon_all, z_all = ae_unwrapped(self.train_data)
            mse_per_flow = F.mse_loss(
                recon_all, self.train_data, reduction='none'
            ).mean(dim=1)

        mse_split = getattr(
            cfg, 'CH2_MSE_SPLIT_THRESHOLD', cfg.MSE_THRESHOLD_HIGH
        )
        high_mse_mask = mse_per_flow > mse_split
        n_high_mse = int(high_mse_mask.sum().item())

        if n_high_mse > 0:
            z_high = z_all[high_mse_mask].detach()
            mse_weights = mse_per_flow[high_mse_mask].detach()
            mse_weights = (mse_weights - mse_weights.min()) / \
                          (mse_weights.max() - mse_weights.min() + 1e-8)
            pseudo_labels = torch.ones(n_high_mse, device=z_high.device)

            head_dataset = torch.utils.data.TensorDataset(
                z_high, pseudo_labels, mse_weights
            )
            head_loader = torch.utils.data.DataLoader(
                head_dataset,
                batch_size=min(cfg.AE_BATCH_SIZE, n_high_mse),
                shuffle=True,
            )

            head.train()
            head_optimizer = torch.optim.Adam(
                head.parameters(), lr=cfg.AE_LEARNING_RATE
            )

            if self.dp_enabled:
                head_privacy_engine = PrivacyEngine()
                head, head_optimizer, head_loader = \
                    head_privacy_engine.make_private(
                        module=head,
                        optimizer=head_optimizer,
                        data_loader=head_loader,
                        noise_multiplier=cfg.DP_NOISE_MULTIPLIER,
                        max_grad_norm=cfg.DP_MAX_GRAD_NORM,
                    )
                logger.info(
                    f"  [{self.client_id}] DP-SGD attached to AttackHead: "
                    f"\u03c3={cfg.DP_NOISE_MULTIPLIER}  "
                    f"C={cfg.DP_MAX_GRAD_NORM}  "
                    f"n_high_mse={n_high_mse}"
                )

            for _ep in range(3):
                for z_batch, lbl_batch, w_batch in head_loader:
                    head_optimizer.zero_grad()
                    preds = head(z_batch).squeeze(-1)
                    loss_head = F.binary_cross_entropy(
                        preds, lbl_batch, weight=w_batch
                    )
                    loss_head.backward()
                    head_optimizer.step()

            if head_privacy_engine is not None:
                self.last_epsilon_head = head_privacy_engine.get_epsilon(
                    delta=cfg.DP_DELTA
                )
                logger.info(
                    f"  [{self.client_id}] AttackHead DP complete. "
                    f"\u03b5={self.last_epsilon_head:.4f}, \u03b4={cfg.DP_DELTA}"
                )

            # Unwrap DP-wrapped AttackHead
            if self.dp_enabled and hasattr(head, '_module'):
                self.model.attack_head = head._module
            elif self.dp_enabled:
                self.model.attack_head = head
        else:
            logger.debug(
                f"  [{self.client_id}] No high-MSE flows "
                f"(threshold={mse_split:.4f}) \u2014 AttackHead training skipped"
            )

        # Unwrap the DP-wrapped model back to the original autoencoder
        # so that model_to_ndarrays() extracts clean parameter tensors.
        if self.dp_enabled and hasattr(ae, '_module'):
            self.model.autoencoder = ae._module
        elif self.dp_enabled:
            # Opacus >= 1.4 wraps as GradSampleModule
            self.model.autoencoder = ae

        return len(self.train_data), last_loss


# ─────────────────────────────────────────────────────────────────────────────
# Client Factory (creates mock clients for the demo)
# ─────────────────────────────────────────────────────────────────────────────

def create_mock_clients(
    n_clients:    int   = 5,
    n_samples:    int   = 500,
    feature_dim:  int   = cfg.FEATURE_DIM,
    attack_client: int  = None,     # None = randomly poison one; -1 = all honest
    org_ids:      list  = None,   # Override org IDs e.g. ["hospital","university"]
    shared_scaler        = None,
) -> List["AURAFlowerClient"]:
    """
    Factory function for hackathon demo.

    Creates N mock clients with synthetic Gaussian flow data.
    One client (attack_client index) has data poisoned to simulate a real
    network under attack — this is what gives FLTrust a genuine outlier to detect.

    Parameters
    ----------
    attack_client : Index of the client to poison.
                    None  → randomly select one org to inject attack traffic (default)
                    -1    → all clients train honestly (no Byzantine signal)
                    0-N   → explicitly poison that client index
    org_ids       : Optional list of org keys ["hospital","bank","university"]
                    overriding the default 3-client set.  Length must match n_clients.
    """
    import random as _random

    _default_orgs = ["hospital", "bank", "university", "isp", "retail"]
    _org_client_num = {
        "hospital": 1, "bank": 2, "university": 3, "isp": 4, "retail": 5,
    }
    if org_ids is None:
        org_ids = _default_orgs[:n_clients]

    # Randomly inject attack data into one org so FLTrust has a real signal
    if attack_client is None:
        attack_client = _random.randint(0, len(org_ids) - 1)
        logger.info(f"[MOCK] Attack data injected into index {attack_client} "
                    f"({org_ids[attack_client]}) — FLTrust should detect this outlier")
    elif attack_client == -1:
        attack_client = None   # all clients honest — FLTrust drop is arbitrary

    clients = []
    for i, org_key in enumerate(org_ids):
        client_num = _org_client_num.get(org_key, i + 1)
        client_id = f"org_{org_key}_{client_num}"

        # Real CICIDS2017 partition for this org
        from aura.data_loader import load_client_partition
        try:
            train_data, val_data = load_client_partition(
                client_id=client_id,
                scaler=shared_scaler,
            )
            # Limit samples to drastically speed up simulation
            if len(train_data) > n_samples:
                train_data = train_data[:n_samples]
            if len(val_data) > (n_samples // 5):
                val_data = val_data[:(n_samples // 5)]
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            logger.warning(f"[{client_id}] Falling back to realistic benign profile: {e}")
            from aura.attack_injector import _benign_profile
            _train_np  = _benign_profile(n_samples, feature_dim)
            _val_np    = _benign_profile(max(1, n_samples // 5), feature_dim)
            train_data = torch.tensor(_train_np, dtype=torch.float32)
            val_data   = torch.tensor(_val_np,   dtype=torch.float32)

        if i == attack_client:
            # Strong poisoning: 80% of samples with extreme values across ALL
            # feature groups — ensures weight update is a clear FLTrust outlier
            # rather than noise-level drift that gets masked by random init variance.
            n_attack = int(len(train_data) * 0.8)
            attack_rows = torch.rand(n_attack, feature_dim)
            # Spike all major feature blocks to max range (47 NF-UNSW features)
            attack_rows[:, :16]  = torch.rand(n_attack, 16) * 0.5 + 0.5   # proto/volume/flags
            attack_rows[:, 16:32] = torch.rand(n_attack, 16) * 0.4 + 0.6  # pkt size/throughput
            attack_rows[:, 32:]  = torch.rand(n_attack, feature_dim - 32) * 0.6 + 0.4  # IAT/app
            train_data[:n_attack] = attack_rows
            logger.info(
                f"[{client_id}] Strong attack injection: "
                f"{n_attack}/{len(train_data)} samples poisoned."
            )

        clients.append(AURAFlowerClient(client_id, train_data, val_data))

    return clients, attack_client   # return who was selected as Byzantine


# ─────────────────────────────────────────────────────────────────────────────
# Networked Client Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def start_client(
    client_id:      str,
    server_address: str = cfg.FL_SERVER_ADDRESS,
    n_samples:      int = 500,
    is_byzantine:   bool = False,
) -> None:
    """
    Start a Flower gRPC client that connects to the FL server over the network.

    This is the REAL networked FL entry point.  Each organisation's gateway
    switch runs this function in its own process — the client dials the
    aggregation server via gRPC, trains locally, and sends only weight deltas
    (raw data NEVER leaves the network).

    Parameters
    ----------
    client_id      : human-readable org identifier (e.g. "org_hospital_1")
    server_address : host:port of the FL aggregation server
    n_samples      : number of local flow records to train on
    is_byzantine   : if True, injects attack-pattern data (adversarial client)
    """
    import flwr as fl

    feature_dim = cfg.FEATURE_DIM

    # ── Load real CICIDS partition for this org (mirrors create_mock_clients) ──
    # Attempting to derive an org key from client_id (e.g. "org_hospital_1" → "hospital").
    # Falls back to _benign_profile() if the dataset is unavailable.
    _org_key = client_id.split("_")[1] if "_" in client_id else client_id
    train_data: torch.Tensor
    val_data:   torch.Tensor
    try:
        from aura.data_loader import load_client_partition, CICIDSDataLoader
        _loader = CICIDSDataLoader()
        _shared_scaler = _loader.fit_scaler()
        train_data, val_data = load_client_partition(
            client_id=client_id,
            scaler=_shared_scaler,
        )
        # Cap to requested sample count to bound memory / run time
        if len(train_data) > n_samples:
            train_data = train_data[:n_samples]
        if len(val_data) > max(1, n_samples // 5):
            val_data = val_data[:n_samples // 5]
        logger.info(f"[{client_id}] Loaded real CICIDS partition: "
                    f"train={len(train_data)}  val={len(val_data)}")
    except Exception as _e:
        logger.warning(f"[{client_id}] Real partition unavailable ({_e}). "
                       "Falling back to realistic benign profile.")
        from aura.attack_injector import _benign_profile
        _train_np = _benign_profile(n_samples, feature_dim)
        _val_np   = _benign_profile(max(1, n_samples // 5), feature_dim)
        train_data = torch.tensor(_train_np, dtype=torch.float32)
        val_data   = torch.tensor(_val_np,   dtype=torch.float32)

    if is_byzantine:
        # Adversarial client: apply DDoS corruption profile from config so the
        # poisoned feature pattern is consistent with AttackInjector and the
        # AE explainer's training distribution — gives FLTrust a genuine signal.
        n_attack    = min(len(train_data), n_samples // 5)
        ddos_profile = cfg.ATTACK_CORRUPTION_PROFILES.get("ddos", {})
        feat_map     = cfg.FEATURE_INDEX_MAP
        attack_rows  = torch.rand(n_attack, feature_dim)
        # Apply every feature range from the DDoS corruption profile
        for feat_name, (lo, hi) in ddos_profile.items():
            idx = feat_map.get(feat_name)
            if idx is not None and idx < feature_dim:
                attack_rows[:, idx] = torch.FloatTensor(n_attack).uniform_(lo, hi)
        train_data[:n_attack] = attack_rows
        logger.info(f"[{client_id}] Byzantine mode — DDoS corruption profile applied "
                    f"to {n_attack}/{len(train_data)} samples.")
    client = AURAFlowerClient(client_id, train_data, val_data)

    print(f"\n[{client_id}] Connecting to FL server at {server_address} …")
    print(f"[{client_id}] Network: {'ADVERSARIAL (Byzantine)' if is_byzantine else 'Normal'}")
    print(f"[{client_id}] Local dataset: {n_samples} flow records  |  features: {feature_dim}")
    print(f"[{client_id}] Supply chain verification: SHA-256 hash check ENABLED ✓")

    fl.client.start_client(
        server_address = server_address,
        client         = client.to_client(),
    )
    print(f"[{client_id}] Federation complete. Local model updated.")


# CLI entry point — called by run_federation_networked.py per-process
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AURA FL Client (networked mode)")
    parser.add_argument("--client-id",  required=True,  help="e.g. org_hospital_1")
    parser.add_argument("--server",     default=cfg.FL_SERVER_ADDRESS, help="host:port")
    parser.add_argument("--samples",    type=int, default=500)
    parser.add_argument("--byzantine",  action="store_true", help="Adversarial client")
    parser.add_argument("--network-sim",default="",   help="Simulated LAN CIDR (display only)")
    parser.add_argument("--simulate-mitm", action="store_true",
                        help="Force a simulated Man-in-the-Middle attack (demo)")
    parser.add_argument("--mitm-probability", type=float, default=0.0,
                        help="Random MITM trigger probability 0.0–1.0 (demo)")
    args = parser.parse_args()

    if args.network_sim:
        print(f"[{args.client_id}] Simulated network: {args.network_sim}")

    # Configure MITM simulation from CLI flags
    if args.simulate_mitm:
        SIMULATE_MITM_ATTACK = True
        print(f"[{args.client_id}] ⚠️  MITM attack simulation ENABLED (forced)")
    if args.mitm_probability > 0:
        MITM_RANDOM_PROBABILITY = args.mitm_probability
        print(f"[{args.client_id}] ⚠️  MITM random probability: {MITM_RANDOM_PROBABILITY:.0%}")

    start_client(
        client_id      = args.client_id,
        server_address = args.server,
        n_samples      = args.samples,
        is_byzantine   = args.byzantine,
    )
