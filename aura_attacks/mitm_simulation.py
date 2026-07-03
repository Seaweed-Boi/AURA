"""
mitm_simulation.py  —  MITM attack simulation for AURA's FL channel
====================================================================
Target  : AURA's FLTrust aggregation layer (fl_server.py)
Dataset : NF-UNSW-NB15-v3

HOW TO RUN (standalone, no server needed):
    python aura_attacks/mitm_simulation.py

What this models
----------------
A compromised relay sitting between fl_client.py and fl_server.py on
the gradient/weight-update channel.  Two distinct attacks:

  EAVESDROP  — relay copies every client gradient before forwarding it,
               then feeds copies into gradient_inversion_attack.py to
               reconstruct raw flow features.  SHA-256 hash verification
               in fl_client.py stops *tampered* weights from being
               loaded, but does nothing to stop *reading* legitimate
               ones in transit.

  TAMPER     — relay rewrites a fraction of client updates (sign-flip /
               scale) before they reach the aggregator, simulating
               Byzantine clients without needing --byzantine fl_client
               instances.  FLTrust should assign near-zero trust scores
               to tampered updates; FedAvg has no defence.

Running the real federated MITM
--------------------------------
Use fl_client.py's built-in flag (confirmed in your repo):

  Terminal 1 (server):
    python aura/fl_server.py

  Terminals 2-4 (honest clients):
    python aura/fl_client.py --client-id org_hospital_1 --network-sim 192.168.1.0/24
    python aura/fl_client.py --client-id org_university_1 --network-sim 172.16.1.0/24
    python aura/fl_client.py --client-id org_isp_1 --network-sim 10.10.0.0/24
    python aura/fl_client.py --client-id org_retail_1 --network-sim 172.31.0.0/24

  Terminal 5 (MITM attacker client — 100% hit rate):
    python aura/fl_client.py --client-id org_bank_1 --network-sim 10.0.1.0/24 \\
        --simulate-mitm --mitm-probability 1.0
"""

import sys
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

AURA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AURA_ROOT))

FEATURE_DIM  = 47
ENCODER_DIMS = [32, 24]
LATENT_DIM   = 16
DECODER_DIMS = [24, 32]
N_CLIENTS    = 5


class _StandaloneAE(nn.Module):
    def __init__(self):
        super().__init__()
        enc_dims = [FEATURE_DIM] + ENCODER_DIMS + [LATENT_DIM]
        enc = []
        for i in range(len(enc_dims)-1):
            enc += [nn.Linear(enc_dims[i], enc_dims[i+1]), nn.ReLU()]
        self.encoder = nn.Sequential(*enc)
        dec_dims = [LATENT_DIM] + DECODER_DIMS + [FEATURE_DIM]
        dec = []
        for i in range(len(dec_dims)-1):
            dec.append(nn.Linear(dec_dims[i], dec_dims[i+1]))
            if i < len(dec_dims)-2:
                dec.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _get_grad(model, x):
    model.zero_grad()
    nn.MSELoss()(model(x), x).backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def _flat(grads):
    return torch.cat([g.flatten() for g in grads])


# ─────────────────────────────────────────────────────────────────────────────
# Aggregators
# ─────────────────────────────────────────────────────────────────────────────

def fedavg(client_grads):
    n = len(client_grads)
    agg = [torch.zeros_like(g) for g in client_grads[0]]
    for grads in client_grads:
        for i, g in enumerate(grads):
            agg[i] += g / n
    return agg


def fltrust(client_grads, root_grad):
    """
    FLTrust (Cao et al. 2020) as implemented in AURA's fl_server.py:
    weight each client gradient by its cosine similarity to the server's
    root-dataset gradient, clip negatives (ReLU), normalise magnitude to
    the root norm before weighting.
    """
    root_flat = _flat(root_grad)
    root_norm = root_flat.norm() + 1e-12

    weights, normed = [], []
    for grads in client_grads:
        g_flat = _flat(grads)
        cos    = F.cosine_similarity(g_flat.unsqueeze(0),
                                     root_flat.unsqueeze(0)).item()
        trust  = max(cos, 0.0)          # ReLU — matches FLTRUST_MIN_TRUST_SCORE=0.0
        weights.append(trust)
        g_norm = g_flat.norm() + 1e-12
        normed.append([g * (root_norm / g_norm) for g in grads])

    total = sum(weights) + 1e-12
    agg   = [torch.zeros_like(g) for g in client_grads[0]]
    for w, grads in zip(weights, normed):
        for i, g in enumerate(grads):
            agg[i] += (w / total) * g
    return agg, weights


def invert_gradient(model, true_grad, steps=200):
    dummy = torch.randn(1, FEATURE_DIM, requires_grad=True)
    opt   = torch.optim.Adam([dummy], lr=0.1)
    for _ in range(steps):
        def closure():
            opt.zero_grad(); model.zero_grad()
            dg = torch.autograd.grad(
                nn.MSELoss()(model(dummy), dummy),
                list(model.parameters()), create_graph=True)
            diff = sum(((a-b)**2).sum() for a,b in zip(dg, true_grad))
            diff.backward()
            return diff
        opt.step(closure)
    return dummy.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    global_model = _StandaloneAE()

    # Simulate 5 org clients (matches FL_MIN_CLIENTS=5 in config.py)
    org_names   = ["hospital", "bank", "university", "isp", "retail"]
    client_data = [
        torch.randn(64, FEATURE_DIM, generator=torch.Generator().manual_seed(10+i))
        for i in range(N_CLIENTS)
    ]
    client_grads = [_get_grad(global_model, d) for d in client_data]

    # Server root dataset (matches FLTRUST_ROOT_SAMPLES=200 in config.py)
    root_data = torch.randn(200, FEATURE_DIM, generator=torch.Generator().manual_seed(999))
    root_grad = _get_grad(global_model, root_data)

    # ── MODE 1: EAVESDROP ────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  AURA MITM — MODE 1: EAVESDROP (gradient inversion)")
    print("="*65)
    print("  A relay copies each client's gradient before forwarding it.")
    print("  SHA-256 hash verification in fl_client.py does NOT stop this")
    print("  (it only stops loading *tampered* weights, not *reading* ones).\n")

    for i, (name, grads, data) in enumerate(zip(org_names, client_grads, client_data)):
        print(f"  Inverting org_{name} gradient …")
        attack_model = copy.deepcopy(global_model)
        recon = invert_gradient(attack_model, [g.clone() for g in grads], steps=150)
        mse = ((recon - data[:1])**2).mean().item()
        cos = F.cosine_similarity(recon, data[:1], dim=-1).mean().item()
        print(f"    MSE={mse:.4f}  cosine_sim={cos:.4f}  "
              f"{'❌ LEAKED' if cos > 0.7 else '✅ protected'}")

    # ── MODE 2: TAMPER ───────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  AURA MITM — MODE 2: TAMPER (Byzantine robustness test)")
    print("="*65)
    print("  Relay rewrites 2/5 client updates (scale × -8.0) before")
    print("  they reach the aggregator — simulating --byzantine fl_client.\n")

    import random
    random.seed(7)
    tampered_grads = []
    tamper_log     = []
    for i, (name, grads) in enumerate(zip(org_names, client_grads)):
        if i < 2:   # tamper first 2 clients
            tg = [g * -8.0 for g in grads]
            tampered_grads.append(tg)
            tamper_log.append(f"  ⚡ org_{name}  → TAMPERED (scale × -8.0)")
        else:
            tampered_grads.append([g.clone() for g in grads])
            tamper_log.append(f"  ✓  org_{name}  → forwarded honestly")

    for line in tamper_log:
        print(line)

    clean_agg   = fedavg(client_grads)
    fedavg_agg  = fedavg(tampered_grads)
    fltrust_agg, trust_scores = fltrust(tampered_grads, root_grad)

    def drift(a, b):
        return (_flat(a) - _flat(b)).norm().item()

    print(f"\n  FLTrust trust scores per client:")
    for name, w in zip(org_names, trust_scores):
        bar  = "█" * int(w * 20)
        flag = "  ← TAMPERED, correctly down-weighted" if trust_scores.index(w) < 2 and w < 0.1 else ""
        print(f"    org_{name:12s}: {w:.4f}  {bar}{flag}")

    print(f"\n  Aggregate drift from clean baseline:")
    print(f"    FedAvg  : {drift(fedavg_agg,  clean_agg):.4f}  ← no defence")
    print(f"    FLTrust : {drift(fltrust_agg, clean_agg):.4f}  ← AURA's defence")

    ratio = drift(fedavg_agg, clean_agg) / (drift(fltrust_agg, clean_agg) + 1e-9)
    print(f"\n  FLTrust reduced drift by {ratio:.1f}× vs FedAvg")
    print()
    print("  To test the real federation, run:")
    print("    Terminal 1 : python aura/fl_server.py")
    print("    Terminals 2-5 : python aura/fl_client.py --client-id org_<name>_1")
    print("    MITM client   : python aura/fl_client.py --client-id org_bank_1 \\")
    print("                        --simulate-mitm --mitm-probability 1.0")
    print("="*65 + "\n")