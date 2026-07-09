"""
mia_attack.py  —  Membership Inference Attack against AURA's FL pipeline
=========================================================================
Target  : AURA FlowAutoencoder  47→[32,24]→16→[24,32]→47
Dataset : NF-UNSW-NB15-v3  (real client partitions)

HOW TO RUN (from your AURA root):
    python aura_attacks/mia_attack.py                  # all 5 org clients
    python aura_attacks/mia_attack.py --client hospital # single client
    python aura_attacks/mia_attack.py --n-samples 500   # more samples

Phase-4 relevance
-----------------
Tune Opacus DP-SGD epsilon by watching AUC drop as noise increases.
Target: AUC ≤ 0.55 (near-random guessing = model leaks nothing).

Data sourcing
-------------
  Members    = real training rows from load_client_partition (data the
               model trained on during FL rounds).
  Non-members = real validation rows from load_client_partition (data
                the model has NEVER seen — held-out 20% split).

Two attack variants:
  1. Threshold attack   — lower recon error => predict "member"
  2. Shadow-model attack (Shokri et al. 2017)
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

AURA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AURA_ROOT))

FEATURE_DIM  = 47
ENCODER_DIMS = [32, 24]
LATENT_DIM   = 16
DECODER_DIMS = [24, 32]


class _StandaloneAE(nn.Module):
    """Mirrors AURA's FlowAutoencoder exactly."""
    def __init__(self):
        super().__init__()
        enc_dims = [FEATURE_DIM] + ENCODER_DIMS + [LATENT_DIM]
        enc = []
        for i in range(len(enc_dims) - 1):
            enc += [nn.Linear(enc_dims[i], enc_dims[i+1]), nn.ReLU()]
        self.encoder = nn.Sequential(*enc)

        dec_dims = [LATENT_DIM] + DECODER_DIMS + [FEATURE_DIM]
        dec = []
        for i in range(len(dec_dims) - 1):
            dec.append(nn.Linear(dec_dims[i], dec_dims[i+1]))
            if i < len(dec_dims) - 2:
                dec.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def recon_error(self, x):
        with torch.no_grad():
            return ((self.forward(x) - x) ** 2).mean(dim=-1)


def _load_real_model():
    try:
        from aura.models import FlowAutoencoder
        path = AURA_ROOT / "saved_models" / "autoencoder_best.pth"
        m = FlowAutoencoder()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        print(f"[MIA] Loaded: {path}")
        # Wrap real model to add recon_error if it doesn't have one
        if not hasattr(m, "recon_error"):
            def recon_error(x):
                with torch.no_grad():
                    out = m(x)
                    recon = out[0] if isinstance(out, tuple) else out
                    return ((recon - x) ** 2).mean(dim=-1)
            m.recon_error = recon_error
        return m
    except Exception as e:
        print(f"[MIA] Could not load real checkpoint ({e}). Using random weights.")
        m = _StandaloneAE(); m.eval(); return m


def _train_shadow(data, epochs=30):
    m = _StandaloneAE()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    fn = nn.MSELoss()
    m.train()
    for _ in range(epochs):
        opt.zero_grad(); fn(m(data), data).backward(); opt.step()
    m.eval()
    return m


def threshold_mia(model, member, nonmember):
    scores = torch.cat([model.recon_error(member),
                        model.recon_error(nonmember)]).numpy()
    labels = [1]*len(member) + [0]*len(nonmember)
    auc    = roc_auc_score(labels, -scores)
    preds  = (scores < scores.mean()).astype(int)
    return {"auc": auc, "accuracy": accuracy_score(labels, preds),
            "threshold": float(scores.mean())}


def shadow_model_mia(victim, member, nonmember, n_shadows=6, n_per=200):
    print(f"  Training {n_shadows} shadow AEs …")
    X, y = [], []
    for i in range(n_shadows):
        g1 = torch.Generator().manual_seed(200+i)
        g2 = torch.Generator().manual_seed(800+i)
        tr  = torch.randn(n_per, FEATURE_DIM, generator=g1)
        hld = torch.randn(n_per, FEATURE_DIM, generator=g2)
        sh  = _train_shadow(tr)
        X += [sh.recon_error(tr).unsqueeze(1),
              sh.recon_error(hld).unsqueeze(1)]
        y += [1]*n_per + [0]*n_per

    clf = LogisticRegression()
    clf.fit(torch.cat(X).numpy(), y)

    X_te = torch.cat([victim.recon_error(member).unsqueeze(1),
                      victim.recon_error(nonmember).unsqueeze(1)]).numpy()
    y_te = [1]*len(member) + [0]*len(nonmember)
    return {"auc":      roc_auc_score(y_te, clf.predict_proba(X_te)[:,1]),
            "accuracy": accuracy_score(y_te, clf.predict(X_te))}


# ─────────────────────────────────────────────────────────────────────────────
# Real-data loader helper
# ─────────────────────────────────────────────────────────────────────────────

def _load_real_client_data(client_key: str, n_samples: int):
    """
    Load real NF-UNSW-NB15-v3 data for one FL client.

    Returns
    -------
    (member, nonmember) : tuple[Tensor, Tensor]
        member    = rows from the client's TRAINING split (model saw these)
        nonmember = rows from the client's VALIDATION split (model never saw these)
    Both tensors are shaped [n_samples, 47].
    """
    from aura.data_loader import load_client_partition

    client_id = f"org_{client_key}_1"
    X_train, X_val = load_client_partition(client_id)

    # Clamp n_samples to the smaller of the two splits so we always
    # have balanced member vs non-member sets for a fair AUC.
    avail = min(len(X_train), len(X_val), n_samples)
    if avail < n_samples:
        print(f"  [INFO] Requested {n_samples} samples but {client_key} has "
              f"train={len(X_train)}, val={len(X_val)}. Using {avail} per class.")

    # Deterministic shuffle so results are reproducible across runs
    rng = torch.Generator().manual_seed(42)
    train_perm = torch.randperm(len(X_train), generator=rng)[:avail]
    val_perm   = torch.randperm(len(X_val),   generator=rng)[:avail]

    member    = X_train[train_perm]   # data the model DID train on
    nonmember = X_val[val_perm]       # data the model has NEVER seen
    return member, nonmember


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ALL_ORGS = ["hospital", "bank", "university", "isp", "retail"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Membership Inference Attack against AURA's FL pipeline "
                    "using real NF-UNSW-NB15-v3 client partitions."
    )
    ap.add_argument(
        "--client", type=str, default=None,
        choices=ALL_ORGS,
        help="Run MIA against a single org client. Default: all 5 orgs."
    )
    ap.add_argument("--n-samples", type=int, default=200,
                    help="Number of member / non-member samples per client.")
    args = ap.parse_args()

    torch.manual_seed(0)
    print("\n" + "="*60)
    print("  AURA — Membership Inference Attack (Privacy Evaluation)")
    print("  Data: real NF-UNSW-NB15-v3 client partitions")
    print("="*60)

    # Always use the real trained checkpoint
    victim = _load_real_model()

    clients = [args.client] if args.client else ALL_ORGS
    all_results = []

    for org in clients:
        print(f"\n{'─'*60}")
        print(f"  Client: org_{org}")
        print(f"{'─'*60}")

        member, nonmember = _load_real_client_data(org, args.n_samples)
        print(f"  Members (training rows):     {member.shape[0]}")
        print(f"  Non-members (held-out rows): {nonmember.shape[0]}")

        print("\n  [Attack 1 — Threshold MIA]")
        t = threshold_mia(victim, member, nonmember)
        print(f"    AUC={t['auc']:.4f}  Acc={t['accuracy']:.4f}  "
              f"Threshold={t['threshold']:.6f}")

        print("\n  [Attack 2 — Shadow-model MIA (Shokri et al. 2017)]")
        s = shadow_model_mia(victim, member, nonmember)
        print(f"    AUC={s['auc']:.4f}  Acc={s['accuracy']:.4f}")

        best_auc = max(t['auc'], s['auc'])
        if best_auc < 0.55:
            verdict = "✅ PRIVATE"
        elif best_auc < 0.70:
            verdict = "⚠️  MODERATE LEAKAGE"
        else:
            verdict = "❌ HIGH LEAKAGE"

        print(f"\n  Verdict for {org}: {verdict}  (best AUC={best_auc:.3f})")
        all_results.append({"client": org, "threshold_auc": t['auc'],
                            "shadow_auc": s['auc'], "best_auc": best_auc,
                            "verdict": verdict})

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY — Membership Inference Attack Results")
    print("="*60)
    print(f"  {'Client':<14} {'Threshold AUC':>14} {'Shadow AUC':>12} {'Best AUC':>10}  Verdict")
    print(f"  {'─'*14} {'─'*14} {'─'*12} {'─'*10}  {'─'*18}")
    for r in all_results:
        print(f"  {r['client']:<14} {r['threshold_auc']:>14.4f} {r['shadow_auc']:>12.4f} "
              f"{r['best_auc']:>10.4f}  {r['verdict']}")

    overall_best = max(r['best_auc'] for r in all_results)
    overall_verdict = "✅ PRIVATE" if overall_best < 0.55 else (
        "⚠️  MODERATE LEAKAGE" if overall_best < 0.70 else "❌ HIGH LEAKAGE")
    print(f"\n  Overall worst-case AUC: {overall_best:.4f}  →  {overall_verdict}")
    print(f"  Target for paper: AUC ≤ 0.55 (indistinguishable from random guessing)")
    print("="*60 + "\n")