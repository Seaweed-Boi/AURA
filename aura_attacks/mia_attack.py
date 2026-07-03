"""
mia_attack.py  —  Membership Inference Attack against AURA's FL pipeline
=========================================================================
Target  : AURA FlowAutoencoder  47→[32,24]→16→[24,32]→47
Dataset : NF-UNSW-NB15-v3

HOW TO RUN (from your AURA root):
    python aura_attacks/mia_attack.py
    python aura_attacks/mia_attack.py --real-model   # uses your real checkpoint

Phase-4 relevance
-----------------
Tune Opacus DP-SGD epsilon by watching AUC drop as noise increases.
Target: AUC ≤ 0.55 (near-random guessing = model leaks nothing).

Two variants:
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-model",  action="store_true")
    ap.add_argument("--n-samples",   type=int, default=200)
    args = ap.parse_args()

    torch.manual_seed(0)
    print("\n" + "="*60)
    print("  AURA — Membership Inference Attack (Phase 4)")
    print("="*60)

    victim = _load_real_model() if args.real_model else (lambda: (lambda m: (m.eval(), m)[1])(_StandaloneAE()))()

    member    = torch.randn(args.n_samples, FEATURE_DIM, generator=torch.Generator().manual_seed(1))
    nonmember = torch.randn(args.n_samples, FEATURE_DIM, generator=torch.Generator().manual_seed(2))

    print("\n[Attack 1 — Threshold MIA]")
    t = threshold_mia(victim, member, nonmember)
    print(f"  AUC={t['auc']:.4f}  Acc={t['accuracy']:.4f}  Threshold={t['threshold']:.6f}")

    print("\n[Attack 2 — Shadow-model MIA]")
    s = shadow_model_mia(victim, member, nonmember)
    print(f"  AUC={s['auc']:.4f}  Acc={s['accuracy']:.4f}")

    best = max(t['auc'], s['auc'])
    print("\n" + "-"*60)
    verdict = "✅ PRIVATE" if best < 0.55 else ("⚠️  MODERATE LEAKAGE" if best < 0.70 else "❌ HIGH LEAKAGE")
    print(f"  Verdict: {verdict}  (best AUC={best:.3f})")
    print("  Target for paper: AUC ≤ 0.55 under Opacus DP-SGD")
    print("  Re-run with --real-model + real NF-UNSW-NB15-v3 held-out rows")
    print("="*60 + "\n")