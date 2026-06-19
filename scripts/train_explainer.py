"""
scripts/train_explainer.py — Train RF Diagnostic Classifier for AE Explainer
===============================================================================

Pipeline
--------
1. Load the NF-UNSW-NB15-v3 validation/test split via CICIDSDataLoader.
2. Load the pre-trained FlowAutoencoder.
3. Pass data through the AE and compute absolute residual vectors: |x - x̂|.
4. Train a RandomForestClassifier (n_estimators=50, max_depth=10)
   where X = residuals [N, F], y = detailed string attack label.
5. Save the trained classifier to saved_models/explainer_rf.pkl.

Usage
-----
  python scripts/train_explainer.py
  python scripts/train_explainer.py --fraction 0.5    # load more data
  python scripts/train_explainer.py --model-path saved_models/aura_bundle.pth
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ── Project imports ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from aura.models import FlowAutoencoder, AURAModelBundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading — uses the same CICIDSDataLoader but extracts the raw
# DataFrame so we can access both the scaled features AND the string
# "Attack" column for fine-grained labels.
# ─────────────────────────────────────────────────────────────────────────────

def load_labelled_data(
    fraction: float = 0.3,
) -> tuple:
    """
    Load scaled features and string attack labels from NF-UNSW-NB15-v3.

    Returns
    -------
    X_scaled : np.ndarray [N, F]  — MinMaxScaled feature matrix (attack + benign)
    y_labels : np.ndarray [N]     — string labels (e.g. 'Benign', 'DoS', 'Exploits')
    feature_cols : list[str]      — ordered feature column names
    """
    from aura.data_loader import CICIDSDataLoader, DATASET_PATH

    loader = CICIDSDataLoader(load_fraction=fraction)
    scaler = loader.fit_scaler()

    # Re-load the full CSV (benign + attack) using the loader's internal method
    # so feature column discovery and cleaning are consistent.
    df = loader._load_csv(str(DATASET_PATH))
    feature_cols = loader._feature_cols

    # ── Extract labels ───────────────────────────────────────────────────────
    label_col  = "Label"  if "Label"  in df.columns else cfg.LABEL_COL.strip()
    attack_col = "Attack" if "Attack" in df.columns else None

    binary_labels = df[label_col].values
    if attack_col is not None:
        string_labels = df[attack_col].astype(str).str.strip().values
    else:
        # Fallback: derive binary labels only
        string_labels = np.where(
            pd.api.types.is_numeric_dtype(df[label_col])
            and (df[label_col].values == 0),
            "Benign", "Unknown"
        )
        logger.warning("'Attack' column not found — using binary labels only.")

    # Fix benign rows: some datasets label benign Attack column as '' or 'NaN'
    is_benign = (binary_labels == 0) if pd.api.types.is_numeric_dtype(
        df[label_col]
    ) else (df[label_col].str.strip().str.upper() == "BENIGN").values

    string_labels[is_benign] = "Benign"

    # ── Scale features ───────────────────────────────────────────────────────
    X = df[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X).clip(0, 1)

    logger.info(
        f"Loaded {len(X_scaled)} rows, {len(feature_cols)} features.  "
        f"Label distribution: {pd.Series(string_labels).value_counts().to_dict()}"
    )

    return X_scaled, string_labels, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Residual Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_residuals(
    ae: FlowAutoencoder,
    X: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    """
    Run X through the autoencoder and return |x - x̂| per sample.

    Returns
    -------
    residuals : np.ndarray [N, F]  — absolute reconstruction error vectors
    """
    ae.eval()
    device = next(ae.parameters()).device

    residuals_list = []
    n = len(X)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_batch = torch.tensor(X[start:end], dtype=torch.float32).to(device)

        with torch.no_grad():
            x_hat, _ = ae(x_batch)
            res = (x_batch - x_hat).abs().cpu().numpy()

        residuals_list.append(res)

    return np.concatenate(residuals_list, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train RF diagnostic classifier on AE residuals"
    )
    parser.add_argument(
        "--fraction", type=float, default=0.3,
        help="Fraction of dataset to load (default: 0.3)"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to saved AE bundle (default: saved_models/aura_bundle.pth)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for RF model (default: saved_models/explainer_rf.pkl)"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Step 1: Load labelled data ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  AURA Explainer RF — Training Pipeline")
    print(f"{'='*60}\n")
    print("[1/5] Loading NF-UNSW-NB15-v3 dataset …")
    X_scaled, y_labels, feature_cols = load_labelled_data(fraction=args.fraction)

    # ── Step 2: Load pre-trained autoencoder ─────────────────────────────────
    print("[2/5] Loading pre-trained FlowAutoencoder …")
    ae = FlowAutoencoder()

    bundle_path = Path(args.model_path) if args.model_path else (
        cfg.MODELS_DIR / "aura_bundle.pth"
    )
    if bundle_path.exists():
        try:
            bundle = AURAModelBundle()
            bundle.load_state_dict(
                torch.load(str(bundle_path), map_location=device)
            )
            ae = bundle.autoencoder
            print(f"  ✓ Loaded AE from {bundle_path}")
        except Exception as e:
            logger.warning(f"Bundle load failed, using fresh AE: {e}")
            print(f"  ⚠ Using untrained AE (bundle load failed: {e})")
    else:
        print(f"  ⚠ No bundle found at {bundle_path} — using untrained AE")

    ae = ae.to(device).eval()

    # ── Step 3: Compute residuals |x - x̂| ───────────────────────────────────
    print("[3/5] Computing residual vectors |x - x̂| …")
    residuals = compute_residuals(ae, X_scaled)
    print(f"  Residual matrix shape: {residuals.shape}")

    # ── Step 4: Filter to attack-only for training ───────────────────────────
    # The RF learns to distinguish BETWEEN attack types, so benign rows are
    # excluded from training (they produce near-zero residuals and would
    # dominate the classifier).  At inference time, the explainer is only
    # invoked when Layer 1 has already flagged an anomaly.
    attack_mask = y_labels != "Benign"
    X_train = residuals[attack_mask]
    y_train = y_labels[attack_mask]

    print(f"  Attack samples for training: {len(X_train)}")
    print(f"  Attack classes: {np.unique(y_train).tolist()}")

    if len(X_train) == 0:
        print("\n  ✗ No attack samples found — cannot train classifier.")
        print("    Ensure the dataset contains rows with Label=1 and an 'Attack' column.")
        sys.exit(1)

    # ── Step 5: Train RandomForestClassifier ─────────────────────────────────
    print("[4/5] Training RandomForestClassifier …")
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",  # handle class imbalance
    )
    clf.fit(X_train, y_train)

    # Quick in-sample evaluation (full cross-val is overkill for this use case)
    y_pred = clf.predict(X_train)
    print(f"\n  In-sample classification report:")
    print(classification_report(y_train, y_pred, zero_division=0))

    # Log feature importances (top 10)
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("  Top 10 feature importances:")
    for rank, idx in enumerate(top_idx, 1):
        fname = feature_cols[idx] if idx < len(feature_cols) else f"Feature_{idx}"
        print(f"    {rank:2d}. [{idx:2d}] {fname:30s}  {importances[idx]:.4f}")

    # ── Step 6: Save model ───────────────────────────────────────────────────
    output_path = Path(args.output) if args.output else (
        cfg.MODELS_DIR / "explainer_rf.pkl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, str(output_path))

    print(f"\n[5/5] ✓ Classifier saved: {output_path}")
    print(f"  Classes:  {clf.classes_.tolist()}")
    print(f"  Features: {clf.n_features_in_}")
    print(f"\n{'='*60}")
    print("  Explainer RF training complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
