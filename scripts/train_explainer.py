"""
scripts/train_explainer.py — Train RF Diagnostic Classifier for AE Explainer
===============================================================================

Pipeline
--------
1. Load the NF-UNSW-NB15-v3 validation/test split via CICIDSDataLoader.
2. Load the pre-trained FlowAutoencoder.
3. Pass data through the AE and compute:
     - Absolute residual vectors: |x - x̂|  [N, F]
     - Bottleneck latent vectors: z          [N, latent_dim]
   Then concatenate → feature matrix [N, F + latent_dim].
4. Train a RandomForestClassifier (n_estimators=500, max_depth=10)
   where X = concat(residuals, latent) [N, F+L], y = detailed attack label.
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
from lightgbm import LGBMClassifier
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
# Feature Extraction: Residuals ∥ Latent vectors
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(
    ae: FlowAutoencoder,
    X: np.ndarray,
    batch_size: int = 2048,
) -> tuple:
    """
    Run X through the autoencoder and return both:
      - |x - x̂|  (absolute residuals, shape [N, F])
      - z         (bottleneck latent vectors, shape [N, latent_dim])

    The concatenated matrix ``concat([residuals, latent], axis=1)`` is returned
    as the primary feature vector for the RF classifier, giving it both the
    reconstruction-error signal *and* the compressed semantic representation.

    Returns
    -------
    features        : np.ndarray [N, F + latent_dim] — concatenated feature matrix
    residual_vectors: np.ndarray [N, F]              — raw residuals (for importances)
    latent_vectors  : np.ndarray [N, latent_dim]     — AE bottleneck activations
    """
    ae.eval()
    device = next(ae.parameters()).device

    residual_list = []
    latent_list   = []
    n = len(X)

    for start in range(0, n, batch_size):
        end     = min(start + batch_size, n)
        x_batch = torch.tensor(X[start:end], dtype=torch.float32).to(device)

        with torch.no_grad():
            x_hat, z = ae(x_batch)
            residual_list.append((x_batch - x_hat).abs().cpu().numpy())
            latent_list.append(z.cpu().numpy())

    residual_vectors = np.concatenate(residual_list, axis=0)
    latent_vectors   = np.concatenate(latent_list,   axis=0)

    features = np.concatenate(
        [
            residual_vectors,
            latent_vectors,
        ],
        axis=1,
    )

    return features, residual_vectors, latent_vectors


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

    # ── Step 3: Compute residuals + latent vectors ───────────────────────────
    print("[3/5] Computing residual & latent feature vectors …")
    features, residual_vectors, latent_vectors = compute_features(ae, X_scaled)
    print(f"  Residual vectors shape : {residual_vectors.shape}")
    print(f"  Latent vectors shape   : {latent_vectors.shape}")
    print(f"  Combined feature shape : {features.shape}")

    # ── Step 4: Filter to attack-only for training ───────────────────────────
    # The RF learns to distinguish BETWEEN attack types, so benign rows are
    # excluded from training (they produce near-zero residuals and would
    # dominate the classifier).  At inference time, the explainer is only
    # invoked when Layer 1 has already flagged an anomaly.
    attack_mask = y_labels != "Benign"
    X_raw    = features[attack_mask]
    # Force plain NumPy str array — PyArrow-backed pandas arrays cause
    # joblib ChunkedArray fancy-indexing failures in child processes.
    y_train  = np.array(y_labels[attack_mask], dtype=str)

    # Build a named DataFrame so LightGBM 4.x doesn't warn about missing
    # feature names when predict() receives a plain numpy slice.
    n_residual = residual_vectors.shape[1]
    n_latent   = latent_vectors.shape[1]
    col_names  = ([f"residual_{i}" for i in range(n_residual)] +
                  [f"latent_{i}"   for i in range(n_latent)])
    X_train = pd.DataFrame(X_raw, columns=col_names)

    print(f"  Attack samples for training: {len(X_train)}")
    print(f"  Attack classes: {np.unique(y_train).tolist()}")

    if len(X_train) == 0:
        print("\n  ✗ No attack samples found — cannot train classifier.")
        print("    Ensure the dataset contains rows with Label=1 and an 'Attack' column.")
        sys.exit(1)

# ── Step 5: Train LGBMClassifier ────────────────────────────────────────
    print("[4/5] Training LGBMClassifier …")
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    clf = LGBMClassifier(
        n_estimators=200,  # 200 is plenty for LGBM; 500 × 5 folds sequentially is too slow
        max_depth=10,
        random_state=42,
        n_jobs=1,          # must be 1 — joblib conflict when cross_val_predict also uses n_jobs=-1
        class_weight="balanced",
        verbose=-1,        # suppress LightGBM's per-tree stdout
    )

    # Honest evaluation: every sample predicted by a model that never saw it
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # n_jobs=1: LightGBM's internal thread pool conflicts with joblib multiprocessing
    y_pred_cv = cross_val_predict(clf, X_train, y_train, cv=skf, n_jobs=1)
    print(f"\n  Cross-validated classification report (5-fold stratified):")
    print(classification_report(y_train, y_pred_cv, zero_division=0))

    # Fit on full data for the saved model (done AFTER reporting)
    clf.fit(X_train, y_train)
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

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(
        y_train,
        y_pred_cv,
        labels=clf.classes_
    )

    _print_confusion_matrix(cm, clf.classes_)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty confusion-matrix printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_confusion_matrix(cm: np.ndarray, class_names: np.ndarray) -> None:
    """
    Print a colour-coded confusion matrix to stdout.

    - Diagonal cells (correct predictions) are highlighted in green.
    - Off-diagonal cells are highlighted in red proportional to magnitude.
    - Per-class recall is shown at the end of each row.
    - No external dependencies beyond stdlib ANSI codes.
    """
    # ── ANSI helpers ─────────────────────────────────────────────────────────
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    DIM    = "\033[2m"

    n          = len(class_names)
    col_w      = 8                          # cell width
    label_w    = max(len(c) for c in class_names) + 2
    row_totals = cm.sum(axis=1, keepdims=True).clip(1)  # avoid /0
    max_off    = cm.copy(); np.fill_diagonal(max_off, 0)
    max_off_v  = max_off.max() or 1

    # ── Header ───────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  Confusion Matrix  (rows = actual, cols = predicted){RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}\n")

    # Column headers (abbreviated to col_w-1 chars)
    header = " " * label_w
    for cls in class_names:
        abbr = cls[:col_w - 1].center(col_w)
        header += f"{BOLD}{abbr}{RESET}"
    header += f"  {BOLD}Recall{RESET}"
    print(header)
    print(DIM + " " * label_w + ("─" * col_w) * n + "  " + "──────" + RESET)

    # ── Rows ─────────────────────────────────────────────────────────────────
    for i, cls in enumerate(class_names):
        row_label = f"{BOLD}{cls[:label_w - 1]:<{label_w - 1}}{RESET} "
        row_str   = row_label

        for j in range(n):
            val = cm[i, j]
            cell = str(val).center(col_w)
            if i == j:                              # diagonal → green
                row_str += f"{GREEN}{BOLD}{cell}{RESET}"
            elif val == 0:
                row_str += f"{DIM}{cell}{RESET}"   # zero → dim
            else:
                # Intensity: dim red → bright red based on relative magnitude
                intensity = val / max_off_v
                if intensity > 0.5:
                    row_str += f"{RED}{BOLD}{cell}{RESET}"
                else:
                    row_str += f"{YELLOW}{cell}{RESET}"

        recall = cm[i, i] / row_totals[i, 0]
        recall_col = (
            f"{GREEN}{BOLD}" if recall >= 0.9 else
            f"{YELLOW}"     if recall >= 0.7 else
            f"{RED}"
        )
        row_str += f"  {recall_col}{recall:6.1%}{RESET}"
        print(row_str)

    # ── Footer ───────────────────────────────────────────────────────────────
    overall_acc = np.diag(cm).sum() / cm.sum()
    print(DIM + " " * label_w + ("─" * col_w) * n + "  " + "──────" + RESET)
    print(
        f"{BOLD}  Overall accuracy: "
        f"{GREEN if overall_acc >= 0.9 else YELLOW if overall_acc >= 0.7 else RED}"
        f"{overall_acc:.2%}{RESET}   "
        f"{DIM}(5-fold cross-validated){RESET}\n"
    )


if __name__ == "__main__":
    main()

