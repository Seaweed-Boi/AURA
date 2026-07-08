"""
aura/split_manager.py — Canonical Train/Test Split Authority
=============================================================

All scripts that train or evaluate models MUST use this module to obtain
train/test indices. Any script that computes its own split independently is a
reproducibility bug.

Design
------
- The split is computed ONCE on the full list of streaming graph windows.
- Windows are first separated into attack-containing and benign-only buckets.
- Each bucket is split chronologically (first 80% → train, last 20% → test).
- Both buckets are re-sorted by original chronological index before returning,
  preserving temporal order required by Mode D's EMA tracker.
- The resulting train/test window indices are saved to splits/canonical_split.npz
  and reloaded on every subsequent call — guaranteeing all scripts see the same
  exact split regardless of invocation order.

Usage
-----
    from aura.split_manager import get_canonical_split
    import config as cfg

    calibration_windows, train_windows, test_windows = get_canonical_split(
        all_windows, test_fraction=cfg.TEST_SPLIT_FRACTION
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

import config as cfg

logger = logging.getLogger(__name__)

# File that persists the canonical index arrays between runs
_SPLIT_FILE = cfg.SPLITS_DIR / "canonical_split.npz"


def get_canonical_split(
    all_windows: List[Tuple],
    test_fraction: float = cfg.TEST_SPLIT_FRACTION,
    calib_fraction: float = cfg.CALIB_SPLIT_FRACTION,
    force_recompute: bool = False,
) -> Tuple[List, List, List]:
    """
    Return (calibration_windows, train_windows, test_windows) using a single
    canonical train/test split that is saved to disk and reused across scripts.

    Parameters
    ----------
    all_windows : list of (graph_dict, labels_tensor) tuples in streaming order.
    test_fraction : fraction of windows held out as test (default 0.20).
    calib_fraction : fraction of *train* windows used for threshold calibration
                     (default 0.10). Taken from the start of train_windows.
    force_recompute : if True, discard any saved split and recompute from scratch.

    Returns
    -------
    calibration_windows : list (subset of train_windows, used for AE/GNN calibration)
    train_windows       : list
    test_windows        : list
    """
    total = len(all_windows)
    if total == 0:
        raise RuntimeError("get_canonical_split received an empty window list.")

    # ── Load existing split if available ────────────────────────────────────
    if _SPLIT_FILE.exists() and not force_recompute:
        data = np.load(_SPLIT_FILE)
        saved_total = int(data["total"])
        if saved_total == total:
            train_idx = data["train_idx"].tolist()
            test_idx  = data["test_idx"].tolist()
            logger.info(
                f"[split_manager] Loaded canonical split from {_SPLIT_FILE} "
                f"(total={total}, train={len(train_idx)}, test={len(test_idx)})"
            )
            return _build_output(all_windows, train_idx, test_idx, calib_fraction)
        else:
            logger.warning(
                f"[split_manager] Saved split has total={saved_total} but current "
                f"dataset has {total} windows. Recomputing split."
            )

    # ── Compute stratified chronological split ──────────────────────────────
    # Attacks are front-loaded in NF-UNSW-NB15-v3; a naive tail-slice would
    # put 0 attacks in the test set.  Splitting each bucket independently
    # guarantees both sets contain attacks.
    attack_idx = [i for i, (_, lbl) in enumerate(all_windows) if lbl.sum() > 0]
    benign_idx = [i for i, (_, lbl) in enumerate(all_windows) if lbl.sum() == 0]

    def _chrono_split(indices: list) -> Tuple[list, list]:
        cut = int(len(indices) * (1.0 - test_fraction))
        return indices[:cut], indices[cut:]

    atk_train, atk_test = _chrono_split(attack_idx)
    ben_train, ben_test = _chrono_split(benign_idx)

    train_idx = sorted(atk_train + ben_train)
    test_idx  = sorted(atk_test  + ben_test)

    # ── Persist to disk ──────────────────────────────────────────────────────
    cfg.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        _SPLIT_FILE,
        total=np.array(total),
        train_idx=np.array(train_idx, dtype=np.int64),
        test_idx=np.array(test_idx,  dtype=np.int64),
    )
    logger.info(
        f"[split_manager] Canonical split saved to {_SPLIT_FILE} "
        f"(total={total}, train={len(train_idx)}, test={len(test_idx)})"
    )

    return _build_output(all_windows, train_idx, test_idx, calib_fraction)


def _build_output(
    all_windows: List[Tuple],
    train_idx: List[int],
    test_idx: List[int],
    calib_fraction: float,
) -> Tuple[List, List, List]:
    """Materialise the window lists and derive the calibration subset."""
    train_windows = [all_windows[i] for i in train_idx]
    test_windows  = [all_windows[i] for i in test_idx]

    n_calib = max(5, int(len(train_windows) * calib_fraction))
    calibration_windows = train_windows[:n_calib]

    logger.info(
        f"[split_manager] train={len(train_windows)}, "
        f"test={len(test_windows)}, "
        f"calib={len(calibration_windows)}"
    )

    # Sanity check — never allow index overlap (would mean data leakage)
    overlap = set(train_idx) & set(test_idx)
    if overlap:
        raise RuntimeError(
            f"[split_manager] FATAL: {len(overlap)} indices appear in both "
            "train and test sets. The split is corrupt."
        )

    return calibration_windows, train_windows, test_windows
