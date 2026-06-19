"""
aura/ae_explainer.py — AE Feature Attribution & Attack Classification (RF)
===========================================================================

Given a per-feature reconstruction residual vector |x - x_hat|,
this module:

  1. Loads a pre-trained RandomForest diagnostic classifier from
     saved_models/explainer_rf.pkl
  2. Predicts the attack type and confidence from the residual pattern
  3. Extracts feature importances × live residuals to compute live_impact
  4. Names the top contributing features in human-readable terms
  5. Produces a plain-English explanation panel for the SOC operator

Design
------
The RF classifier was trained on per-sample absolute residual vectors
(|x - x_hat|) from known attack traffic.  At inference time:
  - predict()       → attack type string
  - predict_proba() → confidence score
  - feature_importances_ × live_residual → per-feature live_impact

This replaces the earlier cosine-similarity signature matching approach
with a data-driven classifier that learns the residual patterns directly.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Index → Human-Readable Name
# All NF-UNSW-NB15-v3 features (IPs, ports, timestamps, Label, Attack stripped)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: Dict[int, str] = {
    0:  "Protocol",
    1:  "L7 Protocol",
    2:  "Inbound Bytes",
    3:  "Inbound Packets",
    4:  "Outbound Bytes",
    5:  "Outbound Packets",
    6:  "TCP Flags",
    7:  "Client TCP Flags",
    8:  "Server TCP Flags",
    9:  "Flow Duration (ms)",
    10: "Duration In",
    11: "Duration Out",
    12: "Min TTL",
    13: "Max TTL",
    14: "Longest Flow Pkt",
    15: "Shortest Flow Pkt",
    16: "Min IP Pkt Len",
    17: "Max IP Pkt Len",
    18: "Src→Dst Bytes/s",
    19: "Dst→Src Bytes/s",
    20: "Retransmitted In Bytes",
    21: "Retransmitted In Pkts",
    22: "Retransmitted Out Bytes",
    23: "Retransmitted Out Pkts",
    24: "Src→Dst Avg Throughput",
    25: "Dst→Src Avg Throughput",
    26: "Pkts ≤128 Bytes",
    27: "Pkts 128–256 Bytes",
    28: "Pkts 256–512 Bytes",
    29: "Pkts 512–1024 Bytes",
    30: "Pkts 1024–1514 Bytes",
    31: "TCP Win Max In",
    32: "TCP Win Max Out",
    33: "ICMP Type",
    34: "ICMP IPv4 Type",
    35: "DNS Query ID",
    36: "DNS Query Type",
    37: "DNS TTL Answer",
    38: "FTP Command Ret Code",
    39: "Src→Dst IAT Min",
    40: "Src→Dst IAT Max",
    41: "Src→Dst IAT Avg",
    42: "Src→Dst IAT Stddev",
    43: "Dst→Src IAT Min",
    44: "Dst→Src IAT Max",
    45: "Dst→Src IAT Avg",
    46: "Dst→Src IAT Stddev",
}

# ─────────────────────────────────────────────────────────────────────────────
# Feature Groups (for grouped explanation display)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_GROUPS: Dict[str, List[int]] = {
    "Volume / Bytes":     [2, 3, 4, 5, 20, 21, 22, 23],
    "Bandwidth / Rates":  [18, 19, 24, 25],
    "Timing / IAT":       [9, 10, 11, 39, 40, 41, 42, 43, 44, 45, 46],
    "TCP / Protocol":     [0, 6, 7, 8],
    "Packet Size":        [14, 15, 16, 17, 26, 27, 28, 29, 30],
    "Application Layer":  [1, 33, 34, 35, 36, 37, 38],
    "Window / TTL":       [12, 13, 31, 32],
}


# Human-readable explanations per attack type
ATTACK_EXPLANATIONS: Dict[str, Dict[str, str]] = {
    "DoS": {
        "icon":    "🌊",
        "summary": "Denial-of-Service flood attack detected",
        "detail":  (
            "Packet rate and bandwidth are abnormally high with near-zero "
            "inter-arrival time — consistent with a UDP/SYN/ICMP flood. "
            "Incomplete TCP handshakes (high SYN, low ACK) confirm the source "
            "is NOT establishing legitimate connections. "
            "Action: rate-limit the source subnet and engage upstream scrubbing."
        ),
        "why_high": "Inbound Packets and TCP Flags are the primary drivers — "
                    "the model has never seen legitimate traffic at this rate.",
    },
    "Reconnaissance": {
        "icon":    "🔍",
        "summary": "Network reconnaissance / port scan detected",
        "detail":  (
            "Multiple extremely short flows with minimal byte transfer and "
            "high RST + SYN flag counts — the attacker is probing which services "
            "are open without completing any connection. "
            "Action: block the scanning IP, alert vulnerability management team."
        ),
        "why_high": "TCP Flags and very short Flow Duration are the primary "
                    "drivers — legitimate flows do not terminate this abruptly en masse.",
    },
    "Exploits": {
        "icon":    "💣",
        "summary": "Exploit attempt detected (buffer overflow / code execution)",
        "detail":  (
            "Unusually large packet sizes combined with TCP flag anomalies and "
            "retransmission bursts suggest payload delivery for a known exploit. "
            "The oversized packets and timing jitter indicate automated tooling. "
            "Action: isolate the target host, check for compromise indicators, "
            "verify patch levels."
        ),
        "why_high": "Longest Flow Pkt and Retransmitted In Bytes are the primary "
                    "drivers — exploit payloads create abnormal packet size distributions.",
    },
    "Fuzzers": {
        "icon":    "🔀",
        "summary": "Fuzzing attack detected (input mutation / protocol abuse)",
        "detail":  (
            "Chaotic timing variance with mixed packet sizes (very large and "
            "very small alternating) on unusual protocols — consistent with "
            "automated fuzzing tools probing for vulnerabilities. "
            "Action: rate-limit the source, review application error logs for crashes."
        ),
        "why_high": "Src→Dst IAT Stddev and packet size variance are the primary "
                    "drivers — fuzzing creates chaotic, non-human traffic patterns.",
    },
    "Generic": {
        "icon":    "⚡",
        "summary": "Generic network attack pattern detected",
        "detail":  (
            "Broad anomalies across volume, timing, and protocol features suggest "
            "a multi-vector or generic network attack. The pattern does not closely "
            "match a single specific attack type but deviates significantly from "
            "normal traffic across multiple feature groups. "
            "Action: escalate to Tier-2 analysis, capture full PCAP for forensics."
        ),
        "why_high": "Spread anomalies across volume and timing features — "
                    "indicates a broad-spectrum attack or novel variant.",
    },
    "Backdoor": {
        "icon":    "🚪",
        "summary": "Backdoor / C2 communication detected",
        "detail":  (
            "Symmetric bidirectional traffic with periodic beaconing intervals "
            "(very consistent IAT with near-zero jitter) over sustained connections. "
            "This is the hallmark of command-and-control communication from an "
            "implanted backdoor. "
            "Action: isolate the host immediately, initiate EDR investigation, "
            "check for lateral movement."
        ),
        "why_high": "Src→Dst IAT Avg/Stddev and symmetric byte ratios are the "
                    "primary drivers — robotic periodic beaconing is never legitimate.",
    },
    "Shellcode": {
        "icon":    "🐚",
        "summary": "Shellcode payload delivery detected",
        "detail":  (
            "Large inbound payload with oversized packets and aggressive TCP push "
            "flags on short-duration flows — consistent with shellcode injection. "
            "The payload size and delivery pattern match known exploit kit behaviour. "
            "Action: quarantine the target, scan for injected code, review memory dumps."
        ),
        "why_high": "Inbound Bytes and Longest Flow Pkt are the primary "
                    "drivers — shellcode payloads create distinctive size signatures.",
    },
    "Worms": {
        "icon":    "🐛",
        "summary": "Network worm propagation detected",
        "detail":  (
            "High bidirectional packet rates with elevated throughput — the pattern "
            "suggests automated self-replication across the network. Both inbound "
            "and outbound traffic spikes indicate the host is both receiving worm "
            "payloads and actively scanning/infecting other hosts. "
            "Action: network-wide containment, identify patient zero, apply patches."
        ),
        "why_high": "Src→Dst Bytes/s and bidirectional packet counts are the "
                    "primary drivers — worm propagation creates symmetric high-rate flows.",
    },
    "Analysis": {
        "icon":    "🔬",
        "summary": "Deep analysis / probing activity detected",
        "detail":  (
            "Extended-duration flows with methodical timing (consistent IAT) and "
            "small packet sizes — consistent with automated service enumeration "
            "or vulnerability scanning tools performing deep analysis. "
            "Action: review scan targets, assess exposure, block the source IP."
        ),
        "why_high": "Flow Duration and Src→Dst IAT Avg are the primary "
                    "drivers — analysis probes are characteristically slow and methodical.",
    },
    "Data Exfiltration": {
        "icon":    "📤",
        "summary": "Data exfiltration (low & slow) detected",
        "detail":  (
            "Extreme asymmetry: large forward (outbound) byte count vs near-zero "
            "backward (inbound) bytes over a sustained, long connection. "
            "Robotic inter-arrival timing (low Stddev) indicates machine-scripted "
            "exfiltration rather than human-driven traffic. "
            "Action: terminate the connection, inspect endpoint for malware, "
            "check DLP logs for data classification hits."
        ),
        "why_high": "Inbound Bytes and Src→Dst Bytes/s ratios are the primary "
                    "drivers — upload-only sustained flows are outside the normal manifold.",
    },
    "Lateral Movement": {
        "icon":    "↔️",
        "summary": "Internal lateral movement / east-west threat detected",
        "detail":  (
            "High timing jitter (Src→Dst IAT Stddev) combined with long idle periods "
            "between bursts is the hallmark of a compromised host performing "
            "internal reconnaissance. The GNN (Layer 2) should confirm abnormal "
            "device-to-device connectivity not seen during training. "
            "Action: isolate the source host, initiate EDR investigation."
        ),
        "why_high": "Src→Dst IAT Stddev and Duration In are the primary drivers — "
                    "the beacon-like sleep-burst pattern is not present in normal flows.",
    },
    "Port Scan": {
        "icon":    "🔍",
        "summary": "Port scanning activity detected",
        "detail":  (
            "Multiple extremely short flows with minimal byte transfer and "
            "high RST + SYN TCP flag counts — the attacker is probing services. "
            "Action: block the scanning IP, alert vulnerability management team."
        ),
        "why_high": "TCP Flags and very short Flow Duration are the primary "
                    "drivers — legitimate flows do not terminate this abruptly.",
    },
    "Web Attack": {
        "icon":    "💉",
        "summary": "Web application attack detected (SQLi / XSS)",
        "detail":  (
            "Elevated PSH flags and large forward payload sizes on short-duration "
            "flows suggest HTTP request manipulation — consistent with SQL injection "
            "or XSS payloads being submitted. "
            "Action: review WAF logs, block the offending IP, audit database "
            "query logs for injection attempts."
        ),
        "why_high": "Client TCP Flags and Server TCP Flags are primary drivers — "
                    "legitimate HTTP traffic does not push this many payloads per flow.",
    },
    "Unknown Anomaly": {
        "icon":    "❓",
        "summary": "Anomalous pattern — no close attack signature match",
        "detail":  (
            "The reconstruction error is elevated but the residual pattern "
            "does not closely match any known attack class. This may indicate "
            "a novel attack variant, misconfigured device, or legitimate but unusual "
            "traffic pattern. "
            "Action: review the top contributing features manually and escalate "
            "to Tier-2 analysis."
        ),
        "why_high": "Spread residuals across multiple unrelated feature groups — "
                    "no single attack taxonomy matches well.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# RF Classifier — Lazy Loader
# ─────────────────────────────────────────────────────────────────────────────

_RF_CACHE = None


def _load_rf_classifier():
    """
    Lazy-load the trained RandomForest diagnostic classifier.
    Returns None if the model file does not exist (graceful fallback).
    """
    global _RF_CACHE
    if _RF_CACHE is not None:
        return _RF_CACHE

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config as cfg

    rf_path = cfg.MODELS_DIR / "explainer_rf.pkl"

    if not rf_path.exists():
        logger.warning(
            f"Explainer RF model not found at {rf_path}.  "
            "Run `python scripts/train_explainer.py` to train it.  "
            "Falling back to residual-only explanation (no attack classification)."
        )
        return None

    try:
        import joblib
        _RF_CACHE = joblib.load(str(rf_path))
        logger.info(f"Loaded explainer RF from {rf_path}  "
                     f"(classes={list(_RF_CACHE.classes_)})")
    except Exception as e:
        logger.error(f"Failed to load explainer RF: {e}")
        return None

    return _RF_CACHE


# ─────────────────────────────────────────────────────────────────────────────
# Core Explanation Function
# ─────────────────────────────────────────────────────────────────────────────

def explain_ae(
    residuals:  np.ndarray,   # [F] mean absolute per-feature residual
    top_k:      int = 5,
) -> Dict:
    """
    Given a per-feature reconstruction residual vector, return a structured
    explanation dict for the dashboard.

    Strategy
    --------
    1. Load the RF classifier (lazy, cached).
    2. Run predict() → attack type string.
    3. Run predict_proba() → confidence score for the predicted class.
    4. Compute live_impact = feature_importances_ × residuals.
    5. Rank features by live_impact and return top_features.

    If the RF model is not available, falls back to ranking features by
    raw residual magnitude (graceful degradation).

    Parameters
    ----------
    residuals  : np.ndarray [F] — mean |x - x_hat| per feature
    top_k      : how many top features to surface

    Returns
    -------
    dict with keys:
      top_features    : list of (feature_name, residual_value, feature_index)
      group_residuals : dict {group_name: mean_residual}
      inferred_attack : str — predicted attack label (or "Unknown Anomaly")
      match_score     : float ∈ [0,1] — RF confidence for predicted class
      explanation     : dict (icon, summary, detail, why_high)
    """
    residuals = np.array(residuals, dtype=np.float32)
    n_feats   = len(residuals)

    # ── Group-level residuals ─────────────────────────────────────────────
    group_residuals: Dict[str, float] = {}
    for group_name, indices in FEATURE_GROUPS.items():
        valid = [residuals[i] for i in indices if i < n_feats]
        group_residuals[group_name] = float(np.mean(valid)) if valid else 0.0

    # ── RF-based attack classification ────────────────────────────────────
    clf = _load_rf_classifier()

    if clf is not None:
        # Reshape for sklearn: [1, F]
        residual_input = residuals.reshape(1, -1)

        # Handle dimension mismatch: pad or truncate to match RF's expected
        # feature count (covers both 43-feature and 47-feature scenarios).
        expected_feats = clf.n_features_in_
        if residual_input.shape[1] < expected_feats:
            pad = np.zeros((1, expected_feats - residual_input.shape[1]),
                           dtype=np.float32)
            residual_input = np.concatenate([residual_input, pad], axis=1)
        elif residual_input.shape[1] > expected_feats:
            residual_input = residual_input[:, :expected_feats]

        # Predict attack type and confidence
        predicted_attack = clf.predict(residual_input)[0]
        proba = clf.predict_proba(residual_input)[0]
        class_idx = list(clf.classes_).index(predicted_attack)
        confidence = float(proba[class_idx])

        # Compute live_impact: feature_importances × live residual
        importances = clf.feature_importances_
        # Align importances with the actual residual length
        imp_len = min(len(importances), n_feats)
        live_impact = np.zeros(n_feats, dtype=np.float32)
        live_impact[:imp_len] = importances[:imp_len] * residuals[:imp_len]

        # Top-K features by live_impact
        top_indices = np.argsort(live_impact)[::-1][:top_k]
        top_features = [
            (FEATURE_NAMES.get(int(i), f"Feature_{i}"),
             float(residuals[i]),
             int(i))
            for i in top_indices
        ]

        best_attack = str(predicted_attack)
        best_score  = confidence

    else:
        # ── Fallback: rank by raw residual magnitude ──────────────────────
        top_indices = np.argsort(residuals)[::-1][:top_k]
        top_features = [
            (FEATURE_NAMES.get(int(i), f"Feature_{i}"),
             float(residuals[i]),
             int(i))
            for i in top_indices
        ]
        best_attack = "Unknown Anomaly"
        best_score  = 0.0

    explanation = ATTACK_EXPLANATIONS.get(
        best_attack, ATTACK_EXPLANATIONS["Unknown Anomaly"]
    )

    return {
        "top_features":    top_features,
        "group_residuals": group_residuals,
        "inferred_attack": best_attack,
        "match_score":     round(best_score, 3),
        "explanation":     explanation,
    }
