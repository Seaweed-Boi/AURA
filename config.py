"""
config.py — AURA Global Configuration
======================================
Single source of truth for ALL hyperparameters, paths, and system constants.
Centralising config prevents magic numbers from scattering across modules.

Research-grade design goals
---------------------------
  * Zero hardcoded values in any other module — every tunable lives here.
  * AE thresholds are resolved dynamically from calibration_results.json
    (written by calibrate_thresholds.py) so they always reflect the actual
    benign MSE distribution — NOT a magic number.
  * Attack corruption profiles are resolved from attack_class_stats.json
    (written by scripts/train_explainer.py) so they reflect real dataset
    percentiles — NOT hand-typed ranges.
  * All paths constructed relative to BASE_DIR so the project is portable.
"""

import json
import logging
import os
from pathlib import Path

_cfg_log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.resolve()
CSV_DIR    = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "saved_models"
LOGS_DIR   = BASE_DIR / "logs"
CONTRACTS_DIR = BASE_DIR / "contracts"

# Ensure output dirs exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET  (NF-UNSW-NB15-v3)
# ─────────────────────────────────────────────────────────────────────────────

# NF-UNSW-NB15-v3 uses real source/destination IPs for genuine topology.
NUM_SYNTHETIC_NODES = 20

# Column name for the target label
LABEL_COL = "Label"

# The label value that represents benign (normal) traffic in NF-UNSW-NB15-v3
# Label column is binary: 0 = Benign, 1 = Attack
BENIGN_LABEL = 0

# Fraction of data to load per CSV (1.0 = all rows; reduce for speed during dev)
DATA_LOAD_FRACTION = 0.3   # 30 % is enough to demo; use 1.0 for full training

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH / TTL EDGE DECAY
# ─────────────────────────────────────────────────────────────────────────────

# Rolling time-window size in simulated "ticks" (1 tick ≈ 1 second of NetFlow)
WINDOW_SIZE = 60          # number of flow rows per graph snapshot

# Time-To-Live: an edge is pruned after this many windows without traffic
EDGE_TTL_WINDOWS = 3

# ─────────────────────────────────────────────────────────────────────────────
# AUTOENCODER (Layer 1 — Statistical Tripwire)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_DIM     = 47    # Number of normalised NetFlow statistical features (NF-UNSW-NB15-v3)
ENCODER_DIMS    = [32, 24]   # Progressive compression (smaller network for 47 features)
LATENT_DIM      = 16    # Bottleneck: the latent fingerprint space
DECODER_DIMS    = [24, 32]   # Mirror of encoder (symmetric reconstruction)

AE_LEARNING_RATE = 1e-3
AE_EPOCHS        = 50        # Full training run on NF-UNSW-NB15-v3 subset
AE_BATCH_SIZE    = 256

# Contrastive negative-sampling margin (pushes attack embeddings away from
# the normal manifold during simulated baseline hardening)
CONTRASTIVE_MARGIN = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# STGNN (Layer 2 — Contextual Validator)
# ─────────────────────────────────────────────────────────────────────────────

# Node feature dimensionality fed to the GNN
# Each node's feature vector = mean of its incident edge (flow) features
GNN_INPUT_DIM  = FEATURE_DIM
GNN_HIDDEN_DIM = 64
GNN_OUTPUT_DIM = 32          # Latent node embedding dimension
GNN_LEARNING_RATE = 5e-4
GNN_EPOCHS     = 50

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC THRESHOLDING (Exponential Moving Average over batch MSE)
# ─────────────────────────────────────────────────────────────────────────────

# EMA smoothing factor (α). Higher = reacts faster but is noisier.
# Lower = more stable but slower to adapt.
EMA_ALPHA = 0.05

# An alert is raised when:  loss > EMA_mean + (EMA_SIGMA_MULTIPLIER × EMA_std)
EMA_SIGMA_MULTIPLIER = 1.5

# Warm-up batches before thresholds are active (avoids cold-start false alarms)
EMA_WARMUP_BATCHES = 50

# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY ENGINE — Temporal Accumulator + EMA Trajectory
# ─────────────────────────────────────────────────────────────────────────────

# Rolling window for per-node event accumulation (seconds).
# Events older than this are purged before escalation rules are evaluated.
TEMPORAL_WINDOW_SECONDS = 300   # 5 minutes — configurable

# EMA trajectory persistence threshold.
# K consecutive readings above 2.0σ → MEDIUM floor.
# K consecutive readings above 2.5σ → HIGH floor.
K_CONSECUTIVE_READINGS  = 2     # configurable


# ─────────────────────────────────────────────────────────────────────────────
# FEDERATED LEARNING (Flower + Krum Aggregation)
# ─────────────────────────────────────────────────────────────────────────────

FL_SERVER_ADDRESS   = "localhost:8080"
FL_NUM_ROUNDS       = 3          # Federation rounds; final round hash is minted
FL_MIN_CLIENTS      = 5          # All 5 org clients contribute each round
FL_MIN_AVAILABLE    = 5          # All 5 orgs must be present before round 1
FL_LOCAL_EPOCHS     = 3          # Local AE training epochs per client per FL round (reduced for fast simulation)

# Krum: number of clients to select per round (must be ≤ total clients - 2)
# Krum drops the m clients whose weight updates are most distant from the median.
KRUM_NUM_TO_SELECT  = 2          # Select 2 from 3 mock clients (drops 1 straggler)

# Straggler policy: if a client doesn't respond within this many seconds, drop it
FL_ROUND_TIMEOUT_SEC = 30

# ─────────────────────────────────────────────────────────────────────────────
# FLTRUST ROOT DATASET SOURCE
# ─────────────────────────────────────────────────────────────────────────────

# "real"      → load a benign partition from NF-UNSW-NB15-v3 as the server's
#               trusted root dataset (best accuracy — recommended for paper).
# "synthetic" → fall back to Gaussian samples when the dataset is unavailable.
# The server always falls back to synthetic if the real load fails, so this is
# a soft preference, not a hard requirement.
FL_ROOT_DATA_SOURCE = "real"

# ─────────────────────────────────────────────────────────────────────────────
# FLTRUST AGGREGATION (replaces Krum — Upgrade 6)
# ─────────────────────────────────────────────────────────────────────────────

# Number of synthetic benign samples the server holds as its trusted root dataset.
# These are used to train the server model by one step each round so it computes
# a reference gradient direction for cosine trust scoring of client updates.
# Range: 100–500 recommended; lower = faster, higher = more robust server gradient.
FLTRUST_ROOT_SAMPLES   = 200

# Learning rate used for the server's single-step root-dataset gradient update.
# Kept separate from AE_LEARNING_RATE so it can be tuned independently.
FLTRUST_SERVER_LR      = 1e-3

# Trust score at or below this value causes the client to be flagged as Byzantine
# in the detection log (fed into Upgrade 3).  ReLU already zeroes negatives;
# this threshold lets you also zero out near-zero trust scores from noisy clients.
FLTRUST_MIN_TRUST_SCORE = 0.0   # 0.0 = ReLU only (strict); raise to e.g. 0.05 to be stricter

# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE ENGINE — Critical Infrastructure Allowlist
# ─────────────────────────────────────────────────────────────────────────────

# Tier-1 "never hard-isolate" nodes (simulated by synthetic node IDs)
# In production these would be real IP CIDRs or hostnames.
CRITICAL_ALLOWLIST = {
    "node_0":  "Domain Controller (AD)",
    "node_1":  "Core HR Database",
    "node_2":  "Payment Gateway",
    "node_3":  "SCADA / ICS Controller",
}

# ─────────────────────────────────────────────────────────────────────────────
# ORGANISATION NETWORK MAP (canonical — referenced by fl_server, client_state)
# ─────────────────────────────────────────────────────────────────────────────

# Maps org key → simulated LAN CIDR for all 5 federation clients.
# Used by the FL simulation console and client-state display.
# Must stay consistent with client_state.ALL_CLIENTS.
ORG_NETWORK_MAP: dict = {
    "hospital":   "192.168.1.0/24",
    "bank":       "10.0.1.0/24",
    "university": "172.16.1.0/24",
    "isp":        "10.10.0.0/24",
    "retail":     "172.31.0.0/24",
}

# Confidence thresholds for the 3-tier response policy
CONFIDENCE_LOW_THRESHOLD  = 0.40   # Below this: log only
CONFIDENCE_MED_THRESHOLD  = 0.70   # Below this: throttle + HITL
# Above MED_THRESHOLD → full isolation for non-critical nodes

# ─────────────────────────────────────────────────────────────────────────────
# BLOCKCHAIN / GANACHE (Immutable Audit Log)
# ─────────────────────────────────────────────────────────────────────────────

GANACHE_URL              = "http://127.0.0.1:7545"
CONTRACT_ADDRESS_FILE    = str(MODELS_DIR / "contract_address.txt")
CONTRACT_ABI_FILE        = str(CONTRACTS_DIR / "ModelRegistry.abi")

# If Ganache is not running, AURA falls back to local SHA-256 file logging
BLOCKCHAIN_FALLBACK_LOG  = str(LOGS_DIR / "blockchain_fallback.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# ISOLATION FOREST (Baseline Sanitisation)
# ─────────────────────────────────────────────────────────────────────────────

# Contamination: expected fraction of mislabelled / poisoned rows in the
# "normal" training split.  NF-UNSW-NB15-v3 is ~94.6% benign but we
# apply a small contamination rate defensively.
IF_CONTAMINATION = 0.02   # 2 % — removes extreme statistical outliers

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

DASHBOARD_REFRESH_INTERVAL_MS = 1500   # Streamlit auto-refresh period
ALERT_LOG_FILE = str(LOGS_DIR / "aura_alerts.jsonl")
EVENT_LOG_FILE = str(LOGS_DIR / "aura_events.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# API SECURITY BLOCKLIST (custom injection endpoint)
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that, if found in a user-submitted injection script, cause the
# request to be rejected with HTTP 400.  Extend this list for production.
# This centralises the filter so it is easily auditable and not buried in
# api_server.py business logic.
SECURITY_BLOCKLIST: list = [
    "os.system",
    "os.popen",
    "subprocess",
    "import os",
    "import sys",
    "import subprocess",
    "__import__",
    "exec(",
    "eval(",
    "open(",
    "socket",
    "shutil",
    "pathlib",
    "requests",
    "urllib",
]

# ─────────────────────────────────────────────────────────────────────────────
# MITM SIMULATION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Standard deviation of the Gaussian noise injected into model weights during
# a simulated Man-in-the-Middle attack.  A small value is sufficient to change
# the SHA-256 hash (even 1-bit flip is detected) while remaining numerically
# similar enough to be a plausible interception scenario.
# Range: 0.001–0.05  |  Lower = subtler tampering, harder to detect by eye.
MITM_NOISE_STD: float = 0.01

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM INJECTION CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# NetFlow feature index lookup — used by _run_inject_inference() in api_server.py
# to resolve feature names to their column positions without hardcoding integers.
# Derived from NF-UNSW-NB15-v3 schema (IPs, ports, timestamps, Label, Attack stripped).
#
# Feature column order (47 features, 0-indexed):
#   0:  PROTOCOL                    1:  L7_PROTO
#   2:  IN_BYTES                    3:  IN_PKTS
#   4:  OUT_BYTES                   5:  OUT_PKTS
#   6:  TCP_FLAGS                   7:  CLIENT_TCP_FLAGS
#   8:  SERVER_TCP_FLAGS            9:  FLOW_DURATION_MILLISECONDS
#  10:  DURATION_IN                11:  DURATION_OUT
#  12:  MIN_TTL                    13:  MAX_TTL
#  14:  LONGEST_FLOW_PKT           15:  SHORTEST_FLOW_PKT
#  16:  MIN_IP_PKT_LEN             17:  MAX_IP_PKT_LEN
#  18:  SRC_TO_DST_SECOND_BYTES    19:  DST_TO_SRC_SECOND_BYTES
#  20:  RETRANSMITTED_IN_BYTES     21:  RETRANSMITTED_IN_PKTS
#  22:  RETRANSMITTED_OUT_BYTES    23:  RETRANSMITTED_OUT_PKTS
#  24:  SRC_TO_DST_AVG_THROUGHPUT  25:  DST_TO_SRC_AVG_THROUGHPUT
#  26:  NUM_PKTS_UP_TO_128_BYTES   27:  NUM_PKTS_128_TO_256_BYTES
#  28:  NUM_PKTS_256_TO_512_BYTES  29:  NUM_PKTS_512_TO_1024_BYTES
#  30:  NUM_PKTS_1024_TO_1514_BYTES
#  31:  TCP_WIN_MAX_IN             32:  TCP_WIN_MAX_OUT
#  33:  ICMP_TYPE                  34:  ICMP_IPV4_TYPE
#  35:  DNS_QUERY_ID               36:  DNS_QUERY_TYPE
#  37:  DNS_TTL_ANSWER             38:  FTP_COMMAND_RET_CODE
#  39:  SRC_TO_DST_IAT_MIN         40:  SRC_TO_DST_IAT_MAX
#  41:  SRC_TO_DST_IAT_AVG         42:  SRC_TO_DST_IAT_STDDEV
#  43:  DST_TO_SRC_IAT_MIN         44:  DST_TO_SRC_IAT_MAX
#  45:  DST_TO_SRC_IAT_AVG         46:  DST_TO_SRC_IAT_STDDEV
FEATURE_INDEX_MAP: dict = {
    "protocol":                 0,
    "l7_proto":                 1,
    "in_bytes":                 2,
    "in_pkts":                  3,
    "out_bytes":                4,
    "out_pkts":                 5,
    "tcp_flags":                6,
    "client_tcp_flags":         7,
    "server_tcp_flags":         8,
    "flow_duration":            9,
    "duration_in":              10,
    "duration_out":             11,
    "min_ttl":                  12,
    "max_ttl":                  13,
    "longest_flow_pkt":         14,
    "shortest_flow_pkt":        15,
    "min_ip_pkt_len":           16,
    "max_ip_pkt_len":           17,
    "src_to_dst_second_bytes":  18,
    "dst_to_src_second_bytes":  19,
    "retransmitted_in_bytes":   20,
    "retransmitted_in_pkts":    21,
    "retransmitted_out_bytes":  22,
    "retransmitted_out_pkts":   23,
    "src_to_dst_avg_throughput": 24,
    "dst_to_src_avg_throughput": 25,
    "num_pkts_up_to_128_bytes": 26,
    "num_pkts_128_to_256_bytes": 27,
    "num_pkts_256_to_512_bytes": 28,
    "num_pkts_512_to_1024_bytes": 29,
    "num_pkts_1024_to_1514_bytes": 30,
    "tcp_win_max_in":           31,
    "tcp_win_max_out":          32,
    "icmp_type":                33,
    "icmp_ipv4_type":           34,
    "dns_query_id":             35,
    "dns_query_type":           36,
    "dns_ttl_answer":           37,
    "ftp_command_ret_code":     38,
    "src_to_dst_iat_min":       39,
    "src_to_dst_iat_max":       40,
    "src_to_dst_iat_avg":       41,
    "src_to_dst_iat_stddev":    42,
    "dst_to_src_iat_min":       43,
    "dst_to_src_iat_max":       44,
    "dst_to_src_iat_avg":       45,
    "dst_to_src_iat_stddev":    46,
}

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC AE THRESHOLD LOADER
# ─────────────────────────────────────────────────────────────────────────────
#
# The AE threshold MUST be derived from the benign traffic distribution, NOT a
# magic number. Specifically it is the EMA-adjusted UCL (μ + 3σ upper control
# limit) over benign reconstruction errors — see Section 3.1 in the paper.
#
# Resolution order:
#   1. logs/calibration_results.json  (written by calibrate_thresholds.py)
#   2. Loud WARNING + magic-number sentinel fallback
#
# Run `python calibrate_thresholds.py` once after training to produce the JSON.
# ─────────────────────────────────────────────────────────────────────────────

_CALIB_JSON_PATH = LOGS_DIR / "calibration_results.json"


def load_ae_thresholds() -> tuple[float, float]:
    """
    Load MSE_THRESHOLD_HIGH and MSE_THRESHOLD_MEDIUM from calibration_results.json.

    Returns
    -------
    (threshold_high, threshold_medium) derived from the benign MSE distribution.

    Raises FileNotFoundError if calibration_results.json is missing, or ValueError
    if the file cannot be parsed. This enforces that no hardcoded threshold values
    are used in the research project.
    """
    if not _CALIB_JSON_PATH.exists():
        msg = (
            "[CONFIG] ❌ logs/calibration_results.json NOT FOUND. "
            "Dynamic AE thresholds are required for this research project. "
            "Run: `python calibrate_thresholds.py` to generate them."
        )
        _cfg_log.error(msg)
        raise FileNotFoundError(msg)

    try:
        data = json.loads(_CALIB_JSON_PATH.read_text())
        high   = float(data["recommended_MSE_THRESHOLD_HIGH"])
        medium = float(data["recommended_MSE_THRESHOLD_MEDIUM"])

        # Sanity-check the P90/P99 collapse issue documented in the paper.
        # If the gap is < 0.002 the three-tier severity system collapses.
        p90 = data.get("p90", medium)
        p99 = data.get("p99", high)
        if abs(p99 - p90) < 0.002:
            _cfg_log.warning(
                "[CONFIG] ⚠️  AE threshold collapse detected: "
                f"P90={p90:.6f}  P99={p99:.6f}  gap={abs(p99-p90):.6f} < 0.002. "
                "Three-tier severity system is functionally degraded. "
                "Consider the parallel fusion architecture (AE + GNN scores combined) "
                "as documented in the Tier 2.5 experiment."
            )

        _cfg_log.info(
            f"[CONFIG] AE thresholds loaded from calibration JSON — "
            f"HIGH={high:.6f}  MEDIUM={medium:.6f}"
        )
        return high, medium

    except Exception as exc:
        msg = (
            f"[CONFIG] ❌ Failed to parse calibration_results.json: {exc}. "
            "Run: `python calibrate_thresholds.py` to regenerate it."
        )
        _cfg_log.error(msg)
        raise ValueError(msg) from exc


# Resolved at import time — every consumer (api_server, dashboard, dashboard_service)
# picks up the calibrated value automatically once calibration_results.json exists.
_ae_thresh_high, _ae_thresh_medium = load_ae_thresholds()

MSE_THRESHOLD_HIGH   = _ae_thresh_high    # EMA-UCL P99 of benign MSE distribution
MSE_THRESHOLD_MEDIUM = _ae_thresh_medium  # EMA-UCL P90 of benign MSE distribution

# ─────────────────────────────────────────────────────────────────────────────
# DATA-DRIVEN ATTACK CORRUPTION PROFILES
# ─────────────────────────────────────────────────────────────────────────────
#
# Each profile maps feature keys → (lo, hi) normalised perturbation ranges.
# These ranges MUST come from the actual NF-UNSW-NB15-v3 dataset, not from
# manually crafted guesses.  The canonical source is:
#   saved_models/attack_class_stats.json
# which is produced by scripts/train_explainer.py and stores per-class
# p05 / p95 percentiles for all 47 features.
#
# Resolution order:
#   1. saved_models/attack_class_stats.json  — real dataset percentiles
#   2. Hardcoded sentinel profiles           — fallback with loud WARNING
# ─────────────────────────────────────────────────────────────────────────────

# Map: config profile key → Attack column value in NF-UNSW-NB15-v3.csv
# Dataset Attack column uses: 'DoS', 'Exploits', 'Fuzzers', 'Reconnaissance',
# 'Backdoor', 'Generic', 'Shellcode', 'Analysis', 'Worms', 'Benign'
_ATTACK_STATS_KEY_MAP: dict = {
    "ddos":           "ddos",
    "lateral":        "lateral",
    "exfil":          "exfil",
    "port_scan":      "portscan",
    "web":            "web",
    "exploits":       "exploits",
    "fuzzers":        "fuzzers",
    "reconnaissance": "portscan",   # mapped to same class in stats JSON
    "backdoor":       "backdoor",
    "custom":         None,          # no real-data class; stays as sentinel
}

# Sentinel profiles — used ONLY when attack_class_stats.json is absent.
# These are hand-approximated and MUST NOT be cited in the paper as ground truth.
_SENTINEL_ATTACK_PROFILES: dict = {
    "ddos": {
        "in_pkts":                  (0.90, 0.99),
        "out_pkts":                 (0.85, 0.99),
        "src_to_dst_second_bytes":  (0.88, 0.99),
        "src_to_dst_iat_avg":       (0.00, 0.03),
        "src_to_dst_iat_stddev":    (0.00, 0.02),
        "tcp_flags":                (0.80, 0.99),
        "flow_duration":            (0.00, 0.05),
    },
    "lateral": {
        "flow_duration":            (0.50, 0.75),
        "in_pkts":                  (0.50, 0.65),
        "src_to_dst_iat_stddev":    (0.75, 0.95),
        "dst_to_src_iat_stddev":    (0.75, 0.95),
        "duration_in":              (0.80, 0.98),
        "duration_out":             (0.03, 0.10),
        "client_tcp_flags":         (0.60, 0.80),
    },
    "exfil": {
        "in_bytes":                 (0.88, 0.99),
        "out_bytes":                (0.00, 0.06),
        "src_to_dst_second_bytes":  (0.85, 0.99),
        "dst_to_src_second_bytes":  (0.00, 0.04),
        "src_to_dst_iat_avg":       (0.40, 0.55),
        "src_to_dst_iat_stddev":    (0.00, 0.03),
        "flow_duration":            (0.80, 0.98),
    },
    "port_scan": {
        "flow_duration":            (0.00, 0.03),
        "in_bytes":                 (0.00, 0.04),
        "out_bytes":                (0.00, 0.03),
        "tcp_flags":                (0.80, 0.99),
        "client_tcp_flags":         (0.70, 0.90),
        "src_to_dst_second_bytes":  (0.05, 0.15),
    },
    "web": {
        "in_bytes":                 (0.80, 0.95),
        "out_bytes":                (0.20, 0.30),
        "client_tcp_flags":         (0.85, 0.98),
        "server_tcp_flags":         (0.85, 0.98),
        "flow_duration":            (0.05, 0.12),
        "src_to_dst_iat_avg":       (0.02, 0.08),
    },
    "exploits": {
        "in_bytes":                 (0.70, 0.95),
        "longest_flow_pkt":         (0.85, 0.99),
        "tcp_flags":                (0.60, 0.85),
        "flow_duration":            (0.10, 0.30),
        "retransmitted_in_bytes":   (0.40, 0.70),
        "src_to_dst_iat_stddev":    (0.50, 0.80),
    },
    "fuzzers": {
        "in_bytes":                 (0.60, 0.90),
        "in_pkts":                  (0.70, 0.95),
        "longest_flow_pkt":         (0.70, 0.99),
        "shortest_flow_pkt":        (0.00, 0.05),
        "src_to_dst_iat_stddev":    (0.80, 0.99),
        "flow_duration":            (0.10, 0.40),
        "protocol":                 (0.80, 0.99),
    },
    "reconnaissance": {
        "flow_duration":            (0.00, 0.05),
        "in_bytes":                 (0.00, 0.08),
        "out_bytes":                (0.00, 0.06),
        "tcp_flags":                (0.70, 0.95),
        "num_pkts_up_to_128_bytes": (0.80, 0.99),
        "src_to_dst_second_bytes":  (0.05, 0.15),
    },
    "backdoor": {
        "flow_duration":            (0.60, 0.90),
        "in_bytes":                 (0.40, 0.65),
        "out_bytes":                (0.40, 0.65),
        "src_to_dst_iat_avg":       (0.50, 0.70),
        "src_to_dst_iat_stddev":    (0.00, 0.05),
        "dst_to_src_iat_avg":       (0.50, 0.70),
        "dst_to_src_iat_stddev":    (0.00, 0.05),
    },
    "custom": {
        # No real-data class for 'custom' — sentinel only.
        "longest_flow_pkt":         (0.85, 0.99),
        "shortest_flow_pkt":        (0.85, 0.99),
        "max_ip_pkt_len":           (0.85, 0.99),
        "min_ip_pkt_len":           (0.85, 0.99),
        "src_to_dst_iat_avg":       (0.02, 0.05),
        "src_to_dst_iat_stddev":    (0.92, 0.99),
        "protocol":                 (0.90, 0.99),
    },
}


def load_attack_corruption_profiles() -> dict:
    """
    Load attack corruption profiles derived from the NF-UNSW-NB15-v3 dataset.

    Reads saved_models/attack_class_stats.json (produced by
    scripts/train_explainer.py) and converts per-class (p05, p95) percentiles
    into the {feature_key: (lo, hi)} format expected by _run_inject_inference().

    The inverse-feature map (dataset column index → config key) is constructed
    from FEATURE_INDEX_MAP so there is a single source of truth for column
    ordering.

    Returns the sentinel profiles (with WARNING) if the JSON is absent.
    """
    stats_path = MODELS_DIR / "attack_class_stats.json"
    if not stats_path.exists():
        _cfg_log.warning(
            "[CONFIG] ⚠️  saved_models/attack_class_stats.json NOT FOUND. "
            "Using sentinel ATTACK_CORRUPTION_PROFILES — these are NOT data-derived. "
            "Run: python scripts/train_explainer.py"
        )
        return _SENTINEL_ATTACK_PROFILES

    try:
        stats_data = json.loads(stats_path.read_text())
    except Exception as exc:
        _cfg_log.error(
            f"[CONFIG] Failed to parse attack_class_stats.json: {exc}. "
            "Falling back to sentinel profiles."
        )
        return _SENTINEL_ATTACK_PROFILES

    # Build index → feature-key reverse map from FEATURE_INDEX_MAP
    idx_to_key: dict[int, str] = {v: k for k, v in FEATURE_INDEX_MAP.items()}

    profiles: dict = {}
    for profile_key, stats_key in _ATTACK_STATS_KEY_MAP.items():
        if stats_key is None or stats_key not in stats_data:
            # Keep the sentinel for profiles without a real-data class
            profiles[profile_key] = _SENTINEL_ATTACK_PROFILES.get(profile_key, {})
            continue

        cls_stats = stats_data[stats_key]
        p05 = cls_stats.get("p05", [])
        p95 = cls_stats.get("p95", [])

        if not p05 or not p95:
            _cfg_log.warning(
                f"[CONFIG] attack_class_stats.json missing p05/p95 for class '{stats_key}'. "
                f"Using sentinel profile for '{profile_key}'."
            )
            profiles[profile_key] = _SENTINEL_ATTACK_PROFILES.get(profile_key, {})
            continue

        # Convert: only include features present in FEATURE_INDEX_MAP
        feature_profile: dict = {}
        for feat_idx, feat_key in idx_to_key.items():
            if feat_idx < len(p05) and feat_idx < len(p95):
                lo = round(float(p05[feat_idx]), 6)
                hi = round(float(p95[feat_idx]), 6)
                # Only include features where the attack class actually differs
                # from the trivial [0,1] range (skip uninformative features)
                if hi > lo:
                    feature_profile[feat_key] = (lo, hi)

        profiles[profile_key] = feature_profile
        _cfg_log.info(
            f"[CONFIG] Loaded dataset-derived profile for '{profile_key}' "
            f"({stats_key}): {len(feature_profile)} features from NF-UNSW-NB15-v3."
        )

    # Always include the sentinel 'custom' profile unchanged
    if "custom" not in profiles:
        profiles["custom"] = _SENTINEL_ATTACK_PROFILES["custom"]

    return profiles


# Resolved at import time — derived from NF-UNSW-NB15-v3 via attack_class_stats.json.
# Falls back to sentinel profiles with WARNING if the JSON is absent.
ATTACK_CORRUPTION_PROFILES: dict = load_attack_corruption_profiles()
