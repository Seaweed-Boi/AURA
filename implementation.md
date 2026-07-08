# AURA — Implementation Ground Truth
### For: AI Models, Code Agents, and New Contributors

> **Directive for AI agents:** This document is the authoritative record of _what has been built_, _why decisions were made_, and _what constraints must be preserved_. Read this file in its entirety before touching any code. Update the relevant section(s) whenever you make a major change — no exceptions.

---

## Table of Contents
1. [System Purpose & Scope](#1-system-purpose--scope)
2. [Tech Stack & Portability Constraints](#2-tech-stack--portability-constraints)
3. [Project File Map](#3-project-file-map)
4. [Data Pipeline (data_loader.py)](#4-data-pipeline-data_loaderpy)
5. [ML Models (models.py)](#5-ml-models-modelspy)
6. [Inference & Thresholding (detector.py)](#6-inference--thresholding-detectorpy)
7. [Explainability (ae_explainer.py)](#7-explainability-ae_explainerpy)
8. [Federated Learning (fl_server.py, fl_client.py)](#8-federated-learning-fl_serverpy-fl_clientpy)
9. [Response Engine (response_engine.py)](#9-response-engine-response_enginepy)
10. [Blockchain Audit (blockchain.py, contracts/)](#10-blockchain-audit-blockchainpy-contracts)
11. [Training Pipeline (train.py)](#11-training-pipeline-trainpy)
12. [Entry Points & Scripts](#12-entry-points--scripts)
13. [Configuration (config.py)](#13-configuration-configpy)
14. [Key Invariants — Do Not Break](#14-key-invariants--do-not-break)
15. [Intended Future Work](#15-intended-future-work)
16. [Changelog](#16-changelog)

---

## 1. System Purpose & Scope

**AURA (Adaptive Unified Response Architecture)** is a zero-day network intrusion detection system designed for the NEXJEM Hackathon 2026. It is built for organisations (hospitals, banks, universities) that share a federated model without sharing raw traffic.

### What it does (end-to-end)
1. Ingests raw NetFlow statistics from the **CICIDS2017** dataset.
2. Detects anomalies at two layers: statistical (autoencoder MSE) + topological (graph neural network).
3. Fuses scores dynamically using EMA-calibrated thresholding.
4. Triggers tiered automated responses (log / throttle / isolate) while protecting critical infrastructure.
5. Federates the model across organisations using Flower + Krum aggregation for Byzantine robustness.
6. Immutably records model checksums on an Ethereum blockchain (Ganache/local fallback).

### What it does NOT do
- It does not perform raw packet capture; it works on pre-processed NetFlow statistical features.
- It does not use real IP addresses; topology is **synthetically reconstructed** from port heuristics (the CICIDS2017 MachineLearningCSV format strips IPs).
- The iptables/tc commands are **simulated** on Windows; in production they run on Linux.
- Blockchain deployment is via a local Ganache development node; production would use a permissioned chain.

---

## 2. Tech Stack & Portability Constraints

| Layer | Technology | Version / Notes |
|---|---|---|
| ML Framework | **PyTorch** | `2.x`, CPU-only by default |
| GNN | **Manual GraphSAGE** via `torch.scatter_add_` | **No `torch_geometric`** — by design |
| Federated Learning | **Flower (`flwr`)** | gRPC + in-process simulation mode |
| Anomaly Detection | **scikit-learn** `IsolationForest` + custom EMA | |
| Blockchain | **Web3.py** + **Solidity 0.8.19** | Ganache for local dev; JSONL fallback |
| Dashboard | **Streamlit** + **Plotly** | |
| Data Processing | **pandas**, **numpy**, **networkx** | |
| Config | `config.py` at root | All hyperparameters live here |

> **Why no `torch_geometric`?** GraphSAGE is implemented manually via `torch.scatter_add_` in `aura/models.py`. This was a deliberate portability decision — `torch_geometric` has complex CUDA/C++ build requirements that fail on many hackathon laptops. **Do not refactor to use `torch_geometric` without explicit approval**, as this will break the install story.

---

## 3. Project File Map

```
TRINETRA---NEXJEM/
├── config.py                    ← ALL hyperparameters and paths (single source of truth)
│
├── aura/                        ← Core library
│   ├── __init__.py
│   ├── data_loader.py           ← CICIDS2017 ingestion, IsolationForest sanitisation, TTL graph builder
│   ├── models.py                ← FlowAutoencoder (L1) + AuraSTGNN/SAGEConv (L2) + AURAModelBundle
│   ├── detector.py              ← EMA threshold tracker + AURAInferenceEngine (L1→L2 cascade)
│   ├── ae_explainer.py          ← Attack classification via cosine similarity on AE residuals
│   ├── response_engine.py       ← 3-tier policy engine (LOG/THROTTLE/ISOLATE), HITL escalation
│   ├── fl_client.py             ← Flower FL client (local training + mock clients for simulation)
│   ├── fl_server.py             ← Krum aggregation strategy + straggler policy + simulation runner
│   ├── blockchain.py            ← Web3/Ganache blockchain logger + JSONL fallback
│   └── attack_injector.py       ← 5 red-team attack profiles for dashboard demo
│
├── contracts/
│   └── ModelRegistry.sol        ← Solidity 0.8.19 smart contract (registerModel, getHash, verifyHash)
│
├── train.py                     ← Two-phase training: AE on benign data, STGNN on attack graphs
├── run.py                       ← CLI launcher: train | dashboard | test | demo subcommands
├── dashboard.py                 ← Streamlit live demo dashboard
├── fl_server_dashboard.py       ← Streamlit page specifically for FL server monitoring
├── run_fl.py                    ← Standalone FL launch script
├── run_federation_networked.py  ← Multi-process networked FL launcher
├── run_orgs.ps1                 ← PowerShell: starts 3 org processes simultaneously
├── verify_chain.py              ← Verifies blockchain hash integrity post-federation
├── inspect_csv.py               ← Quick CSV inspection utility script
└── requirements.txt
```

---

## 4. Data Pipeline (`data_loader.py`)

### Class: `CICIDSDataLoader`

**Dataset:** CICIDS2017 `MachineLearningCSV` variant — 78 NetFlow statistical features per row. **The IP columns are stripped** by the dataset provider; we never have real src/dst IPs.

### Processing Chain (in order)

```
Raw CSV  →  strip column whitespace  →  drop Inf/NaN (ffill→bfill→dropna)
         →  split into BENIGN vs ATTACK rows
         →  BENIGN only: IsolationForest sanitisation (contamination=0.02)
         →  MinMaxScaler fitted on sanitised BENIGN (NOT on full data—see why below)
         →  Rolling window of WINDOW_SIZE=60 rows per graph snapshot
         →  Synthetic node assignment (port-bucket heuristic, no real IPs)
         →  TTL edge decay (EDGE_TTL_WINDOWS=3 windows before edge pruned)
         →  Node feature = mean of all incident edge features
         →  yield (graph_dict, label_tensor)
```

### Why scaler is fitted on BENIGN only
Fitting the scaler on mixed data (including attacks) causes extreme attack feature values (e.g. DDoS byte floods) to compress the normal traffic range, reducing sensitivity to subtle anomalies. The autoencoder must see "normal" in its trained range.

### Synthetic Node Mapping (`_assign_synthetic_nodes`)
Since IPs are stripped, topology is reconstructed as:
```python
src_id = row_index % NUM_SYNTHETIC_NODES          # = 20 nodes
dst_id = (row_index + port_bucket + 10) % 20
```
Port buckets: HTTP=1, HTTPS=2, SSH=3, DNS=4, other=0. Self-loops are displaced by +1. This is honest as "simulated topology" and sufficient for GNN message-passing.

### TTL Edge Decay (`TTLEdgeTracker`)
Each `(src, dst)` edge has a TTL counter. Active edges reset to `EDGE_TTL_WINDOWS=3`. Dormant edges decrement each window tick. At TTL=0, the edge is pruned from the graph. This prevents stale "trusted" edges from masking reactivated lateral movement paths.

### Output Format
```python
graph_dict = {
    "x":          FloatTensor[N=20, F=78],   # Per-node mean feature vectors
    "edge_index": LongTensor[2, E],           # COO sparse adjacency (post-TTL)
    "edge_attr":  FloatTensor[E, F=78],       # Per-edge (flow) features
    "ttl_state":  dict,                       # {(src,dst): ttl_remaining} for UI
    "window_id":  str,                        # "filename:wN" for tracing
}
label_tensor = LongTensor[E]   # 0=benign, 1=attack (per-edge)
```

### CSV File Order
```python
CSV_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",          # Pure BENIGN — used for AE training baseline
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]
```
Monday CSV is pure benign and is the scaler-fit target. All others contain attacks.

---

## 5. ML Models (`models.py`)

### Layer 1: `FlowAutoencoder` — Statistical Tripwire

**Input:** Edge (flow) feature vectors `[E, F=78]`  
**Architecture:**
```
Encoder: F → 64 → 32 → Z=16      (Dropout 0.2, ReLU, no activation on bottleneck)
Decoder: Z=16 → 32 → 64 → F      (Dropout 0.2, ReLU, Sigmoid on output)
```
**Anomaly score:** `per-sample MSE = mean((x - x_hat)^2, dim=1)`  
**Loss (training):**
- Phase 1 (MSE only): `L = MSE(x, x_hat)`
- Phase 2 (MSE + Contrastive): `L = MSE + 0.1 * relu(margin - ||z_pos - z_neg||₂)`

The contrastive term pushes attack latents away from the normal manifold, sharpening detection boundaries.

**Key method:** `ae.explain_features(x)` returns `[F]` mean absolute residual per feature — used by the explainer to infer attack category.

---

### Layer 2: `AuraSTGNN` — Contextual Validator

**Input:** Node feature matrix `[N=20, F=78]` from `_build_node_features` + `edge_index [2, E]`  
**Architecture:**
```
SAGEConv(78 → 64) → Dropout(0.3) → SAGEConv(64 → 32) → Linear(32→16) → ReLU → Linear(16→1) → Sigmoid
```
**Output:** Per-node anomaly probability `[N]` ∈ (0, 1)

**Why GraphSAGE (inductive), not GCN (transductive)?**
- GCN requires all nodes present at training time. New IP/device = crash.
- GraphSAGE learns an *aggregation function* — any new node with features can be embedded without retraining.
- Critical in a dynamic network where devices are added/removed continuously.

**`SAGEConv` forward (per layer):**
```
h_v' = LayerNorm(ReLU(W_self · h_v  +  W_neigh · mean_{u ∈ N(v)}(h_u)  + b))
```
Implemented using `torch.scatter_add_` (no `torch_geometric`). W_self and W_neigh are kept separate (not concatenated) to independently weight self vs neighbour information.

---

### `AURAModelBundle`
Wrapper holding both `FlowAutoencoder` and `AuraSTGNN` as a single `nn.Module`. Used by the FL framework so both models are always synchronised across federation rounds.

**Save paths (from config):**
- `saved_models/autoencoder_best.pth` — best val-loss AE checkpoint during training
- `saved_models/stgnn_trained.pth` — final STGNN weights
- `saved_models/aura_bundle.pth` — full AURAModelBundle state dict
- `saved_models/scaler.joblib` — fitted MinMaxScaler (required at inference)

---

## 6. Inference & Thresholding (`detector.py`)

### `EMAThresholdTracker`

Maintains an online Exponential Moving Average of batch-level MSE losses.

```
EMA_mean_t  = α × loss_t  +  (1−α) × EMA_mean_{t−1}          where α = EMA_ALPHA = 0.05
EMA_var_t   = (1−α) × (EMA_var_{t−1} + α × (loss_t − EMA_mean_t)²)
threshold_t = EMA_mean_t + σ_mult × sqrt(EMA_var_t)           where σ_mult = 3.0
```

- Alert is **suppressed for the first 50 batches** (`EMA_WARMUP_BATCHES`) — cold-start guard.
- This mirrors statistical process control (SPC) control charts.

**Why EMA over static thresholds?**
Network behaviour drifts (firmware update → universal packet size change). A static threshold would flood analysts with false positives or miss real attacks. EMA adapts continuously.

---

### `AURAInferenceEngine`

The stateful inference orchestrator. Must be instantiated once per session and reused across all graph windows (EMA state must persist).

**Processing flow per window:**
```
graph_dict
    │
    ▼
[L1] FlowAutoencoder.anomaly_score(edge_attr) → batch_mse
    │
    ├─ is_anomalous? (EMA threshold check BEFORE update)
    │       │
    │       │  YES:
    │       ▼
    │   ae_explainer.explain_ae(residuals) → attack classification
    │       │
    │       ▼
    │   [L2] AuraSTGNN.topology_anomaly_score(x, edge_index)
    │       │
    │       ▼
    │   triggered_nodes = [nodes with GNN score > 0.60]
    │
    ├─ EMA.update(batch_mse) (AFTER check, updates history)
    │
    ▼
fuse_scores(ae_score, ae_threshold, gnn_scores)
    →  confidence ∈ [0, 1]
    →  severity: NORMAL / LOW / MEDIUM / HIGH
    →  AnomalyEvent (structured, JSON-serialisable)
    →  persisted to logs/aura_alerts.jsonl
```

**L2 is conditional:** Only invoked when L1 triggers. This is a cost-saving cascade — GNN inference is skipped on quiet networks.

**Severity classification:**
| Condition | Severity |
|---|---|
| L1 not triggered | NORMAL |
| confidence < 0.40 | LOW |
| confidence < 0.70 OR no GNN nodes flagged | MEDIUM |
| confidence ≥ 0.70 AND ≥1 GNN node flagged | HIGH |

**Score fusion:**
```python
ae_contrib  = min(1.0, ae_score / ae_threshold) × 0.55
gnn_contrib = max_gnn_node_score × 0.45
confidence  = (ae_contrib + gnn_contrib) / 1.0
```
If L2 was not invoked (GNN scores = None), L1 carries 100% weight.

---

## 7. Explainability (`ae_explainer.py`)

### Purpose
When Layer 1 fires, the explainer answers: *which features caused the anomaly, and what attack does this look like?*

### `explain_ae(residuals: np.ndarray[78]) → dict`
1. **Top-K features:** Sorts `|x - x_hat|` descending, maps index → human-readable CICIDS feature name.
2. **Group residuals:** Aggregates residuals across 7 semantic groups (Volume, Bandwidth, Timing/IAT, TCP Flags, Idle/Active, Bulk Transfer, Window/Segment).
3. **Attack signature matching:** Cosine similarity between the normalised residual vector and 5 pre-defined sparse attack signature vectors.

**Attack signatures (based on `attack_injector.py` profiles + CICIDS2017 taxonomy):**
| Attack | Primary Features |
|---|---|
| DDoS | Flow Packets/s, SYN flags, near-zero IAT |
| Port Scan | RST flags, very short Flow Duration, minimal bytes |
| Lateral Movement | Flow IAT Std (jitter), Idle Mean (beaconing) |
| Data Exfiltration | Total Fwd Bytes >> Bwd Bytes, sustained duration |
| Web Attack | Fwd PSH Flags, PSH Flag Count, large fwd payload |

If `max cosine similarity < 0.30`, the result is labelled `"Unknown Anomaly"`.

Output dict keys: `top_features`, `group_residuals`, `inferred_attack`, `match_score`, `explanation` (icon, summary, detail, why_high).

---

## 8. Federated Learning (`fl_server.py`, `fl_client.py`)

### Federation Setup
- **Framework:** Flower (`flwr`)
- **Clients:** 3 organisations — `hospital` (192.168.1.0/24), `bank` (10.0.1.0/24), `university` (172.16.1.0/24)
- **Rounds:** 3 (`FL_NUM_ROUNDS`)
- **Server address:** `localhost:8080`
- **Min clients to start:** 3
- **Round timeout:** 30 seconds

### Krum Aggregation (`krum_select`, `krum_aggregate`)

**Why Krum?** Standard FedAvg is vulnerable to model poisoning — a compromised client pushes a malicious weight update that globally degrades the model. Krum (Blanchard et al., 2017) is mathematically proven Byzantine-fault tolerant.

**Algorithm:**
1. Flatten each client's weight update to 1D vector.
2. Compute pairwise squared Euclidean distances.
3. For each client `i`: Krum score = sum of `k` smallest distances to others (`k = n - num_to_select - 2`).
4. Select `KRUM_NUM_TO_SELECT=2` clients with the **lowest** scores (most central = most honest).
5. Mean of selected updates → new global model.

**Byzantine guarantee:** With `n=3`, `num_to_select=2`, one poisoned client is tolerated (it gets dropped each round).

**Attack assignment:** The `bank` organisation is always the Byzantine client in simulation mode. If bank is offline, all clients are honest (no spurious Krum drop).

### Straggler Policy
If a client doesn't respond in `FL_ROUND_TIMEOUT_SEC=30s`, Flower's gRPC layer adds it to the `failures` list (not `results`). If `len(results) < min_fit_clients`, the round is **abandoned** and the previous global model is preserved. This prevents Denial-of-Service against the federation from a single dead client.

### SHA-256 Model Hashing
```python
hash = SHA256(concat all weight bytes as float32, C-contiguous)
```
Computed server-side after Krum aggregation. Blockchain mint occurs on the **final round only** (intermediate rounds are convergence steps). The hash is written to `logs/hash_registry.json` as the trusted reference — `verify_chain.py` reads this to validate.

### `KrumFedAURA` Strategy
Extends Flower's `FedAvg`, overriding only `aggregate_fit`. Maintains `_hash_history` list for dashboard display. Clears `hash_registry.json` at instantiation (fresh FL session).

### Simulation vs. Networked Mode
- **Simulation (`run_federation_simulation`):** In-process, no real gRPC sockets. Mock clients run locally. Used by dashboard.
- **Networked (`fl_server_dashboard.py`, `run_orgs.ps1`):** True multi-process. Each org runs `fl_client.py` in its own process connecting via gRPC.

---

## 9. Response Engine (`response_engine.py`)

### Design Principles
1. **Human-In-The-Loop (HITL):** The system NEVER auto-isolates Tier-1 Critical Assets. It throttles and pages an analyst.
2. **Blast-Radius Control:** Three tiers minimise collateral damage.
3. **Auditability:** Every action → `IncidentRecord` → appended to `logs/aura_events.jsonl`.

### Policy Matrix
| Severity | Critical Node (Allowlist) | Standard Node |
|---|---|---|
| LOW | LOG_ONLY | LOG_ONLY |
| MEDIUM | THROTTLE + HITL | THROTTLE + HITL |
| HIGH | THROTTLE + HITL | **ISOLATE** |

### Critical Infrastructure Allowlist (from `config.py`)
```python
CRITICAL_ALLOWLIST = {
    "node_0": "Domain Controller (AD)",
    "node_1": "Core HR Database",
    "node_2": "Payment Gateway",
    "node_3": "SCADA / ICS Controller",
}
```
> **HARD RULE:** Do not auto-isolate these nodes under any circumstances. The policy reason: isolating a Domain Controller during a potential attack is worse than the attack itself.

### Commands (simulated on Windows, real on Linux)
- **THROTTLE:** `tc qdisc add dev eth0 root handle 1: htb default 12 && tc class add ... rate 10kbps`
- **ISOLATE:** `iptables -A INPUT -s <IP> -j DROP && iptables -A OUTPUT -d <IP> -j DROP`

**Windows simulation:** If `platform.system() == "Windows"`, commands are prefixed with `[SIMULATED-TC]` / `[SIMULATED-IPTABLES]` and logged only.

### Idempotency (Dedup)
`_actioned_nodes` dict tracks `{node_id: last_action_timestamp}`. If a node is re-triggered within `_dedup_window_sec=30s`, the action is recorded as `ALREADY_ACTIONED` — prevents kernel iptables rule table overflow.

### HITL Alert
In demo: writes to `logs/aura_alerts.jsonl`. In production: PagerDuty / Slack webhook / SIEM forwarding.

---

## 10. Blockchain Audit (`blockchain.py`, `contracts/`)

### Architecture Role
This is the **Root of Trust** / **Non-Repudiation** layer. It is NOT a poisoning defence (Krum handles that). It ensures the server cannot secretly distribute a different model than what is recorded.

### `AURABlockchainLogger`
Two modes, same public interface:
```python
bc.log_model_update(model_version: str, model_hash: str) → tx_id
bc.verify_model(model_version: str, model_hash: str) → (bool, source_str)
```

**Mode selection at init:**
- **Blockchain mode:** Web3.py connects to Ganache at `localhost:7545`. Contract address loaded from `saved_models/contract_address.txt` or mock-deployed.
- **Local fallback:** Ganache unreachable → writes to `logs/blockchain_fallback.jsonl`. Same interface.

### `ModelRegistry.sol` (Solidity 0.8.19)
Three functions: `registerModel(version, bytes32_hash)`, `getHash(version)`, `verifyHash(version, bytes32_hash)`. The ABI is hardcoded in `blockchain.py` (avoids requiring Truffle/Hardhat at runtime).

### Hash verification flow (per federation round)
1. Server computes `SHA256(all model weight bytes as float32 C-contiguous)`.
2. On final round only: hash is minted to blockchain (or fallback JSONL).
3. Clients receive the global model weights, independently compute the hash, and compare against the registered hash.
4. **Match → deploy.** Mismatch → reject and alert (supply-chain attack detected).

### Two Persistent Records
- `logs/hash_registry.json` — trusted server-side reference (read by `verify_chain.py`)
- `logs/model_hashes.jsonl` OR Ganache contract — the "public" ledger

---

## 11. Training Pipeline (`train.py`)

### Two-Phase Training

**Phase 1: Autoencoder**
- Data: Monday CSV benign flows only (pure baseline)
- Split: 80% train / 20% val
- Epochs: 30 (or 5 in `--quick` mode)
- Optimizer: Adam, lr=1e-3, CosineAnnealingLR scheduler
- Gradient clipping: `max_norm=1.0`
- **Epoch 1–65%:** Pure MSE reconstruction
- **Epoch 66%–end:** MSE + Contrastive loss (synthetic negative samples via `batch + randn*0.3`)
- Saves: `saved_models/autoencoder_best.pth` (best val MSE), `saved_models/scaler.joblib`

**Phase 2: STGNN (weakly supervised)**
- Data: Monday + attack CSVs (up to 200 graph windows)
- Node labels: approximated from edge labels — any node incident to an attack edge is labelled `1`
- Loss: Binary Cross-Entropy
- Epochs: `GNN_EPOCHS=20`
- Saves: `saved_models/stgnn_trained.pth`

**Final:** Both models merged into `AURAModelBundle` → `saved_models/aura_bundle.pth`

### CLI Usage
```bash
python train.py                  # Full training (AE + STGNN)
python train.py --ae-only        # AE only (faster, for iteration)
python train.py --quick          # 5-epoch smoke test
python train.py --epochs 15      # Override epoch count
```

---

## 12. Entry Points & Scripts

| Script | Purpose |
|---|---|
| `run.py train [--quick]` | Launch `train.py` with CLI args |
| `run.py dashboard` | Start Streamlit dashboard |
| `run.py test` | Sanity test pipeline without CSV (synthetic data) |
| `run.py demo` | CLI pipeline demo (no browser required) |
| `dashboard.py` | Full Streamlit interactive demo UI |
| `fl_server_dashboard.py` | FL server monitoring Streamlit page |
| `run_fl.py` | Standalone FL launch (simulation mode) |
| `run_federation_networked.py` | Multi-process real gRPC FL launch |
| `run_orgs.ps1` | PowerShell: starts 3 org processes in parallel |
| `verify_chain.py` | Post-federation hash integrity verifier |
| `inspect_csv.py` | Quick CSV column/shape inspector utility |

---

## 13. Configuration (`config.py`)

All constants live here. **Do not hardcode values in operational scripts.**

| Constant | Value | Purpose |
|---|---|---|
| `CSV_DIR` | `./CSV's/MachineLearningCVE/` | CICIDS2017 dataset location |
| `MODELS_DIR` | `./saved_models/` | Model checkpoint directory |
| `LOGS_DIR` | `./logs/` | Alert + event logs |
| `NUM_SYNTHETIC_NODES` | 20 | Simulated IP endpoints |
| `LABEL_COL` | `" Label"` | Note leading space in raw CSV |
| `DATA_LOAD_FRACTION` | 0.30 | 30% of rows per CSV (reduce for speed) |
| `WINDOW_SIZE` | 60 | Rows per graph snapshot |
| `EDGE_TTL_WINDOWS` | 3 | Windows before edge pruned |
| `FEATURE_DIM` | 78 | NetFlow feature count |
| `LATENT_DIM` | 16 | AE bottleneck size |
| `AE_EPOCHS` | 30 | Standard training epochs |
| `AE_BATCH_SIZE` | 256 | |
| `EMA_ALPHA` | 0.05 | EMA smoothing (lower = more stable) |
| `EMA_SIGMA_MULTIPLIER` | 3.0 | 3σ = 99.7% normal coverage |
| `EMA_WARMUP_BATCHES` | 50 | Cold-start guard |
| `GNN_HIDDEN_DIM` | 64 | SAGEConv hidden size |
| `GNN_OUTPUT_DIM` | 32 | SAGEConv output embedding size |
| `FL_NUM_ROUNDS` | 3 | Federation rounds |
| `FL_MIN_CLIENTS` | 3 | Quorum for round start |
| `KRUM_NUM_TO_SELECT` | 2 | Clients kept per Krum round |
| `FL_ROUND_TIMEOUT_SEC` | 30 | Straggler hard timeout |
| `CONFIDENCE_LOW_THRESHOLD` | 0.40 | Below → LOG_ONLY |
| `CONFIDENCE_MED_THRESHOLD` | 0.70 | Below → THROTTLE; Above → ISOLATE |
| `GANACHE_URL` | `http://127.0.0.1:7545` | Local Ethereum dev node |
| `IF_CONTAMINATION` | 0.02 | IsolationForest outlier fraction |
| `DASHBOARD_REFRESH_INTERVAL_MS` | 1500 | Streamlit auto-refresh |

---

## 14. Key Invariants — Do Not Break

These are non-negotiable constraints. Violating them will break the system's core safety or portability guarantees.

### ML Architecture
- **`torch_geometric` is NOT used.** GraphSAGE is implemented via `torch.scatter_add_` in `models.py`. Do not introduce `torch_geometric` as a dependency.
- **`FEATURE_DIM=78` is fixed** by the CICIDS2017 dataset. If you ever change the feature count, you must retrain both models from scratch and update all saved `.pth` files.
- **The `AURAModelBundle` wraps both models.** Federation passes this bundle as a unit. Never federating only one sub-model without the other will desynchronise the system.

### Data Pipeline
- **Scaler is always fitted on sanitised BENIGN data only** (Monday CSV). Never fit on mixed data.
- **`IsolationForest` contamination runs on the benign split before scaler fit.** Do not remove this step — it is the poisoned-baseline defence.
- **Synthetic node mapping is deterministic** (`row_index % NUM_SYNTHETIC_NODES`). Its outputs are reproducible given the same dataframe, which allows consistent graph construction across runs.

### Response Engine
- **Critical nodes in `CRITICAL_ALLOWLIST` must NEVER be auto-isolated.** Any change to the isolation policy for critical nodes MUST go through explicit human review. A Domain Controller is more valuable online than isolated.
- **Dedup window (30 seconds) must remain.** Without it, repeated triggers create stacked iptables rules that overflow the kernel rule table.

### Federated Learning
- **Blockchain hash is minted on the FINAL round only.** Intermediate-round hashes are logged locally but not recorded on-chain. This is intentional — only the converged production model gets blockchain provenance.
- **`hash_registry.json` is cleared at the start of each FL session.** This ensures only the current session's final model hash is present. `verify_chain.py` reads this file as the trusted reference.

### Configuration
- **All numeric constants go in `config.py`.** Never hardcode thresholds, learning rates, or paths in operational modules.

---

## 15. Intended Future Work

The following improvements are acknowledged as out-of-scope for the hackathon but are the logical next steps:

1. **Full ST-GNN:** Replace the per-snapshot GNN with LSTM cells for true temporal modelling (current temporal structure is approximated by processing consecutive snapshots, not explicit recurrent state).
2. **Real IP topology:** Deploy on a network tap that provides actual src/dst IP. Remove `_assign_synthetic_nodes` and use real adjacency.
3. **Production blockchain:** Deploy `ModelRegistry.sol` to a permissioned chain (e.g. Hyperledger Besu); remove Ganache dependency.
4. **Real HITL integration:** Replace the `print()` alert with a PagerDuty/Slack/SIEM webhook.
5. **Live capture:** Integrate Zeek/Suricata NetFlow export directly instead of CICIDS2017 CSVs.
6. **Larger federation:** Scale beyond 3 mock clients to real gRPC clients across subnets.

---

## 16. Changelog

> AI agents: append a one-line entry here with `[DATE] [AUTHOR/AGENT] — <description>` every time a significant change is made to the codebase.

| Date | Author | Change |
|---|---|---|
| 2026-03-28 | Antigravity (AI) | **Upgrade 1** — Added `policy_engine.py`, `response_policy.yaml`, `scripts/{isolate,throttle,log_only}.sh`. Replaced hardcoded `tc`/`iptables` calls in `response_engine.py` with `policy_engine.execute_response()`. HITL gate enforced: isolation requires `y`; any other input auto-degrades to throttle. `pyyaml` added to `requirements.txt`. |
| 2026-03-28 | Antigravity (AI) | **Severity Engine Upgrade** — `config.py`: added `TEMPORAL_WINDOW_SECONDS=300`, `K_CONSECUTIVE_READINGS=5`. `detector.py`: (1) `EMAThresholdTracker` — added `threshold_2sigma`/`threshold_2_5sigma` properties, trajectory counters `_consecutive_above_2sigma`/`_consecutive_above_2_5sigma` updated in `update()`, exposed in `state_dict()`; (2) `AURAInferenceEngine` — EMA computed before `l1_triggered` (EMA-first order), trajectory-triggered events invoke L2 GNN, `_classify_severity()` accepts `consec_2sigma`/`consec_2_5sigma` and floors severity via trajectory, `_apply_temporal_escalation()` adds per-node sliding window accumulator with LOW×3→MEDIUM, LOW×5 or MED×3→HIGH escalation rules and HIGH-event window reset. |
| 2026-03-28 | Antigravity (AI) | **Custom Script Injection Panel** — New `api_server.py` (Flask, port 5001): `/api/nodes` returns live node registry; `/api/inject_custom` validates script (blocks `os.system`, `subprocess`, `import os`, `import sys`), logs `CUSTOM_INJECT` event to alert log, writes `pending_inject.json` for dashboard poll. `dashboard.py`: HTML/JS component inserted between attack buttons and Normal Traffic button — `<textarea id=custom-script-input>`, `<select id=custom-target-node>` (baked node list + live fetch from /api/nodes), `<button id=btn-inject-custom>` (amber border, POSTs to /api/inject_custom, inline error/success feedback, 2-second yellow flash on success, no page reload). `flask>=3.0.0` added to `requirements.txt`. |
| 2026-03-28 | Antigravity (AI) | **Custom Injection Bug Fixes** — Fix 1 (node colour bridge): `dashboard.py` now polls `logs/pending_inject.json` every rerun cycle; on a fresh entry (< 30 s) sets node state to `⚡ Evaluating…` (yellow), builds `AnomalyEvent` from MSE, drives `responder.act()` → Response Actions panel shows tier, appends `CUSTOM_INJECT` alert to Alert History. File consumed with `{}` after read. Fix 2 (AE reconstruction pass): `api_server.py` now generates anomalous NetFlow features (high packet-size variance, chaotic IAT, unusual port entropy), runs them through lazy-loaded `FlowAutoencoder`, calls `explain_ae()` for per-feature attribution, writes `logs/last_explanation.json` with node/mse/top_features/observed/baseline/timestamp, and updates `pending_inject.json` with real MSE. `dashboard.py` polls `last_explanation.json` (mtime check < 30 s) and renders amber-bordered feature table (Feature | Observed | Baseline | Squared Error bar) below the AE panel. |
| 2026-03-28 | Antigravity (AI) | **Upgrade 6 — FLTrust Aggregation** — `config.py`: added `FLTRUST_ROOT_SAMPLES=200`, `FLTRUST_SERVER_LR=1e-3`, `FLTRUST_MIN_TRUST_SCORE=0.0`. `aura/fl_server.py`: added `_build_root_dataset()` (synthetic benign root data in MinMax-normalised range), `fltrust_aggregate()` (Cao et al. 2020: server one-step gradient → cosine trust scores → ReLU → magnitude-normalised weighted aggregation → returns `(new_arrays, trust_scores, flagged_indices)`). `KrumFedAURA.aggregate_fit()` now calls `fltrust_aggregate()` instead of `krum_select/krum_aggregate`; Krum functions retained as legacy fallback. Added `_write_trust_log()` method — appends per-round trust records to `logs/fltrust_trust_log.jsonl` (pre-wired for Upgrade 3 Byzantine detection). Return dict exposes `trust_scores` and `fltrust_flagged` alongside existing `krum_*` keys for dashboard backward compat. Byzantine client correctly identified with trust=0.0 in simulation tests. |

---

## 17. Session Handoff State (Current)

> **Handoff Document generated at the end of the current session.**

### Files Touched & Created
1. **`response_policy.yaml` (NEW):** Defines the 3-tier response rules based on severity and asset class.
2. **`policy_engine.py` (NEW):** Parses `response_policy.yaml` and handles subprocess scaling with a mandatory Human-In-The-Loop (HITL) prompt for isolation actions.
3. **`scripts/isolate.sh`, `scripts/throttle.sh`, `scripts/log_only.sh` (NEW):** Shell scripts invoked by the policy engine.
4. **`api_server.py` (NEW):** Standalone Flask server (port 5001) for the custom script injection backend. Validates scripts, writes synchronization logs, and computes AE inference for anomalous NetFlow features.
5. **`response_engine.py`:** Replaced hardcoded `tc`/`iptables` execution with `policy_engine.execute_response()`.
6. **`detector.py`:** Upgraded `EMAThresholdTracker` to include trajectory length (`consecutive_above_2sigma`), rearranged `AURAInferenceEngine` to process EMA before L1 trigger, and implemented a sliding-window temporal accumulator for escalating severities (`LOW×3 → MEDIUM`, etc).
7. **`dashboard.py`:** Inserted HTML/JS injection panel. Added real-time polling loops to consume `logs/pending_inject.json` (triggering response engine and topology color updates) and `logs/last_explanation.json` (rendering AE feature reconstruction table).
8. **`config.py`:** Centralized extraction for all new features. Added `TEMPORAL_WINDOW_SECONDS`, `K_CONSECUTIVE_READINGS`, `FEATURE_INDEX_MAP`, `MSE_THRESHOLD_HIGH/MEDIUM`, and `ATTACK_CORRUPTION_PROFILES`.
9. **`requirements.txt`:** Picked up `pyyaml` and `flask>=3.0.0`.

### Key Functions Modified
- **`detector.EMAThresholdTracker.update`:** Calculates consecutive anomaly trajectories alongside EMA.
- **`detector.AURAInferenceEngine._classify_severity`:** Modified to accept trajectory bounds and floor severity levels based on persistence (e.g., `MEDIUM` if > 2.0σ for K readings).
- **`detector.AURAInferenceEngine._apply_temporal_escalation`:** Orchestrates the per-node timeframe window to compound minor alerts into higher severity events.
- **`api_server._run_inject_inference`:** Now accepts an `attack_type`, dynamically pulls the correct profile from `config.py`, resolves feature indices using `FEATURE_INDEX_MAP`, runs the AE forward pass, calls `explain_ae`, and persists outputs.

### Known Issues & Quirks
- **Stale JSON State:** The synchronization bridge between the Streamlit dashboard and the Flask API server relies heavily on JSON files (`pending_inject.json`, `last_explanation.json`) acting as shared memory. File locking limits aren't implemented, which may lead to race conditions under extreme simultaneous load.
- **HITL Prompt Locality:** The HITL `y/n` prompt blocks on Terminal `stdin`. In a distributed or detached nohup environment, the execution will pause indefinitely.

### Next Pending Upgrade (from `AURA_UPGRADES.md`)
**UPGRADE 2 — Client-Side SHA-256 Hashing + Dashboard**
- **Objective:** Establish chain-of-custody by having each Federated Learning client hash its local weight tensor prior to transmission to the aggregation server.
- **Tasks:**
  - Update local FL training loop (`fl_client.py` equivalent) to compute a SHA-256 `local_hash` of the state dict.
  - Modify the Ganache Smart Contract (`ModelRegistry.sol`) to accept and store the `local_hash` along with the global hash.
  - Build `hash_dashboard.html` with vanilla JS polling a new Flask endpoint (`/api/hashes`) to display the live ledger of client commits.
