# AURA — Adaptive Unified Response Architecture

> **Zero-Day Network Threat Detection using Graph Neural Networks, Federated Learning & Blockchain Audit**
> 
> Team Trinetra · SPECTRE Hackathon 2026

---

## What is AURA?

AURA is a production-grade, privacy-preserving network intrusion detection system that combines:

- **GraphSAGE (Inductive GNN)** — models network topology as a graph; detects anomalies in traffic relationships, not just individual packet stats
- **Flow Autoencoder** — reconstruction-error based anomaly scoring; learns what "normal" looks like and flags deviations
- **EMA Dynamic Thresholding** — self-calibrating threshold using Exponential Moving Average (no hardcoded cutoffs)
- **Federated Learning with FLTrust aggregation (Cao et al.)** — multiple network segments train locally; only model weights are shared, never raw traffic (privacy-preserving). Cosine trust scoring replaces distance-based Krum to eliminate false rejection of honest outlier clients
- **Client-Side SHA-256 Hash Verification** — every client independently hashes received global weights and verifies against the blockchain ledger before loading. Simulated Man-in-the-Middle (MITM) attack defense included
- **Blockchain Audit Log** — every global model update is SHA-256 hashed and written immutably to ledger (tamper-evident supply-chain defence)
- **Reconstruction Error Explainability** — per-feature AE attribution + attack signature matching via cosine similarity, producing human-readable SOC operator explanations
- **YAML-Driven Response Engine** — operator-configurable scripts (LOG → THROTTLE → ISOLATE) loaded from `response_policy.yaml` with Human-in-the-Loop (HITL) gates for isolation actions
- **3-Tier Automated Response** — LOG → THROTTLE → ISOLATE based on severity and node criticality
- **React + Vite Dashboards** — live anomaly detection dashboard + dedicated FL Server Console (replaces Streamlit for performance)

---

## Architecture

```
Raw Network Traffic (CICIDS2017)
          │
          ▼
┌─────────────────────────────────────┐
│  Phase 1: Data Ingestion            │  IsolationForest baseline sanitisation
│  TTL Edge Decay                     │  Synthetic node topology mapping
│  MinMax Feature Scaling             │  Streaming graph windows
│  Randomized Watts-Strogatz Topology │  Small-world rewiring per epoch
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Phase 2: Anomaly Detection         │
│  Layer 1: FlowAutoencoder           │  78→64→32→16→32→64→78 (MSE score)
│  Layer 2: AuraSTGNN                 │  GraphSAGE 78→64→32→1 (topology score)
│  EMA Threshold (3σ)                 │  Adaptive, warms up over 50 windows
│  AE Explainability                  │  Per-feature attribution + attack sig matching
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Phase 3: Federated Learning        │  Flower (flwr) framework
│  FLTrust aggregation (Cao et al.)   │  Cosine trust scoring; Byzantine → zero trust
│  Client-Side SHA-256 Verification   │  MITM defense: reject tampered weights
│  Straggler Timeout (30s)            │  Gradient clipping (norm=1.0)
│  Networked + Simulation Modes       │  Real gRPC transport or in-process demo
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Phase 4: Response Engine           │  YAML-driven scripts (response_policy.yaml)
│  LOW    → LOG_ONLY (log_only.sh)    │
│  MEDIUM → THROTTLE (throttle.sh)    │  tc 10kbps + HITL notification
│  HIGH + Critical → THROTTLE         │  Never auto-isolates DC/SCADA/DB
│  HIGH + Standard → ISOLATE + HITL   │  iptables DROP (blast-radius contained)
│  Custom Script Injection API        │  Flask API validates & logs injection events
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Phase 5: Blockchain Audit          │  SHA-256 model hash per FL round
│  ModelRegistry.sol                  │  Solidity 0.8.19 smart contract
│  Local fallback (no Ganache)        │  JSONL ledger if blockchain offline
│  Trusted Hash Registry              │  Separate ground truth for tamper detection
│  verify_chain.py                    │  Cross-checks ledger vs registry
└─────────────────────────────────────┘
```

---

## Project Structure

```
TRINETRA/
├── aura/
│   ├── __init__.py
│   ├── data_loader.py           # CICIDS2017 pipeline, IsolationForest sanitisation
│   ├── models.py                # FlowAutoencoder + AuraSTGNN (manual SAGEConv)
│   ├── detector.py              # EMA dynamic thresholding, cascade L1→L2
│   ├── response_engine.py       # 3-tier policy engine, HITL, iptables simulation
│   ├── fl_client.py             # Flower FL client + SHA-256 hash verification + MITM defense
│   ├── fl_server.py             # FLTrust aggregation, straggler policy, blockchain audit
│   ├── blockchain.py            # Web3 + local fallback audit logger
│   ├── attack_injector.py       # 6 attack profiles for red-team simulation
│   └── ae_explainer.py          # AE Feature Attribution & Attack Classification (Upgrade 7)
├── contracts/
│   └── ModelRegistry.sol        # Solidity smart contract for model hash registry
├── scripts/
│   ├── isolate.sh               # Network isolation script (iptables DROP)
│   ├── throttle.sh              # Bandwidth throttle script (tc qdisc)
│   └── log_only.sh              # Passive logging script
├── frontend/                    # React + Vite UI (Operations + FL Server Console)
├── dashboard.py                 # [Legacy] Streamlit dashboard — use frontend/ instead
├── fl_server_dashboard.py       # [Legacy] Streamlit FL console — use frontend/ instead
├── api_server.py                # Flask REST API backend (port 5001)
├── policy_engine.py             # YAML-driven response script execution engine (Upgrade 1)
├── response_policy.yaml         # Operator-configurable response rules
├── train.py                     # Two-phase training pipeline
├── run.py                       # Quick-start launcher (train/dashboard/demo/test/federation)
├── run_federation_networked.py  # Cross-network FL launcher (real gRPC, separate processes)
├── calibrate_thresholds.py      # AE MSE threshold calibration + feature index audit
├── verify_chain.py              # Blockchain integrity verifier
├── config.py                    # All hyperparameters, paths, attack corruption profiles
├── requirements.txt
└── README.md
```

---

## Dataset

**CICIDS2017** — Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017

- **Download:** https://www.unb.ca/cic/datasets/ids-2017.html
- **Variant used:** `MachineLearningCSV` (78 statistical flow features + Label)
- **Place files in:** `CSV's/MachineLearningCVE/`

The dataset is **not included** in this repository due to size (several GB).

Attack types covered: DDoS, Port Scan, Brute Force, Web Attacks (XSS, SQLi), Infiltration, Botnet, DoS

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/SenseiSuraj24/TRINETRA.git
cd TRINETRA
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Train the models (quick — ~2 min)

```bash
python run.py train --quick
```

### 4. Calibrate thresholds (recommended before first demo)

```bash
python calibrate_thresholds.py               # uses saved AE checkpoint
python calibrate_thresholds.py --train-quick  # trains a fresh AE quickly then calibrates
python calibrate_thresholds.py --audit-only   # only audits FEATURE_INDEX_MAP vs CSV
```

### 5. Launch the dashboard (React + Vite)

**Prerequisites:** Node.js 18+ and npm.

```bash
python run.py dashboard
# API:  http://localhost:5001
# UI:   http://localhost:5173  (opens automatically)
```

**Manual start (two terminals):**

```bash
# Terminal 1 — API backend
python api_server.py

# Terminal 2 — React UI
cd frontend && npm install && npm run dev
# Open http://localhost:5173
```

The UI includes both **Operations Dashboard** (`/`) and **FL Server Console** (`/fl-server`) via sidebar navigation.

### 6. [Legacy] Streamlit dashboards

Streamlit UIs are retained for reference but are no longer recommended (laggy). Use the React frontend instead.

```bash
streamlit run dashboard.py          # port 8501
streamlit run fl_server_dashboard.py  # port 8502
```

### 8. Run sanity tests

```bash
python run.py test
```

### 9. CLI pipeline demo (no browser)

```bash
python run.py demo
```

### 10. Federated Learning — Simulation Mode

```bash
python run.py federation
```

### 11. Federated Learning — Networked Mode (real gRPC)

```bash
# All 3 orgs on one machine (separate processes, gRPC transport):
python run_federation_networked.py

# Server-only (wait for remote clients from other machines):
python run_federation_networked.py --server-only --server-address 0.0.0.0:8080

# Individual client on a remote machine:
python -m aura.fl_client --client-id org_hospital_1 --server <SERVER_IP>:8080
python -m aura.fl_client --client-id org_bank_2    --server <SERVER_IP>:8080 --byzantine
python -m aura.fl_client --client-id org_university_3 --server <SERVER_IP>:8080
```

### 12. Verify blockchain integrity

```bash
python verify_chain.py
```

---

## Implemented Upgrades

### Upgrade 1 — Custom Script Execution Engine
Replaced hardcoded `iptables` / `tc` response actions with an operator-configurable YAML policy file:

| File | Purpose |
|---|---|
| `response_policy.yaml` | Declarative severity → asset_class → script mapping |
| `policy_engine.py` | Loads YAML rules, matches events, executes scripts with HITL gate for isolation |
| `scripts/isolate.sh` | Network isolation (iptables DROP) |
| `scripts/throttle.sh` | Bandwidth throttle (tc qdisc htb rate 10kbps) |
| `scripts/log_only.sh` | Passive logging |

**HITL Gate:** If the matched script contains `isolate`, the operator is prompted for confirmation. Rejection auto-degrades to throttle — a node never exits the response engine in an uncontrolled state.

---

### Upgrade 2 — Client-Side SHA-256 Hashing & MITM Defense
Each FL client independently hashes received global model weights (SHA-256) and verifies against the blockchain ledger **before** loading them into the local model:

- **Hash verification** runs in both `fit()` and `evaluate()` lifecycle methods
- **MITM simulation** injects Gaussian noise into received weights, causing hash mismatch → weights are **rejected** and the client trains on its last known-good local model
- **CLI flags:** `--simulate-mitm` (forced) or `--mitm-probability 0.3` (random 30% trigger)
- **API Server** (`api_server.py`) provides a Flask API for custom script injection with security validation (blocks `os.system`, `subprocess`, `import os`, `import sys`)

---

### Upgrade 3 — Byzantine Detection Logging
FLTrust trust scores are logged per-round to `logs/fltrust_trust_log.jsonl`:
```json
{"round": 1, "trust_scores": [0.89, 0.01, 0.91], "flagged_indices": [1], "timestamp": 1711647200}
```
Zero-trust clients (gradient direction opposes benign improvement) are flagged as Byzantine suspects. This feeds into the FL Server Console dashboard for live visualization.

---

### Upgrade 4 — Randomized Network Topology per Epoch
The network graph fed to GraphSAGE uses Watts-Strogatz small-world rewiring to validate inductive generalization. `generate_topology()` generates a fresh graph each epoch using `networkx.watts_strogatz_graph(n, k=4, p=0.15, seed=None)`.

---

### Upgrade 6 — FLTrust Aggregation (Replaced Krum)
**Core architectural migration from Krum to FLTrust (Cao et al., 2020):**

| Aspect | Krum (Legacy) | FLTrust (Active) |
|---|---|---|
| **Selection** | Distance-based rejection | Cosine trust scoring |
| **False positives** | Drops honest outlier clients (rare data) | Preserves them (gradient direction matches) |
| **Byzantine defense** | Drops geometrically extreme updates | ReLU(cosine) → adversarial reversal = zero trust |
| **Magnitude attack** | Vulnerable to scale amplification | Normalises client update to server magnitude |

The server holds a small root dataset (`FLTRUST_ROOT_SAMPLES = 200` synthetic benign samples) and trains one gradient step per round as a reference direction. Legacy Krum code is retained as a fallback but is **not** used in the active aggregation path.

---

### Upgrade 7 — Reconstruction Error Explainability Layer
`aura/ae_explainer.py` provides interpretable anomaly reports:

- **Per-feature attribution:** Top-K contributing features ranked by reconstruction error
- **Feature group aggregation:** Volume/Bytes, Bandwidth, Timing/IAT, TCP Flags, Idle/Active, Bulk Transfer, Window/Segment
- **Attack signature matching:** Cosine similarity between residual vector and pre-defined attack signature vectors (DDoS, Port Scan, Lateral Movement, Data Exfiltration, Web Attack)
- **Human-readable explanations:** Icon, summary, detailed SOC narrative, and "why this is anomalous" for each attack category

---

## Key Technical Decisions

| Decision | Why |
|---|---|
| **GraphSAGE over GCN** | Inductive — detects threats on new/unseen nodes without retraining |
| **Manual SAGEConv** | No torch_geometric dependency; implemented via `torch.scatter_add_` — more portable |
| **EMA threshold over static** | Network baseline drifts (Monday ≠ Friday traffic) — adaptive 3σ detection |
| **FLTrust over Krum** | Trust-scored gradient aggregation — cosine similarity between client and server gradients; Byzantine clients score zero. Preserves rare-data honest clients that Krum would drop |
| **Client-side hash verification** | Supply-chain defence — SHA-256 hash checked against blockchain before loading weights. Blocks Man-in-the-Middle weight tampering |
| **IsolationForest sanitisation** | Removes 2% extreme outliers from benign baseline before scaler fitting — prevents data poisoning |
| **Never isolate critical infra** | Auto-isolating a Domain Controller is worse than the attack — HITL required |
| **YAML-driven response policy** | Operators can change response behaviour by editing `response_policy.yaml` without touching Python code |
| **Blockchain for model hashes** | Supply-chain attack defence — detects silent model weight tampering |
| **AE Explainability** | SOC operators need to know *which features* drove the alert, not just a scalar MSE — interpretable by design |
| **Dual-mode FL** | Simulation mode for demos + real networked gRPC mode for production/cross-machine deployment |

---

## Attack Simulation (Dashboard)

The dashboard includes 6 red-team attack profiles:

| Attack | Feature Perturbation |
|---|---|
| **DDoS Flood** | Max packet rate, near-zero IAT, high SYN flags |
| **Port Scan** | Near-zero duration/bytes, high RST flags |
| **Lateral Movement** | High IAT std (beaconing), edges rewired to critical nodes |
| **Data Exfiltration** | Very high forward bytes, near-zero backward bytes |
| **Web Attack** | Large payload, high PSH flags, short duration |
| **Custom Injection** | Operator-defined via API (`POST /api/inject_custom`) |

---

## Response Policy Matrix

| Severity | Node Type | Action | Script |
|---|---|---|---|
| LOW | Any | `LOG_ONLY` | `scripts/log_only.sh` |
| MEDIUM | Any | `THROTTLE` + HITL notification | `scripts/throttle.sh` |
| HIGH | Critical (DC / SCADA / DB / Payment GW) | `THROTTLE` + HITL | `scripts/throttle.sh` |
| HIGH | Standard workstation | `ISOLATE` (HITL gate) | `scripts/isolate.sh` |

---

## Model Parameters

| Component | Architecture | Parameters |
|---|---|---|
| FlowAutoencoder | 78→64→32→16→32→64→78 | 15,390 |
| AuraSTGNN | SAGEConv 78→64→32→1 | 14,913 |
| **Total** | | **30,303** |

---

## FLTrust Configuration

| Parameter | Value | Purpose |
|---|---|---|
| `FLTRUST_ROOT_SAMPLES` | 200 | Server-held benign samples for reference gradient |
| `FLTRUST_SERVER_LR` | 1e-3 | LR for server's one-step root gradient |
| `FLTRUST_MIN_TRUST_SCORE` | 0.0 | Trust ≤ this → flagged Byzantine (ReLU-only default) |
| `FL_NUM_ROUNDS` | 3 | Federation rounds (1 blockchain mint on final) |
| `FL_MIN_CLIENTS` | 3 | Quorum for aggregation |
| `FL_ROUND_TIMEOUT_SEC` | 30 | Straggler hard timeout |

---

## Dashboards

### Anomaly Detection Dashboard (`dashboard.py`)
- **Live threat feed** with real-time anomaly detection events
- **Custom script injection** via embedded API form
- **AE Explainability panels** showing per-feature attribution and attack classification
- **Response action logs** with iptables/tc simulation output
- **Network topology visualization** with critical node highlighting

### FL Server Console (`fl_server_dashboard.py`)
- **Org node readiness** panel with quarantine status
- **Step-by-step pipeline animation**: Collect Weights → FLTrust Filter → Aggregate → Mint Hash → Broadcast + Verify
- **Per-round trust score table** with FLTrust cosine similarity metrics
- **Blockchain ledger feed** showing minted vs. local hashes
- **Client hash verification outcome** (match / mismatch)
- **Auto-quarantine**: FLTrust-flagged Byzantine orgs are automatically blocked from future FL runs

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/nodes` | Returns the current live node registry |
| `POST` | `/api/inject_custom` | Validates and logs a custom script injection event |

**Security:** Scripts are statically analysed. Any script containing `os.system`, `subprocess`, `import os`, or `import sys` is rejected with HTTP 400. Scripts are **not** executed — they are logged and passed through AE inference for anomaly scoring.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | PyTorch 2.x (CPU) |
| GNN | Manual GraphSAGE (`torch.scatter_add_`) |
| Federated Learning | Flower (`flwr`) + FLTrust |
| Anomaly Detection | scikit-learn IsolationForest + custom EMA |
| Explainability | Per-feature AE attribution + cosine signature matching |
| Blockchain | Web3.py + Solidity 0.8.19 / local fallback |
| Dashboards | Streamlit + Plotly |
| API Server | Flask (port 5001) |
| Response Policy | PyYAML + shell scripts |
| Threshold Calibration | numpy percentile analysis + feature index audit |
| Data | pandas, numpy, networkx |
| Graph Topology | networkx Watts-Strogatz small-world |

---

## Future Work — Plan Mycelium (Theory Extension)

> **Serverless Swarm Learning via Gossip Protocols**

Standard Federated Learning has a structural weakness: the central server is a Single Point of Failure. Mycelium eliminates the central aggregator entirely using **Decentralized Stochastic Gradient Descent (D-SGD)** over a peer-to-peer Gossip Protocol.

The weight update rule for node *i* at time *t* relies on a doubly stochastic mixing matrix *W*:

$$x_i^{(t+1)} = \sum_{j \in \mathcal{N}(i)} W_{ij} x_j^{(t)} - \alpha \nabla f_i(x_i^{(t)})$$

Because *W* is doubly stochastic, the network mathematically converges to the same global loss minimum as centralized FL — without any central coordinator.

---

## Team

**Team Trinetra** · SPECTRE Hackathon 2026
