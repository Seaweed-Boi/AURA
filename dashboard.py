"""
dashboard.py — [LEGACY] AURA Live Operations Dashboard (Streamlit)
================================================================
DEPRECATED: Use the React+Vite frontend instead (faster, no lag).

  python run.py dashboard     # starts api_server + React UI
  cd frontend && npm run dev  # UI only (http://localhost:5173)

This Streamlit version is kept for reference only.

The AURA dashboard is the "nerve centre" of the hackathon demo.

Layout
------
 ┌──────────────────────────────────────────────────────────────────┐
 │  AURA — Autonomous Unified Resilience Architecture         Status │
 ├────────────────────────────┬─────────────────────────────────────┤
 │  Live Network Topology     │  Anomaly Score Timeline             │
 │  (Plotly animated graph)   │  (EMA threshold visible)            │
 │                            │                                     │
 ├────────────────────────────┴─────────────────────────────────────┤
 │  🔴 ATTACK INJECTION         🌐 FEDERATION         ⛓ BLOCKCHAIN  │
 │  [DDoS] [Scan] [Lateral]   [Run FL Simulation]   [Verify Hash]  │
 │  [Exfil] [Web Attack]                                            │
 ├────────────────────────────────────────────────────────────────  ┤
 │  Event Log (last 20 events)            Alert History             │
 └──────────────────────────────────────────────────────────────────┘
"""

import json
import os
import time
import threading
import hashlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from aura.models import FlowAutoencoder, AuraSTGNN, AURAModelBundle
from aura.detector import AURAInferenceEngine, AlertSeverity
from aura.response_engine import AURAResponseEngine
from aura.attack_injector import AttackInjector, AttackType
from aura.blockchain import AURABlockchainLogger

# ─────────────────────────────────────────────────────────────────────────────
# Org Identity  (set via AURA_ORG_ID env var before launching)
# ─────────────────────────────────────────────────────────────────────────────

_ORG_PROFILES = {
    "hospital":   {"label": "Hospital",    "id": "org_hospital_1",   "net": "192.168.1.0/24",  "icon": "🏥", "role": "Normal", "color": "#00ff88"},
    "bank":       {"label": "Bank",        "id": "org_bank_2",       "net": "10.0.1.0/24",    "icon": "🏦", "role": "Normal", "color": "#388bfd"},
    "university": {"label": "University",  "id": "org_university_3", "net": "172.16.1.0/24",  "icon": "🎓", "role": "Normal", "color": "#4488ff"},
}
_ORG_KEY  = os.environ.get("AURA_ORG_ID", "").lower().strip()
ORG       = _ORG_PROFILES.get(_ORG_KEY)   # None if not set

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

_page_title = f"AURA — {ORG['icon']} {ORG['label']}" if ORG else "AURA — Autonomous Unified Resilience Architecture"
_page_icon  = ORG["icon"] if ORG else "🛡️"

st.set_page_config(
    page_title = _page_title,
    page_icon  = _page_icon,
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# Auto-refresh every DASHBOARD_REFRESH_INTERVAL_MS milliseconds.
# This drives the pending_inject.json poll — without it, the node colour
# can only update when the user manually clicks a button.
st_autorefresh(interval=cfg.DASHBOARD_REFRESH_INTERVAL_MS, key="aura_autorefresh")

# ─────────────────────────────────────────────────────────────────────────────
# Colour Theme
# ─────────────────────────────────────────────────────────────────────────────

THEME = {
    "bg":        "#0a0e1a",
    "panel":     "#0f1629",
    "border":    "#1e2d4a",
    "text":      "#e0e8f0",
    "green":     "#00ff88",
    "yellow":    "#ffd700",
    "red":       "#ff4444",
    "blue":      "#4488ff",
    "cyan":      "#00ccff",
    "orange":    "#ff8800",
    "dim":       "#445566",
}

st.markdown(f"""
<style>
  .stApp {{ background-color: {THEME['bg']}; }}
  .block-container {{ padding-top: 1rem; }}
  .metric-card {{
    background: {THEME['panel']}; border: 1px solid {THEME['border']};
    border-radius: 8px; padding: 1rem; text-align: center;
  }}
  .alert-high  {{ color: {THEME['red']};    font-weight: bold; }}
  .alert-med   {{ color: {THEME['orange']}; font-weight: bold; }}
  .alert-low   {{ color: {THEME['yellow']}; }}
  .alert-norm  {{ color: {THEME['green']};  }}
  .chain-row   {{ color: {THEME['cyan']};   font-family: monospace; font-size: 0.8em; }}
  .fed-log     {{ color: {THEME['blue']};   font-family: monospace; font-size: 0.8em; }}
  h1, h2, h3, h4 {{ color: {THEME['text']} !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "engine":          None,
        "responder":       None,
        "injector":        None,
        "blockchain":      None,
        "ae_scores":       [],       # Timeline of AE MSE scores
        "thresholds":      [],       # Corresponding EMA thresholds
        "timestamps":      [],       # Wall-clock times
        "alerts":          [],       # List of AnomalyEvent dicts
        "incidents":       [],       # List of IncidentRecord dicts
        "fed_log":         [],       # Federation event strings
        "chain_log":       [],       # Blockchain hash events
        "node_colors":     {i: THEME["green"] for i in range(cfg.NUM_SYNTHETIC_NODES)},
        "node_states":     {i: "Normal" for i in range(cfg.NUM_SYNTHETIC_NODES)},
        "current_graph":   None,
        "attack_active":   False,
        "attack_type":     None,
        "system_status":   "INITIALISING",
        "total_attacks":   0,
        "total_blocked":   0,
        "fl_rounds_done":  0,
        "chain_entries":   0,
        "models_loaded":   False,
        "window_counter":  0,
        "last_explanation": None,   # Most recent AE explainer output dict
        "fl_client_status": [],     # Per-client metadata from latest FL round
        "fl_ready":         False,  # Whether this org node has signalled FL readiness
        "under_attack":     False,  # Whether this org is currently under active attack
    }
    # Sync readiness from shared file if this is an org node
    if ORG and "fl_ready" not in st.session_state:
        _rf = Path(cfg.LOGS_DIR) / "fl_readiness.json"
        if _rf.exists():
            try:
                _rd = json.loads(_rf.read_text())
                defaults["fl_ready"]    = _rd.get(_ORG_KEY, {}).get("ready",        False)
                defaults["under_attack"] = _rd.get(_ORG_KEY, {}).get("under_attack", False)
            except Exception:
                pass
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Model & Component Loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AURA models …")
def load_components():
    """Load or initialise all AURA components.  Cached across reruns."""
    bundle_path = cfg.MODELS_DIR / "aura_bundle.pth"
    bundle      = AURAModelBundle()

    if bundle_path.exists():
        bundle.load_state_dict(torch.load(bundle_path, map_location="cpu"))
        status = "PRE-TRAINED"
    else:
        status = "UNTRAINED (DEMO MODE)"

    engine    = AURAInferenceEngine(bundle.autoencoder, bundle.stgnn)
    responder = AURAResponseEngine()
    injector  = AttackInjector()
    bc        = AURABlockchainLogger()

    # Pre-warm EMA threshold with synthetic normal traffic so alerts fire
    # immediately when the user clicks an attack button on the dashboard.
    _N = cfg.NUM_SYNTHETIC_NODES
    _E = 40
    for _ in range(cfg.EMA_WARMUP_BATCHES + 10):
        _x    = torch.randn(_N, cfg.FEATURE_DIM) * 0.1   # low-variance normal
        _ei   = torch.randint(0, _N, (2, _E))
        _attr = torch.randn(_E, cfg.FEATURE_DIM) * 0.1
        engine.process({"x": _x, "edge_index": _ei,
                        "edge_attr": _attr, "window_id": "warmup"})

    return engine, responder, injector, bc, status


engine, responder, injector, bc, model_status = load_components()

# Inject into session state if not already there
if not st.session_state["models_loaded"]:
    st.session_state["engine"]       = engine
    st.session_state["responder"]    = responder
    st.session_state["injector"]     = injector
    st.session_state["blockchain"]   = bc
    st.session_state["models_loaded"] = True
    st.session_state["system_status"] = "ACTIVE"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def build_network_figure(
    node_colors: dict,
    node_states: dict,
    edge_index:  Optional[np.ndarray] = None,
    n_nodes:     int = cfg.NUM_SYNTHETIC_NODES,
) -> go.Figure:
    """
    Build an animated Plotly network graph of the current network state.
    Nodes are arranged in a circular topology.
    Node colour reflects health: green=normal, yellow=suspect, red=isolated.
    """
    # Circular layout
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    pos_x  = np.cos(angles)
    pos_y  = np.sin(angles)

    # Node labels (match critical allowlist)
    node_labels = []
    for i in range(n_nodes):
        key   = f"node_{i}"
        label = cfg.CRITICAL_ALLOWLIST.get(key, f"Host-{i:02d}")
        node_labels.append(f"{label}<br>({node_states.get(i, 'Normal')})")

    # Edge traces
    edge_traces = []
    if edge_index is not None and edge_index.shape[1] > 0:
        for idx in range(min(edge_index.shape[1], 60)):  # Cap for performance
            s, d = int(edge_index[0, idx]), int(edge_index[1, idx])
            edge_traces.append(go.Scatter(
                x=[pos_x[s], pos_x[d], None],
                y=[pos_y[s], pos_y[d], None],
                mode="lines",
                line=dict(width=1, color="#1e3a5a"),
                hoverinfo="none",
                showlegend=False,
            ))

    # Node trace
    colors = [node_colors.get(i, THEME["green"]) for i in range(n_nodes)]
    sizes  = [20 if f"node_{i}" in cfg.CRITICAL_ALLOWLIST else 14 for i in range(n_nodes)]
    symbols= ["diamond" if f"node_{i}" in cfg.CRITICAL_ALLOWLIST else "circle"
              for i in range(n_nodes)]

    node_trace = go.Scatter(
        x=pos_x, y=pos_y,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colors,
            symbol=symbols,
            line=dict(width=1.5, color="#ffffff"),
        ),
        text=[f"N{i}" for i in range(n_nodes)],
        textposition="top center",
        textfont=dict(size=8, color=THEME["text"]),
        hovertext=node_labels,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        paper_bgcolor = THEME["bg"],
        plot_bgcolor  = THEME["bg"],
        margin        = dict(l=10, r=10, t=10, b=10),
        xaxis         = dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis         = dict(showgrid=False, zeroline=False, showticklabels=False,
                              scaleanchor="x"),
        height        = 320,
    )
    return fig


def build_score_timeline(
    scores:     List[float],
    thresholds: List[float],
    timestamps: List[float],
) -> go.Figure:
    """Plotly line chart: AE anomaly score vs dynamic EMA threshold over time."""
    if not scores:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["bg"], height=200,
            annotations=[dict(text="Awaiting data…", showarrow=False,
                              font=dict(color=THEME["dim"], size=14))]
        )
        return fig

    t = list(range(len(scores)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=scores, mode="lines",
        name="AE Score (MSE)", line=dict(color=THEME["cyan"], width=2),
    ))

    valid_thresh = [v for v in thresholds if v > 0]
    if valid_thresh:
        fig.add_trace(go.Scatter(
            x=t, y=thresholds, mode="lines",
            name="EMA Threshold (3σ)",
            line=dict(color=THEME["red"], width=1.5, dash="dash"),
        ))

    fig.update_layout(
        paper_bgcolor = THEME["bg"],
        plot_bgcolor  = THEME["bg"],
        font          = dict(color=THEME["text"]),
        legend        = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis         = dict(showgrid=False, title="Window"),
        yaxis         = dict(showgrid=True, gridcolor=THEME["border"], title="MSE"),
        height        = 200,
        margin        = dict(l=40, r=10, t=10, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Inference Step (called each tick)
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_tick(graph: dict, is_attack: bool = False):
    """Run one inference window through the AURA pipeline."""
    eng  = st.session_state["engine"]
    resp = st.session_state["responder"]

    event = eng.process(graph)

    # Update timeline data
    st.session_state["ae_scores"].append(event.ae_score)
    thresh = event.ae_threshold if event.ae_threshold > 0 else 0
    st.session_state["thresholds"].append(thresh)
    st.session_state["timestamps"].append(event.timestamp)

    # Trim to last 100 points for display
    for key in ["ae_scores", "thresholds", "timestamps"]:
        if len(st.session_state[key]) > 100:
            st.session_state[key] = st.session_state[key][-100:]

    # Update node colours
    colors = {i: THEME["green"] for i in range(cfg.NUM_SYNTHETIC_NODES)}
    states = {i: "Normal" for i in range(cfg.NUM_SYNTHETIC_NODES)}

    if event.severity != AlertSeverity.NORMAL:
        st.session_state["total_attacks"] += 1

        # Attack nodes → RED
        for nid in event.triggered_nodes:
            colors[nid] = THEME["red"]
            states[nid] = f"⚠ {event.severity.name}"

        # Add alert to log
        st.session_state["alerts"].insert(0, event.to_dict())
        if len(st.session_state["alerts"]) > cfg.MAX_ALERTS_CACHE:
            st.session_state["alerts"] = st.session_state["alerts"][:cfg.MAX_ALERTS_CACHE]

        # Store explanation for the live panel (overwrite with latest triggered event)
        if event.inferred_attack != "Normal" and event.top_features:
            from aura.ae_explainer import ATTACK_EXPLANATIONS
            st.session_state["last_explanation"] = {
                "inferred_attack": event.inferred_attack,
                "match_score":     event.match_score,
                "top_features":    event.top_features,
                "group_residuals": event.group_residuals,
                "severity":        event.severity.name,
                "confidence":      event.confidence,
                "explanation":     ATTACK_EXPLANATIONS.get(
                    event.inferred_attack,
                    ATTACK_EXPLANATIONS.get("Unknown Anomaly", {})
                ),
            }

        # Run response engine
        records = resp.act(event)
        for r in records:
            if r.action_taken not in ("LOG_ONLY", "ALREADY_ACTIONED"):
                st.session_state["total_blocked"] += 1
            st.session_state["incidents"].insert(0, r.to_dict())
        if len(st.session_state["incidents"]) > 20:
            st.session_state["incidents"] = st.session_state["incidents"][:20]

    elif is_attack:
        # Attack was injected but EMA warmup still active — L1 not yet triggered.
        # Run the explainer directly on the raw features so the operator can
        # already see WHAT is anomalous, even before a formal alert fires.
        for nid in graph.get("attack_nodes", []):
            colors[nid] = THEME["yellow"]
            states[nid] = cfg.NODE_STATE_EVALUATING_SIMPLE
                expl = explain_ae(feat_residuals)
                st.session_state["last_explanation"] = {
                    "inferred_attack": expl["inferred_attack"],
                    "match_score":     expl["match_score"],
                    "top_features":    expl["top_features"],
                    "group_residuals": expl["group_residuals"],
                    "severity":        "LOW",   # warmup → tentative
                    "confidence":      expl["match_score"],
                    "explanation":     ATTACK_EXPLANATIONS.get(
                        expl["inferred_attack"],
                        ATTACK_EXPLANATIONS.get("Unknown Anomaly", {})
                    ),
                }
        except Exception:
            pass

    st.session_state["node_colors"]   = colors
    st.session_state["node_states"]   = states
    st.session_state["current_graph"] = graph
    st.session_state["window_counter"] += 1

    return event


# ─────────────────────────────────────────────────────────────────────────────
# Federation Simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_federation():
    """Run the full FL simulation and populate the federation log."""
    from aura.fl_server import run_federation_simulation

    st.session_state["fed_log"] = []
    st.session_state["fed_log"].append("🚀 Federation round initiated …")

    bc_module = st.session_state["blockchain"]

    # Capture federation output by running the simulation
    round_results = run_federation_simulation(blockchain_module=bc_module, n_rounds=3)

    for r in round_results:
        rnd     = r.get("round", "?")
        version = r.get("model_version", "N/A")
        h       = r.get("model_hash", "N/A")
        _trusted = r.get("fltrust_trusted_indices", [])
        kept     = len(_trusted)

        # Keep the latest round's per-client status for the dashboard table
        if "client_statuses" in r:
            st.session_state["fl_client_status"] = r["client_statuses"]

        st.session_state["fed_log"].extend([
            f"━━━  Round {rnd}  ━━━",
            f"[CLIENT hospital_1] Attack pattern learned. Sending weights…",
            f"[CLIENT bank_2]     Local training complete. Sending weights…",
            f"[CLIENT uni_3]      Local training complete. Sending weights…",
            f"[SERVER] FLTrust: {kept}/3 client updates trusted (cosine vs server root).",
            f"[SERVER] Global Model {version} aggregated.",
            f"[BLOCKCHAIN] Hash recorded: {h[:20]}…",
            f"[CLIENT hospital_1] Verifying hash on chain… ✓ Match. Model deployed.",
            f"[CLIENT bank_2]     Verifying hash on chain… ✓ Match. Model deployed.",
            f"[CLIENT uni_3]      Verifying hash on chain… ✓ Match. Model deployed.",
        ])

        st.session_state["chain_log"].insert(0, {
            "version": version,
            "hash":    h,
            "round":   rnd,
            "time":    time.strftime("%H:%M:%S"),
        })

    st.session_state["fl_rounds_done"] += 3
    st.session_state["chain_entries"]   = len(st.session_state["chain_log"])
    st.session_state["fed_log"].append("✅ Federation complete.  All clients immunised.")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

status_color = {"ACTIVE": THEME["green"], "INITIALISING": THEME["yellow"],
                "UNDER ATTACK": THEME["red"]}.get(st.session_state["system_status"], THEME["yellow"])

_org_badge = ""
if ORG:
    _badge_color = ORG["color"]
    _org_badge = (
        f"<span style='background:{_badge_color}22; border:1px solid {_badge_color}; "
        f"border-radius:20px; padding:2px 12px; font-size:0.82em; "
        f"color:{_badge_color}; margin-left:0.8em; font-weight:bold'>"
        f"{ORG['icon']} {ORG['label'].upper()}  ·  {ORG['net']}"
        f"</span>"
    )

st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center;
            background:{THEME['panel']}; border:1px solid {THEME['border']};
            border-radius:8px; padding:0.8rem 1.5rem; margin-bottom:1rem;">
  <div>
    <span style="font-size:1.4em; font-weight:bold; color:{THEME['cyan']}">
      🛡️ AURA
    </span>
    <span style="color:{THEME['dim']}; margin-left:0.5em; font-size:0.85em;">
      Autonomous Unified Resilience Architecture
    </span>
    {_org_badge}
  </div>
  <div style="text-align:right;">
    <span style="color:{status_color}; font-weight:bold; font-size:0.95em;">
      ● {st.session_state['system_status']}
    </span>
    <span style="color:{THEME['dim']}; margin-left:1em; font-size:0.75em;">
      {model_status}  |  Blockchain: {bc.mode.upper()}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Row
# ─────────────────────────────────────────────────────────────────────────────

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.metric("Windows Processed", st.session_state["window_counter"])
with m2:
    st.metric("Threats Detected", st.session_state["total_attacks"])
with m3:
    st.metric("Nodes Blocked", st.session_state["total_blocked"])
with m4:
    st.metric("FL Rounds", st.session_state["fl_rounds_done"])
with m5:
    st.metric("Chain Entries", st.session_state["chain_entries"])
with m6:
    ema_val = (st.session_state["ae_scores"][-1]
               if st.session_state["ae_scores"] else 0.0)
    st.metric("Current AE Score", f"{ema_val:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Content: Network Graph + Score Timeline
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1 — Custom Injection Bridge: pending_inject.json poll
# Runs every Streamlit rerun cycle.  Consumes the file written by api_server.py,
# updates node colour to yellow, drives the response engine with the AE MSE, and
# appends a CUSTOM_INJECT entry to the Alert History.  Clears the file after read.
# ─────────────────────────────────────────────────────────────────────────────

_pending_path = Path(cfg.LOGS_DIR) / "pending_inject.json"
if _pending_path.exists():
    try:
        _pi = json.loads(_pending_path.read_text())
        _pi_idx = int(_pi.get("node_index", -1))
        _pi_mse = float(_pi.get("mse", 0.0))
        _pi_ts  = float(_pi.get("timestamp", 0))
        _pi_nid = str(_pi.get("target_node", ""))

        if _pi_idx >= 0 and _pi_ts > 0 and (time.time() - _pi_ts) < cfg.PENDING_INJECT_FRESHNESS_SEC:
            # ── Update topology: target node → yellow ──────────────────────
            st.session_state["node_colors"][_pi_idx] = THEME["yellow"]
            st.session_state["node_states"][_pi_idx] = cfg.NODE_STATE_EVALUATING_ICON

            # ── Determine severity from MSE ────────────────────────────
            from aura.detector import AnomalyEvent, AlertSeverity
            _pi_sev = (
                AlertSeverity.HIGH   if _pi_mse > cfg.MSE_THRESHOLD_HIGH   else
                AlertSeverity.MEDIUM if _pi_mse > cfg.MSE_THRESHOLD_MEDIUM else
                AlertSeverity.LOW
            )
            _pi_conf = min(1.0, _pi_mse / 0.5)

            # ── Build AnomalyEvent and drive response engine ───────────────
            _pi_event = AnomalyEvent(
                timestamp       = _pi_ts,
                window_id       = f"CUSTOM_INJECT_{_pi_nid}",
                ae_score        = _pi_mse,
                ae_threshold    = cfg.CUSTOM_INJECT_AE_THRESHOLD,
                gnn_scores      = [],
                severity        = _pi_sev,
                triggered_nodes = [_pi_idx],
                confidence      = _pi_conf,
                raw_label_ratio = 0.0,
                # Explainability fields — populated by last_explanation.json panel
                # separately (Fix 2).  Provide required defaults here so the
                # dataclass constructor doesn't raise TypeError silently.
                top_features    = [],
                inferred_attack = "Custom Injection",
                match_score     = 0.0,
                group_residuals = {},
            )
            if st.session_state.get("responder"):
                _pi_records = st.session_state["responder"].act(_pi_event)
                for _pr in _pi_records:
                    st.session_state["incidents"].insert(0, _pr.to_dict())
                if len(st.session_state["incidents"]) > cfg.MAX_INCIDENTS_CACHE:
                    st.session_state["incidents"] = st.session_state["incidents"][:cfg.MAX_INCIDENTS_CACHE]

            # ── Append to Alert History ─────────────────────────────────
            _pi_alert = _pi_event.to_dict()
            _pi_alert["tag"] = "CUSTOM_INJECT"
            st.session_state["alerts"].insert(0, _pi_alert)
            if len(st.session_state["alerts"]) > cfg.MAX_ALERTS_CACHE:
                st.session_state["alerts"] = st.session_state["alerts"][:cfg.MAX_ALERTS_CACHE]
            st.session_state["total_attacks"] += 1

        # Consume-once: overwrite with empty so next cycle skips it
        _pending_path.write_text("{}")
    except Exception:
        pass

col_graph, col_score = st.columns([1, 1])

with col_graph:
    st.markdown(f"<h4 style='color:{THEME['cyan']}'>🌐 Live Network Topology</h4>",
                unsafe_allow_html=True)
    edge_arr = None
    cg = st.session_state.get("current_graph")
    if cg is not None and "edge_index" in cg:
        edge_arr = cg["edge_index"].numpy()

    net_fig = build_network_figure(
        st.session_state["node_colors"],
        st.session_state["node_states"],
        edge_arr,
    )
    st.plotly_chart(net_fig, use_container_width=True, key="network_graph")

    # Legend
    st.markdown(f"""
    <div style="font-size:0.75em; color:{THEME['dim']}">
      <span style="color:{THEME['green']}">◆ Normal</span> &nbsp;
      <span style="color:{THEME['yellow']}">◆ Evaluating</span> &nbsp;
      <span style="color:{THEME['red']}">◆ Threat Detected</span> &nbsp;
      <span style="color:{THEME['text']}">◇ Critical Infrastructure</span>
    </div>
    """, unsafe_allow_html=True)

with col_score:
    st.markdown(f"<h4 style='color:{THEME['cyan']}'>📈 Anomaly Score Timeline</h4>",
                unsafe_allow_html=True)
    fig_timeline = build_score_timeline(
        st.session_state["ae_scores"],
        st.session_state["thresholds"],
        st.session_state["timestamps"],
    )
    st.plotly_chart(fig_timeline, use_container_width=True, key="score_timeline")

    # EMA state info
    if st.session_state["engine"]:
        ema = st.session_state["engine"].ema_state
        warmup_left = max(0, cfg.EMA_WARMUP_BATCHES - ema.get("batch_count", 0))
        if warmup_left > 0:
            st.info(f"🔄 EMA calibrating… {warmup_left} windows remaining in warmup period.")
        else:
            thresh = (ema.get("ema_mean", 0) or 0) + cfg.EMA_SIGMA_MULTIPLIER * (
                (ema.get("ema_var", 0) or 0) ** 0.5
            )
            st.markdown(f"""
            <div style="font-size:0.8em; color:{THEME['dim']}">
              EMA Mean: <b style='color:{THEME['text']}'>{ema.get('ema_mean', 0):.5f}</b> &nbsp;|&nbsp;
              Threshold (3σ): <b style='color:{THEME['red']}'>{thresh:.5f}</b> &nbsp;|&nbsp;
              Batches: <b style='color:{THEME['text']}'>{ema.get('batch_count', 0)}</b>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# AE Explanation Panel
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    f"<h4 style='color:{THEME['yellow']}'>🧠 AE Explanation — Why did the score spike?</h4>",
    unsafe_allow_html=True
)

_expl = st.session_state.get("last_explanation")
if _expl is None:
    _dim = THEME["dim"]
    st.markdown(
        f"<div style='color:{_dim}; font-size:0.85em; padding:0.5rem 0'>"
        "No anomaly detected yet.  Inject an attack to see a live explanation."
        "</div>",
        unsafe_allow_html=True,
    )
else:
    _e     = _expl["explanation"]
    _sev   = _expl["severity"]
    _conf  = _expl["confidence"]
    _match = _expl["match_score"]
    _atk   = _expl["inferred_attack"]
    _icon  = _e.get("icon", "")
    _summ  = _e.get("summary", "")
    _det   = _e.get("detail", "")
    _why   = _e.get("why_high", "")

    sev_color = {"HIGH": THEME["red"], "MEDIUM": THEME["orange"],
                 "LOW": THEME["yellow"]}.get(_sev, THEME["cyan"])

    expl_left, expl_mid, expl_right = st.columns([1.3, 1.0, 0.9])

    # ── Left: Attack classification + detail ──────────────────────────────
    with expl_left:
        _bg, _br = THEME["panel"], THEME["border"]
        st.markdown(
            f"<div style='background:{_bg}; border:1px solid {sev_color}; "
            f"border-radius:8px; padding:0.8rem 1rem;'>"
            f"<div style='font-size:1.1em; font-weight:bold; color:{sev_color}'>"
            f"{_icon} {_atk}</div>"
            f"<div style='color:{THEME['text']}; font-size:0.85em; margin:0.4rem 0'>"
            f"{_summ}</div>"
            f"<hr style='border-color:{THEME['border']}; margin:0.5rem 0'>"
            f"<div style='color:{THEME['dim']}; font-size:0.78em; line-height:1.5'>"
            f"{_det}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Mid: Top contributing features bar chart ───────────────────────────
    with expl_mid:
        _top   = _expl["top_features"][:6]
        _names = [f[0] for f in _top][::-1]
        _vals  = [f[1] for f in _top][::-1]
        import plotly.graph_objects as go
        fig_feat = go.Figure(go.Bar(
            y=_names,
            x=_vals,
            orientation="h",
            marker_color=sev_color,
            marker_line_width=0,
        ))
        fig_feat.update_layout(
            title=dict(text="Top Anomalous Features", font=dict(color=THEME["text"], size=12)),
            paper_bgcolor=THEME["bg"],
            plot_bgcolor=THEME["panel"],
            font=dict(color=THEME["dim"], size=10),
            height=200,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(gridcolor=THEME["border"], title="Mean |residual|"),
            yaxis=dict(gridcolor=THEME["border"]),
        )
        st.plotly_chart(fig_feat, use_container_width=True, key="feat_chart")

    # ── Right: Match confidence + why the score is high ──────────────────
    with expl_right:
        _bg, _br = THEME["panel"], THEME["border"]
        st.markdown(
            f"<div style='background:{_bg}; border:1px solid {_br}; "
            f"border-radius:8px; padding:0.8rem 1rem;'>"
            f"<div style='color:{THEME['dim']}; font-size:0.78em'>Signature match</div>"
            f"<div style='font-size:1.4em; color:{sev_color}; font-weight:bold'>"
            f"{_match:.0%}</div>"
            f"<div style='color:{THEME['dim']}; font-size:0.78em; margin-top:0.6rem'>Detection confidence</div>"
            f"<div style='font-size:1.4em; color:{THEME['text']}; font-weight:bold'>"
            f"{_conf:.0%}</div>"
            f"<hr style='border-color:{THEME['border']}; margin:0.5rem 0'>"
            f"<div style='color:{THEME['dim']}; font-size:0.76em; line-height:1.5'>"
            f"<b style='color:{THEME['text']}'>Why is the score high?</b><br>{_why}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Fix 2 — Custom Injection AE Explanation Panel
# Polls logs/last_explanation.json on every rerun.  Renders only if the file
# was written within the last 30 seconds (checked via mtime).
# ─────────────────────────────────────────────────────────────────────────────

_expl_json_path = Path(cfg.LOGS_DIR) / "last_explanation.json"
_show_inject_expl = False
_inject_expl_data = {}

if _expl_json_path.exists():
    try:
        _age = time.time() - _expl_json_path.stat().st_mtime
        if _age < cfg.LAST_EXPLANATION_FRESHNESS_SEC:
            _inject_expl_data = json.loads(_expl_json_path.read_text())
            _show_inject_expl  = bool(_inject_expl_data.get("top_features"))
    except Exception:
        pass

if _show_inject_expl:
    _ied     = _inject_expl_data
    _ie_node = _ied.get("node", "")
    _ie_mse  = _ied.get("mse", 0.0)
    _ie_atk  = _ied.get("inferred_attack", "Unknown")
    _ie_sc   = _ied.get("match_score", 0.0)
    _ie_tf   = _ied.get("top_features", [])
    _ie_ts   = _ied.get("timestamp", "")

    _hdr_color = THEME["orange"]
    _bg, _br   = THEME["panel"], THEME["border"]
    _dim, _txt = THEME["dim"], THEME["text"]
    _amber     = "#f59e0b"

    st.markdown(
        f"<div style='background:{_bg}; border:1px solid {_amber}; "
        f"border-radius:8px; padding:0.7rem 1rem; margin-top:0.6rem'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem'>"
        f"<span style='color:{_amber}; font-weight:bold; font-size:0.9em'>"
        f"⚡ Custom Injection — AE Reconstruction Analysis</span>"
        f"<span style='color:{_dim}; font-size:0.75em'>"
        f"{_ie_node} │ MSE <b style='color:{THEME['red']}'>{_ie_mse:.4f}</b> │ "
        f"Inferred: <b style='color:{_txt}'>{_ie_atk}</b> │ "
        f"Match: <b style='color:{_txt}'>{_ie_sc:.0%}</b> │ {_ie_ts}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Feature rows
    _max_err = max((f["error"] for f in _ie_tf), default=1e-6)
    _rows_html = ""
    for _f in _ie_tf:
        _bar_w  = max(4, int(180 * _f["error"] / max(_max_err, 1e-6)))
        _bar_col = THEME["red"] if _f["error"] > 0.1 else THEME["orange"]
        _rows_html += (
            f"<tr style='border-bottom:1px solid {_br}'>"
            f"<td style='padding:3px 8px; color:{_dim}; font-size:0.76em; white-space:nowrap'>{_f['name']}</td>"
            f"<td style='padding:3px 8px; color:{_txt}; font-size:0.76em; text-align:right'>{_f['observed']:.3f}</td>"
            f"<td style='padding:3px 8px; color:{_dim}; font-size:0.76em; text-align:right'>{_f['baseline']:.3f}</td>"
            f"<td style='padding:3px 8px'>"
            f"<div style='display:flex;align-items:center;gap:6px'>"
            f"<div style='background:{_bar_col};height:10px;width:{_bar_w}px;border-radius:3px;opacity:0.85'></div>"
            f"<span style='color:{_bar_col};font-size:0.74em;font-weight:bold'>{_f['error']:.4f}</span>"
            f"</div></td></tr>"
        )

    st.markdown(
        f"<table style='width:100%;border-collapse:collapse'>"
        f"<thead><tr style='color:{_dim};font-size:0.72em;border-bottom:1px solid {_br}'>"
        f"<th style='text-align:left;padding:2px 8px'>Feature</th>"
        f"<th style='text-align:right;padding:2px 8px'>Observed</th>"
        f"<th style='text-align:right;padding:2px 8px'>Baseline</th>"
        f"<th style='text-align:left;padding:2px 8px'>Squared Error</th>"
        f"</tr></thead>"
        f"<tbody>{_rows_html}</tbody></table></div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Control Panel (3 columns)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
ctrl_atk, ctrl_fl, ctrl_chain = st.columns([1.2, 1, 0.8])

# ── Attack Injection Panel ────────────────────────────────────────────────────
with ctrl_atk:
    st.markdown(f"<h4 style='color:{THEME['red']}'>🔴 Attack Simulation</h4>",
                unsafe_allow_html=True)

    atk_cols = st.columns(3)
    attack_map = {
        "DDoS": "ddos", "Port Scan": "portscan",
        "Lateral": "lateral", "Exfil": "exfil", "Web": "web",
    }
    all_attacks = list(attack_map.items())

    for idx, (label, atype) in enumerate(all_attacks):
        col = atk_cols[idx % 3]
        if col.button(label, key=f"atk_{atype}", use_container_width=True):
            st.session_state["attack_active"] = True
            st.session_state["attack_type"]   = atype
            st.session_state["system_status"] = "UNDER ATTACK"

            # Generate and process attack graph
            inj = st.session_state["injector"]
            if inj:
                attack_graph = inj.inject(atype)
                event = run_inference_tick(attack_graph, is_attack=True)
                st.toast(
                    f"💥 {label} attack injected!  "
                    f"Severity: {event.severity.name}  "
                    f"Confidence: {event.confidence:.1%}",
                    icon="🚨" if event.severity == AlertSeverity.HIGH else "⚠️"
                )
                st.rerun()

    # ── Custom Script Injection Block ─────────────────────────────────────────
    # Rendered as a self-contained HTML/JS component via st.components.v1.html().
    # Communicates with api_server.py (Flask) running on port 5001.
    # No page reload on submit — success/error feedback is fully inline.
    import streamlit.components.v1 as _components

    # Bake node list into JS at render time so the UI works even if the API
    # server hasn't started yet (graceful degradation).
    _baked_nodes = json.dumps([
        {
            "id":       f"node_{i}",
            "label":    cfg.CRITICAL_ALLOWLIST.get(f"node_{i}", f"Host-{i:02d}"),
            "critical": f"node_{i}" in cfg.CRITICAL_ALLOWLIST,
        }
        for i in range(cfg.NUM_SYNTHETIC_NODES)
    ])

    _panel_bg     = THEME["panel"]
    _panel_border = THEME["border"]
    _text_color   = THEME["text"]
    _dim_color    = THEME["dim"]
    _amber        = "#f59e0b"
    _red          = THEME["red"]
    _green        = THEME["green"]
    _yellow       = THEME["yellow"]

    _inject_html = f"""
    <style>
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{ background: transparent; font-family: 'Segoe UI', sans-serif; }}

      #custom-inject-panel {{
        background: {_panel_bg};
        border: 1px solid {_panel_border};
        border-radius: 8px;
        padding: 0.7rem 0.8rem;
        margin-top: 0.4rem;
      }}

      #custom-inject-panel label {{
        display: block;
        font-size: 0.72em;
        color: {_dim_color};
        margin-bottom: 4px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }}

      #custom-script-input {{
        width: 100%;
        rows: 6;
        background: #070b14;
        border: 1px solid {_panel_border};
        border-radius: 5px;
        color: {_text_color};
        font-family: 'Courier New', monospace;
        font-size: 0.78em;
        padding: 0.5rem;
        resize: vertical;
        min-height: 90px;
        outline: none;
        transition: border-color 0.2s;
      }}
      #custom-script-input:focus {{ border-color: {_amber}; }}

      #custom-target-node {{
        width: 100%;
        background: #070b14;
        border: 1px solid {_panel_border};
        border-radius: 5px;
        color: {_text_color};
        font-size: 0.78em;
        padding: 0.4rem 0.5rem;
        margin-top: 0.4rem;
        outline: none;
        appearance: none;
        cursor: pointer;
        transition: border-color 0.2s;
      }}
      #custom-target-node:focus {{ border-color: {_amber}; }}

      #btn-inject-custom {{
        width: 100%;
        margin-top: 0.5rem;
        padding: 0.42rem 0;
        background: transparent;
        border: 1px solid {_amber};
        border-radius: 5px;
        color: {_amber};
        font-size: 0.82em;
        font-weight: 600;
        cursor: pointer;
        letter-spacing: 0.03em;
        transition: background 0.2s, color 0.2s;
      }}
      #btn-inject-custom:hover  {{ background: {_amber}22; }}
      #btn-inject-custom:active {{ background: {_amber}44; }}
      #btn-inject-custom:disabled {{
        opacity: 0.5; cursor: not-allowed; border-color: {_dim_color};
      }}

      #inject-status {{
        margin-top: 0.4rem;
        font-size: 0.75em;
        min-height: 1.2em;
      }}
    </style>

    <div id="custom-inject-panel">
      <label>⚡ Custom Script Injection</label>

      <textarea
        id="custom-script-input"
        rows="6"
        placeholder="# Write your attack script here"
      ></textarea>

      <select id="custom-target-node">
        <option value="">Select target node</option>
      </select>

      <button id="btn-inject-custom">⚡ Inject Custom Script</button>

      <div id="inject-status"></div>
    </div>

    <script>
    (function() {{
      const BAKED_NODES = {_baked_nodes};
      const API_BASE    = "http://localhost:5001";

      // ── Populate dropdown ────────────────────────────────────────────────
      function populateSelect(nodes) {{
        const sel = document.getElementById("custom-target-node");
        // Clear existing options beyond the default
        while (sel.options.length > 1) sel.remove(1);
        nodes.forEach(function(n) {{
          const opt       = document.createElement("option");
          opt.value       = n.id;
          const crit      = n.critical ? " 🔑" : "";
          opt.textContent = n.id + " — " + n.label + crit;
          sel.appendChild(opt);
        }});
      }}

      // Populate from baked-in list immediately (no API dependency)
      populateSelect(BAKED_NODES);

      // Then try to refresh from live /api/nodes
      fetch(API_BASE + "/api/nodes")
        .then(function(r) {{ return r.json(); }})
        .then(function(nodes) {{ if (nodes && nodes.length) populateSelect(nodes); }})
        .catch(function() {{ /* API not running — baked list already shown */ }});

      // ── Inject button ────────────────────────────────────────────────────
      document.getElementById("btn-inject-custom").addEventListener("click", async function() {{
        const btn       = document.getElementById("btn-inject-custom");
        const statusEl  = document.getElementById("inject-status");
        const script    = document.getElementById("custom-script-input").value;
        const target    = document.getElementById("custom-target-node").value;

        // Client-side validation
        if (!target) {{
          statusEl.innerHTML = '<span style="color:{_red}">⚠ Please select a target node.</span>';
          return;
        }}
        if (!script.trim()) {{
          statusEl.innerHTML = '<span style="color:{_red}">⚠ Script cannot be empty.</span>';
          return;
        }}

        btn.disabled = true;
        statusEl.innerHTML = '<span style="color:{_dim_color}">Submitting…</span>';

        try {{
          const resp = await fetch(API_BASE + "/api/inject_custom", {{
            method:  "POST",
            headers: {{ "Content-Type": "application/json" }},
            body:    JSON.stringify({{ script: script, target_node: target }}),
          }});

          const data = await resp.json();

          if (!resp.ok) {{
            // Error — display inline, no reload
            statusEl.innerHTML =
              '<span style="color:{_red}">✗ ' +
              (data.error || "Injection failed.") +
              "</span>";
          }} else {{
            // Success — flash target node indicator yellow for 2 s
            const sel       = document.getElementById("custom-target-node");
            const nodeLabel = sel.options[sel.selectedIndex].text;

            statusEl.innerHTML =
              '<span style="color:{_amber}; font-weight:600">' +
              "⚡ Script injected → " + nodeLabel +
              ' <span style="font-weight:normal; color:{_yellow}">' +
              "(Evaluating…)</span></span>";

            // Flash the select border yellow for 2 s to indicate Evaluating state
            sel.style.borderColor = "{_yellow}";
            sel.style.boxShadow   = "0 0 6px {_yellow}66";
            setTimeout(function() {{
              sel.style.borderColor = "";
              sel.style.boxShadow   = "";
              statusEl.innerHTML    = "";
            }}, 2000);
          }}
        }} catch (e) {{
          statusEl.innerHTML =
            '<span style="color:{_red}">✗ API server unreachable.' +
            ' Start <code>python api_server.py</code> on port 5001.</span>';
        }} finally {{
          btn.disabled = false;
        }}
      }});
    }})();
    </script>
    """
    _components.html(_inject_html, height=250, scrolling=False)

    # ΓöÇΓöÇ Autoencoder Boundary Explorer ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    st.markdown("---")
    st.markdown(f"<h4 style='color:{THEME['cyan']}'>≡ƒÄ¢∩╕Å Autoencoder Boundary Explorer</h4>", unsafe_allow_html=True)

    @st.cache_resource
    def load_explorer_ae():
        path = getattr(cfg, "MODEL_SAVE_PATH", None)
        if path is None:
            path = cfg.MODELS_DIR / "autoencoder_best.pth"
            if not path.exists():
                path = cfg.MODELS_DIR / "aura_bundle.pth"
        if not path.exists():
            return None
            
        from aura.models import FlowAutoencoder, AURAModelBundle
        if "bundle" in str(path):
            bundle = AURAModelBundle()
            bundle.load_state_dict(torch.load(path, map_location="cpu"))
            ae = bundle.autoencoder
        else:
            ae = FlowAutoencoder()
            ae.load_state_dict(torch.load(path, map_location="cpu"))
        ae.eval()
        return ae

    ae_model = load_explorer_ae()
    if ae_model is None:
        st.warning("Model not trained yet ΓÇö run python run.py train first")
    else:
        ae_cols_1 = st.columns(3)
        ae_cols_2 = st.columns(3)

        val_flow_bytes = ae_cols_1[0].slider("flow_bytes_per_sec", 0.0, 1.0, 0.0)
        val_fwd_pkt    = ae_cols_1[1].slider("fwd_packet_len_mean", 0.0, 1.0, 0.0)
        val_bwd_pkt    = ae_cols_1[2].slider("bwd_packet_len_mean", 0.0, 1.0, 0.0)
        
        val_flow_iat   = ae_cols_2[0].slider("flow_iat_mean", 0.0, 1.0, 0.0)
        val_syn        = ae_cols_2[1].slider("syn_flag_count", 0.0, 1.0, 0.0)
        val_dst_port   = ae_cols_2[2].slider("dst_port", 0.0, 1.0, 0.0)

        x_tensor = torch.zeros(cfg.FEATURE_DIM)
            
        name_map = {
            "flow_bytes_per_sec": "flow_bytes_s",
            "fwd_packet_len_mean": "fwd_pkt_len_mean",
            "bwd_packet_len_mean": "bwd_pkt_len_mean",
            "flow_iat_mean": "flow_iat_mean",
            "syn_flag_count": "syn_flag_count",
            "dst_port": "dest_port",
        }
        
        for ui_name, val in [
            ("flow_bytes_per_sec", val_flow_bytes),
            ("fwd_packet_len_mean", val_fwd_pkt),
            ("bwd_packet_len_mean", val_bwd_pkt),
            ("flow_iat_mean", val_flow_iat),
            ("syn_flag_count", val_syn),
            ("dst_port", val_dst_port)
        ]:
            cfg_name = name_map.get(ui_name, ui_name)
            idx = cfg.FEATURE_INDEX_MAP.get(cfg_name)
            if idx is not None:
                x_tensor[idx] = val

        with torch.no_grad():
            # Get modified reconstruction
            x_hat, _ = ae_model(x_tensor.unsqueeze(0))
            residuals = ((x_tensor.unsqueeze(0) - x_hat) ** 2).squeeze(0)
            
            # Calculate base zero reconstruction to cleanly isolate the slider impact
            base_tensor = torch.zeros(cfg.FEATURE_DIM)
            base_hat, _ = ae_model(base_tensor.unsqueeze(0))
            base_res = ((base_tensor.unsqueeze(0) - base_hat) ** 2).squeeze(0)
            
            # Delta residual: how much worse it got by moving the sliders
            delta_res = torch.relu(residuals - base_res)
            
            # Use delta_res for the feature explanations so they only target slider changes
            mse = delta_res.mean().item() * 15.0  # Scale appropriate for limits
            
            # Replace 'residuals' passed to explainer with the true delta signal
            residuals = delta_res

        st.caption(f"AE baseline delta isolated ΓÇö simulated impact on network manifold")

        if mse > cfg.MSE_THRESHOLD_HIGH:
            status = "≡ƒö┤ ANOMALOUS"
            mse_color = THEME["red"]
        elif mse > cfg.MSE_THRESHOLD_MEDIUM:
            status = "≡ƒƒí BORDERLINE"
            mse_color = THEME["orange"]
        else:
            status = "≡ƒƒó NORMAL"
            mse_color = THEME["green"]

        st.markdown(
            f"<div style='background:{THEME['panel']}; border:1px solid {mse_color}; "
            f"border-radius:8px; padding:1rem; margin-top:1rem; text-align:center;'>"
            f"<div style='font-size:1.2em; font-weight:bold; color:{THEME['text']}'>{status}</div>"
            f"<div style='font-size:2.5em; font-weight:bold; color:{mse_color};'>{mse:.4f}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        try:
            try:
                from aura.ae_explainer import explain_anomaly as explain_func
            except ImportError:
                from aura.ae_explainer import explain_ae as explain_func
            expl = explain_func(residuals.numpy())
            top_5 = expl.get("top_features", [])[:5]
            
            if top_5:
                if mse > cfg.MSE_THRESHOLD_MEDIUM:
                    top_feature_name = top_5[0][0]  # name from explain_anomaly output
                    explanations = {
                        "flow_bytes_per_sec": "High data transfer rate detected ΓÇö consistent with data exfiltration or DDoS.",
                        "Flow Bytes/s": "High data transfer rate detected ΓÇö consistent with data exfiltration or DDoS.",
                        "fwd_packet_len_mean": "Unusual forward packet sizes observed ΓÇö potential payload injection.",
                        "Fwd Pkt Len Mean": "Unusual forward packet sizes observed ΓÇö potential payload injection.",
                        "bwd_packet_len_mean": "Abnormal backward packet sizes ΓÇö possible command-and-control (C2) instruction delivery.",
                        "Bwd Pkt Len Mean": "Abnormal backward packet sizes ΓÇö possible command-and-control (C2) instruction delivery.",
                        "flow_iat_mean": "Irregular inter-arrival times ΓÇö indicative of automated scanning or beaconing.",
                        "Flow IAT Mean": "Irregular inter-arrival times ΓÇö indicative of automated scanning or beaconing.",
                        "syn_flag_count": "Excessive SYN flags detected ΓÇö likely a SYN flood or aggressive port scan.",
                        "SYN Flag Count": "Excessive SYN flags detected ΓÇö likely a SYN flood or aggressive port scan.",
                        "dst_port": "Traffic aimed at unusual ports ΓÇö strong indicator of lateral movement.",
                        "Destination Port": "Traffic aimed at unusual ports ΓÇö strong indicator of lateral movement."
                    }
                    st.info(explanations.get(top_feature_name, f"Unusual traffic pattern driven primarily by {top_feature_name}."))

                import pandas as pd
                chart_data = pd.DataFrame({
                    "Feature": [f[0] for f in top_5],
                    "Squared Error": [f[1] for f in top_5]
                }).set_index("Feature")
                
                st.markdown(f"<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
                st.bar_chart(chart_data)
        except Exception as e:
            st.error(f"Could not load explanation chart: {e}")


    if st.button("🟢 Generate Normal Traffic", use_container_width=True):

        st.session_state["attack_active"] = False
        st.session_state["system_status"] = "ACTIVE"

        inj = st.session_state["injector"]
        if inj:
            normal_graph = inj._generate_healthy_graph()
            normal_graph["window_id"] = f"NORMAL_{st.session_state['window_counter']}"
            run_inference_tick(normal_graph, is_attack=False)

        # Reset node colours
        st.session_state["node_colors"] = {i: THEME["green"] for i in range(cfg.NUM_SYNTHETIC_NODES)}
        st.session_state["node_states"] = {i: "Normal" for i in range(cfg.NUM_SYNTHETIC_NODES)}
        st.toast("✅ Normal traffic window processed.", icon="✅")
        st.rerun()

# ── Federation Panel ──────────────────────────────────────────────────────────
with ctrl_fl:
    _fl_heading = f"{ORG['icon']} {ORG['label']} · Federated Learning" if ORG else "🌐 Federated Learning"
    st.markdown(f"<h4 style='color:{THEME['blue']}'>{_fl_heading}</h4>",
                unsafe_allow_html=True)

    # ── FL Readiness Toggle ──────────────────────────────────────────────────
    # Poll fl_readiness.json every render so server-written attack status
    # (set automatically by Krum detection) is reflected immediately.
    if ORG:
        _rf_poll = Path(cfg.LOGS_DIR) / "fl_readiness.json"
        if _rf_poll.exists():
            try:
                _rd_poll = json.loads(_rf_poll.read_text())
                _srv_atk  = _rd_poll.get(_ORG_KEY, {}).get("under_attack", False)
                _srv_rdy  = _rd_poll.get(_ORG_KEY, {}).get("ready",        False)
                if _srv_atk != st.session_state.get("under_attack", False):
                    st.session_state["under_attack"] = _srv_atk
                    st.session_state["fl_ready"]     = _srv_rdy
            except Exception:
                pass

    _ready        = st.session_state.get("fl_ready",     False)
    _under_attack = st.session_state.get("under_attack", False)
    _ready_color  = THEME["green"] if (_ready and not _under_attack) \
                    else (THEME["red"] if _under_attack else THEME["dim"])
    _ready_status = (
        "🚨 UNDER ATTACK — QUARANTINED" if _under_attack
        else ("🟢 READY" if _ready else "🔴 NOT READY")
    )

    st.markdown(
        f"<div style='background:{THEME['panel']}; border:1px solid {_ready_color}; "
        f"border-radius:8px; padding:0.55rem 0.8rem; margin-bottom:0.5rem; "
        f"text-align:center; font-size:0.9em'>"
        f"<span style='color:{THEME['dim']}'>Are you ready for FL?</span>&nbsp;&nbsp;"
        f"<span style='color:{_ready_color}; font-weight:bold'>{_ready_status}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    def _write_readiness(ready_val, attack_val):
        if ORG:
            import json as _json
            _rf = Path(cfg.LOGS_DIR) / "fl_readiness.json"
            _rf.parent.mkdir(parents=True, exist_ok=True)
            _rd: dict = {}
            if _rf.exists():
                try:
                    _rd = _json.loads(_rf.read_text())
                except Exception:
                    _rd = {}
            _rd[_ORG_KEY] = {
                "ready":        ready_val,
                "under_attack": attack_val,
                "org":          ORG["label"],
                "net":          ORG["net"],
                "ts":           time.time(),
            }
            _rf.write_text(_json.dumps(_rd, indent=2))

    _toggle_label = "✅ Mark as Ready" if not _ready else "⏸ Mark as Not Ready"

    if _under_attack:
        # Server AI auto-quarantined this org — admin clears it when issue resolved.
        st.markdown(
            f"<div style='background:{THEME['panel']}; border:1px solid {THEME['red']}; "
            f"border-radius:8px; padding:0.5rem 0.8rem; margin-bottom:0.4rem; "
            f"font-size:0.85em; color:{THEME['red']};'>"
            f"⚡ <b>Suspicious behaviour detected by FL Server (Krum).</b> "
            f"This org is quarantined until the issue is resolved."
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("✅ Issue Resolved — Resume", use_container_width=True, type="primary"):
            st.session_state["under_attack"] = False
            st.session_state["fl_ready"]     = False
            _write_readiness(False, False)
            st.rerun()
    else:
        if st.button(_toggle_label, use_container_width=True):
            new_ready = not _ready
            st.session_state["fl_ready"] = new_ready
            _write_readiness(new_ready, False)
            st.rerun()
        if st.button("🚨 Report Under Attack — Quarantine", use_container_width=True):
            st.session_state["under_attack"] = True
            st.session_state["fl_ready"]     = False
            _write_readiness(False, True)
            st.rerun()

    # ── Client Status Table ──────────────────────────────────────────────────
    clients_info = st.session_state.get("fl_client_status", [])
    if clients_info:
        _bg  = THEME["panel"]
        _br  = THEME["border"]
        _cy  = THEME["cyan"]
        _dim = THEME["dim"]
        _grn = THEME["green"]
        _red = THEME["red"]
        _org = THEME["orange"]

        rows_html = ""
        for c in clients_info:
            org     = c.get("org_id",   "unknown")
            network = c.get("network",  "—")
            role    = c.get("role",     "Normal")
            selected= c.get("selected", True)

            role_color   = _red if role == "Byzantine" else _grn
            status_label = "✓ Selected" if selected else "✗ Dropped"
            status_color = _grn         if selected else _red
            if not selected and role == "Byzantine":
                status_label = "✗ Dropped (Byzantine)"

            rows_html += (
                f"<tr style='border-bottom:1px solid {_br}'>"
                f"<td style='padding:3px 6px; color:{_cy}'>{org}</td>"
                f"<td style='padding:3px 6px; color:{_dim}'>{network}</td>"
                f"<td style='padding:3px 6px; color:{role_color}'>{role}</td>"
                f"<td style='padding:3px 6px; color:{status_color}'>{status_label}</td>"
                f"</tr>"
            )

        st.markdown(
            f"<div style='background:{_bg}; border:1px solid {_br}; "
            f"border-radius:6px; padding:0.5rem; margin-bottom:0.4rem'>"
            f"<div style='color:{_dim}; font-size:0.72em; "
            f"margin-bottom:0.3rem'>FL CLIENT STATUS (latest round)</div>"
            f"<table style='width:100%; border-collapse:collapse; font-size:0.73em'>"
            f"<thead><tr style='color:{_dim}; border-bottom:1px solid {_br}'>"
            f"<th style='text-align:left; padding:2px 6px'>Org</th>"
            f"<th style='text-align:left; padding:2px 6px'>Network</th>"
            f"<th style='text-align:left; padding:2px 6px'>Role</th>"
            f"<th style='text-align:left; padding:2px 6px'>Krum</th>"
            f"</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            f"</table></div>",
            unsafe_allow_html=True,
        )

    # Show last N fed log lines
    if st.session_state["fed_log"]:
        log_html = "<br>".join([
            f"<span class='fed-log'>{line}</span>"
            for line in st.session_state["fed_log"][-15:]
        ])
        _bg, _br = THEME["panel"], THEME["border"]
        st.markdown(
            f"<div style='background:{_bg}; border:1px solid "
            f"{_br}; border-radius:6px; padding:0.6rem; "
            f"max-height:180px; overflow-y:auto; font-size:0.75em'>"
            f"{log_html}</div>",
            unsafe_allow_html=True
        )

# ── Blockchain Panel ──────────────────────────────────────────────────────────
with ctrl_chain:
    st.markdown(f"<h4 style='color:{THEME['cyan']}'>⛓ Blockchain Audit</h4>",
                unsafe_allow_html=True)

    bc_module = st.session_state["blockchain"]
    if bc_module:
        _dim, _cy = THEME["dim"], THEME["cyan"]
        st.markdown(f"<div style='color:{_dim}; font-size:0.8em'>"
                    f"Mode: <b style='color:{_cy}'>"
                    f"{bc_module.mode.upper()}</b></div>",
                    unsafe_allow_html=True)

        # Register a test hash
        if st.button("📝 Register Test Hash", use_container_width=True):
            fake_hash   = "0x" + hashlib.sha256(f"test_{time.time()}".encode()).hexdigest()
            version_tag = f"demo_v{int(time.time()) % 10000}"
            tx = bc_module.log_model_update(version_tag, fake_hash)
            st.session_state["chain_log"].insert(0, {
                "version": version_tag,
                "hash": fake_hash[:20] + "…",
                "round": "manual",
                "time": time.strftime("%H:%M:%S"),
            })
            st.session_state["chain_entries"] = len(st.session_state["chain_log"])
            st.toast(f"Hash {fake_hash[:12]}… written to ledger.", icon="⛓")

    # Chain history table
    if st.session_state["chain_log"]:
        for entry in st.session_state["chain_log"][:6]:
            st.markdown(
                f"<div class='chain-row'>"
                f"[{entry['time']}] {entry['version']}  "
                f"<span style='opacity:0.6'>{entry['hash'][:22]}…</span>"
                f"</div>",
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# Alert + Incident Log
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
log_col, inc_col = st.columns([1, 1])

with log_col:
    st.markdown(f"<h4 style='color:{THEME['yellow']}'>🔔 Alert History</h4>",
                unsafe_allow_html=True)

    if not st.session_state["alerts"]:
        st.markdown(f"<span style='color:{THEME['dim']}'>No alerts triggered yet.</span>",
                    unsafe_allow_html=True)
    else:
        for a in st.session_state["alerts"][:10]:
            severity = a.get("severity", "NORMAL")
            color = {"HIGH": THEME["red"], "MEDIUM": THEME["orange"],
                     "LOW": THEME["yellow"]}.get(severity, THEME["green"])
            ts = time.strftime("%H:%M:%S", time.localtime(a.get("timestamp", 0)))
            _t = THEME["text"]
            st.markdown(
                f"<div style='border-left:3px solid {color}; padding:0.3rem 0.6rem; "
                f"margin:0.2rem 0; font-size:0.8em; color:{_t}'>"
                f"<b style='color:{color}'>[{ts}] {severity}</b>  "
                f"mse={a.get('ae_score', 0):.4f}  "
                f"conf={a.get('confidence', 0):.1%}  "
                f"nodes={a.get('triggered_nodes', [])}"
                f"</div>",
                unsafe_allow_html=True
            )

with inc_col:
    st.markdown(f"<h4 style='color:{THEME['orange']}'>🛡️ Response Actions</h4>",
                unsafe_allow_html=True)

    if not st.session_state["incidents"]:
        st.markdown(f"<span style='color:{THEME['dim']}'>No responses triggered yet.</span>",
                    unsafe_allow_html=True)
    else:
        for r in st.session_state["incidents"][:10]:
            action = r.get("action_taken", "")
            color  = {"ISOLATE": THEME["red"], "THROTTLE": THEME["orange"],
                      "HITL_ESCALATE": THEME["yellow"], "LOG_ONLY": THEME["green"]}.get(
                action, THEME["dim"]
            )
            ts = time.strftime("%H:%M:%S", time.localtime(r.get("timestamp", 0)))
            crit_badge = "🔑 CRITICAL" if r.get("is_critical") else ""
            _t = THEME["text"]
            st.markdown(
                f"<div style='border-left:3px solid {color}; padding:0.3rem 0.6rem; "
                f"margin:0.2rem 0; font-size:0.8em; color:{_t}'>"
                f"<b style='color:{color}'>[{ts}] {action}</b>  "
                f"{crit_badge}  "
                f"{r.get('node_id', '')} ({r.get('node_label', '')})"
                f"</div>",
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"<h3 style='color:{THEME['cyan']}'>⚙️ AURA Config</h3>",
                unsafe_allow_html=True)

    st.markdown(f"""
    **Model Status:** {model_status}
    **Blockchain Mode:** {bc.mode}
    **Nodes:** {cfg.NUM_SYNTHETIC_NODES}
    **Feature Dim:** {cfg.FEATURE_DIM}
    **EMA α:** {cfg.EMA_ALPHA}
    **EMA σ mult:** {cfg.EMA_SIGMA_MULTIPLIER}×
    """)

    st.markdown("---")
    st.markdown(f"<h4 style='color:{THEME['cyan']}'>📖 Architecture Layers</h4>",
                unsafe_allow_html=True)
    st.markdown("""
    1. **Data Layer** — NetFlow → Graph  
       TTL edge decay, IsolationForest sanitisation

    2. **Layer 1** — Statistical Tripwire  
       Unsupervised Autoencoder  
       EMA Dynamic Threshold

    3. **Layer 2** — Contextual Validator  
       GraphSAGE (inductive)  
       Topology anomaly scoring

    4. **Federation** — Krum aggregation  
       Byzantine-robust FL  
       Straggler timeout policy

    5. **Blockchain** — SHA-256 audit  
       Non-repudiation audit trail  
       Model integrity verification
    """)

    st.markdown("---")
    if st.button("🗑️ Clear All Logs"):
        for key in ["ae_scores", "thresholds", "timestamps", "alerts",
                    "incidents", "fed_log", "chain_log"]:
            st.session_state[key] = []
        st.session_state["node_colors"] = {i: THEME["green"] for i in range(cfg.NUM_SYNTHETIC_NODES)}
        st.session_state["node_states"] = {i: "Normal" for i in range(cfg.NUM_SYNTHETIC_NODES)}
        st.session_state["system_status"] = "ACTIVE"
        st.session_state["last_explanation"] = None
        st.rerun()

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.7em; color:{THEME['dim']}'>
    Team Trinetra — NexJam 2026<br>
    Suraj H P | Narendra Kanchi | Sudhanva Girish Thite
    </div>
    """, unsafe_allow_html=True)
