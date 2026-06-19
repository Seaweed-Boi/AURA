"""
fl_server_dashboard.py — AURA Federated Learning Server Dashboard
==================================================================

Dedicated server-side console showing:
  • All five org clients (Hospital / Bank / University / ISP / Retail)
  • Step-by-step FL pipeline animation (collect → FLTrust → aggregate → mint → broadcast)
  • Blockchain hash minting live feed
  • Per-round history table with FLTrust trust scores
  • Client hash verification outcome
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

_READINESS_FILE = Path(cfg.LOGS_DIR) / "fl_readiness.json"

def _read_readiness() -> dict:
    """Return {org_key: {ready, org, net, ts}} from shared file."""
    if not _READINESS_FILE.exists():
        return {}
    try:
        return json.loads(_READINESS_FILE.read_text())
    except Exception:
        return {}

def _write_readiness_server(org_key: str, under_attack: bool) -> None:
    """
    Server-side write to fl_readiness.json.
    Called automatically when FLTrust flags a Byzantine client so the org
    dashboard reflects quarantine status without any manual button press.
    """
    try:
        _READINESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        rd: dict = {}
        if _READINESS_FILE.exists():
            try:
                rd = json.loads(_READINESS_FILE.read_text())
            except Exception:
                rd = {}
        entry = dict(rd.get(org_key, {}))
        entry["under_attack"] = under_attack
        if under_attack:
            entry["ready"] = False   # force not-ready while quarantined
        entry["ts"] = time.time()
        rd[org_key] = entry
        _READINESS_FILE.write_text(json.dumps(rd, indent=2))
    except Exception:
        pass  # non-critical — federation continues regardless

# ─────────────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────────────
THEME = {
    "bg":       "#0d1117",
    "panel":    "#161b22",
    "panel2":   "#1c2430",
    "border":   "#30363d",
    "cyan":     "#58d1e8",
    "green":    "#3fb950",
    "red":      "#f85149",
    "orange":   "#d29922",
    "blue":     "#388bfd",
    "purple":   "#bc8cff",
    "yellow":   "#e3b341",
    "dim":      "#8b949e",
    "text":     "#c9d1d9",
}

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "AURA · FL Server Console",
    page_icon   = "⚙️",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  html, body, [class*="st-"] {{
    background-color: {THEME['bg']};
    color: {THEME['text']};
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }}
  .block-container {{ padding: 1rem 1.5rem; }}
  h1, h2, h3, h4 {{ color: {THEME['cyan']}; }}

  /* Client card */
  .client-card {{
    background: {THEME['panel']};
    border: 1px solid {THEME['border']};
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.4rem;
  }}
  .client-card.selected  {{ border-color: {THEME['green']}; }}
  .client-card.dropped   {{ border-color: {THEME['red']}; }}
  .client-card.byzantine {{ border-color: {THEME['orange']}; }}
  .client-card.idle      {{ border-color: {THEME['border']}; }}

  /* Pipeline step */
  .pipe-step {{
    background: {THEME['panel2']};
    border: 1px solid {THEME['border']};
    border-radius: 8px;
    padding: 0.5rem 0.7rem;
    text-align: center;
    font-size: 0.78em;
    transition: border-color 0.2s;
  }}
  .pipe-step.active  {{ border-color: {THEME['cyan']}; background: #1a2a38; }}
  .pipe-step.done    {{ border-color: {THEME['green']}; background: #0f2117; }}
  .pipe-step.pending {{ opacity: 0.45; }}

  /* Hash card */
  .hash-row {{
    background: {THEME['panel']};
    border-left: 3px solid {THEME['purple']};
    border-radius: 4px;
    padding: 0.35rem 0.6rem;
    margin: 0.25rem 0;
    font-size: 0.74em;
    font-family: monospace;
  }}
  .hash-row.final {{ border-left-color: {THEME['green']}; }}

  /* Round history row */
  .round-row {{
    display: flex; gap: 0.5rem;
    padding: 0.25rem 0.5rem;
    border-bottom: 1px solid {THEME['border']};
    font-size: 0.76em;
  }}
  .verify-ok   {{ color: {THEME['green']}; }}
  .verify-warn {{ color: {THEME['red']}; }}
  div[data-testid="stMetricValue"] {{ font-size: 1.6em; color: {THEME['cyan']}; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────

_ORGS = [
    {"id": "org_hospital_1",    "label": "Hospital",    "net": "192.168.1.0/24",  "role": "Normal"},
    {"id": "org_bank_2",        "label": "Bank",        "net": "10.0.1.0/24",     "role": "Normal"},
    {"id": "org_university_3",  "label": "University",  "net": "172.16.1.0/24",   "role": "Normal"},
    {"id": "org_isp_4",         "label": "ISP",         "net": "10.10.0.0/24",    "role": "Normal"},
    {"id": "org_retail_5",      "label": "Retail",      "net": "172.31.0.0/24",   "role": "Normal"},
]
_ORG_KEYS = ["hospital", "bank", "university", "isp", "retail"]

# Canonical mapping: org id → short key (used everywhere for active_orgs checks)
_ORG_ID_TO_KEY = {
    "org_hospital_1":   "hospital",
    "org_bank_2":       "bank",
    "org_university_3": "university",
    "org_isp_4":        "isp",
    "org_retail_5":     "retail",
}
_ORG_KEY_TO_ID = {v: k for k, v in _ORG_ID_TO_KEY.items()}

_PIPE_STEPS = [
    ("📥", "Collect\nWeights"),
    ("🔬", "FLTrust\nFilter"),
    ("🧮", "Aggregate"),
    ("⛓",  "Mint\nHash"),
    ("📡", "Broadcast\n+ Verify"),
]

def _init():
    defaults = {
        "fl_running":       False,
        "fl_done":          False,
        "round_results":    [],       # list of per-round dicts
        "hash_ledger":      [],       # blockchain entries (final rounds)
        "hash_local":       [],       # intermediate hash records
        "client_cards":     {o["id"]: {"status": "idle", "round": 0,
                                        "selected": None, "verified": None}
                             for o in _ORGS},
        "pipe_state":       [0] * len(_PIPE_STEPS),  # 0=pending,1=active,2=done
        "current_round":    0,
        "total_rounds":     cfg.FL_NUM_ROUNDS,
        "fl_log":           [],
        "fltrust_scores_hist": [],  # list of dicts per round (trust_scores from server)
        "global_hash":      None,
        "global_version":   None,
        "verify_results":   {},   # org_id → True/False
        "byzantine_org":    None,  # org key FLTrust flagged (low trust)
        "attack_idx":       None,  # index of client with injected attack data
        "active_orgs":      [],    # orgs that participated in the last FL run
        "quarantined_orgs": [],    # orgs blocked due to active attack
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ─────────────────────────────────────────────────────────────────────────────
# Helper — log line
# ─────────────────────────────────────────────────────────────────────────────
def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state["fl_log"].insert(0, f"[{ts}] {msg}")
    st.session_state["fl_log"] = st.session_state["fl_log"][:120]


# ─────────────────────────────────────────────────────────────────────────────
# FL Simulation with step-by-step UI updates
# ─────────────────────────────────────────────────────────────────────────────

def run_fl_with_animation(pipe_ph, card_placeholders, log_ph, ledger_ph,
                           metrics_ph, round_hist_ph):
    """
    Drive the FL simulation round-by-round, updating Streamlit placeholders
    at each pipeline step so the operator sees live progress.
    """
    from aura.fl_server import hash_model_weights, KrumFedAURA
    from aura.fl_client import create_mock_clients
    from aura.models import AURAModelBundle as MB
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitIns

    # ── Determine active orgs from shared readiness file ────────────────────────
    readiness = _read_readiness()
    all_org_keys = _ORG_KEYS
    # Exclude orgs that are: (a) not ready, OR (b) currently under active attack.
    # Under-attack orgs are quarantined — they must NOT contribute weights to FL
    # because their data is compromised. They are blocked before FL starts, not
    # FLTrust-flagged after aggregation (FLTrust is a last-resort signal; quarantine is proactive).
    quarantined = [k for k in all_org_keys
                   if readiness.get(k, {}).get("under_attack", False)]
    active_orgs = [k for k in all_org_keys
                   if readiness.get(k, {}).get("ready", False)
                   and k not in quarantined]
    # Fall back to all 5 if no readiness data exists (demo mode)
    if not active_orgs and not quarantined:
        active_orgs = all_org_keys
        _log("No readiness data found — using all 5 orgs (demo mode)")
    else:
        inactive = [k for k in all_org_keys if k not in active_orgs and k not in quarantined]
        if quarantined:
            _log(f"🚨 Quarantined (under attack — BLOCKED from FL): {[k.upper() for k in quarantined]}")
        _log(f"Active: {active_orgs}  |  Offline: {inactive if inactive else 'none'}")

    st.session_state["fl_running"]    = True
    st.session_state["fl_done"]       = False
    st.session_state["round_results"] = []
    st.session_state["hash_ledger"]   = []
    st.session_state["hash_local"]    = []
    st.session_state["fl_log"]        = []
    st.session_state["fltrust_scores_hist"] = []
    st.session_state["verify_results"] = {}
    st.session_state["current_round"] = 0

    # Reset client cards
    for o in _ORGS:
        st.session_state["client_cards"][o["id"]] = {
            "status": "idle", "round": 0, "selected": None, "verified": None,
        }

    # Use real blockchain logger so the hash is written to blockchain_fallback.jsonl
    # and verify_chain.py can cross-check it against hash_registry.json
    from aura.blockchain import AURABlockchainLogger
    bc_module = AURABlockchainLogger()

    n_rounds = cfg.FL_NUM_ROUNDS
    st.session_state["total_rounds"] = n_rounds
    st.session_state["active_orgs"]  = active_orgs
    st.session_state["quarantined_orgs"] = quarantined

    # Mark quarantined org cards immediately so they show as blocked
    for _qk in quarantined:
        _qid = _ORG_KEY_TO_ID[_qk]
        st.session_state["client_cards"][_qid]["status"] = "quarantined"

    # All orgs participate honestly by default — no attack injected.
    # Attack injection only happens when triggered externally (future feature).
    # FLTrust still runs and may flag a low-trust client due to random
    # weight init variance, but honest orgs are never auto-quarantined.
    attack_org        = None
    attack_client_arg = -1   # -1 = all honest

    clients, attack_idx = create_mock_clients(
        n_clients     = len(active_orgs),
        n_samples     = 300,
        org_ids       = active_orgs,
        attack_client = attack_client_arg,
    )
    st.session_state["byzantine_org"] = None
    st.session_state["attack_idx"]    = attack_idx
    _log("\u2705 All active orgs sending honest data this run")

    # Pass the live client count so aggregate_fit's quorum check never
    # abandons rounds when fewer than 5 orgs are active (e.g. one quarantined).
    n_active = len(active_orgs)
    strategy = KrumFedAURA(
        blockchain_module      = bc_module,
        num_rounds             = n_rounds,
        min_fit_clients        = n_active,
        min_available_clients  = n_active,
    )

    global_model  = MB()
    global_params = [p.detach().cpu().numpy() for p in global_model.parameters()]

    _log(f"Federation started — {n_rounds} rounds, {len(clients)} clients")

    for rnd in range(1, n_rounds + 1):
        st.session_state["current_round"] = rnd
        is_final = (rnd == n_rounds)
        _log(f"━━━  Round {rnd}/{n_rounds}  ━━━")

        # ── STEP 0: Collect Weights ───────────────────────────────────────
        _set_pipe(0, "active"); _render_pipe(pipe_ph); _render_clients(card_placeholders)

        _org_meta = {o["id"]: o for o in _ORGS}   # lookup by id
        fit_results = []
        for i, client in enumerate(clients):
            org_key   = active_orgs[i]
            org_info  = _org_meta.get(f"org_{org_key}_1") or next(
                (o for o in _ORGS if org_key in o["id"]), _ORGS[0])
            org_label = org_info["label"]
            org_id    = org_info["id"]
            st.session_state["client_cards"][org_id]["status"] = "sending"
            st.session_state["client_cards"][org_id]["round"]  = rnd
            _render_clients(card_placeholders)

            fit_ins = FitIns(
                parameters = ndarrays_to_parameters(global_params),
                config     = {"local_epochs": cfg.FL_LOCAL_EPOCHS, "round": rnd},
            )
            fit_res = client.fit(fit_ins)
            fit_results.append((None, fit_res))
            raw_loss = fit_res.metrics.get('train_loss', None)
            loss_str = f"{raw_loss:.4f}" if isinstance(raw_loss, (int, float)) else str(raw_loss)
            _log(f"  [{org_label}] Weights received — loss={loss_str}")

        time.sleep(0.35)
        _set_pipe(0, "done")

        # ── STEP 1: FLTrust (server-side trust scores in aggregate_fit) ────
        _set_pipe(1, "active"); _render_pipe(pipe_ph)
        _log(f"  [SERVER] Running FLTrust aggregation on {len(clients)} updates …")

        n_cl = len(clients)

        # ── STEP 2: Aggregate (+ blockchain mint on final round) ──────────
        _set_pipe(1, "done")
        _set_pipe(2, "active"); _render_pipe(pipe_ph)

        # Route through strategy.aggregate_fit — FLTrust, hash,
        # blockchain mint (bc_module) and registry write all in one call
        new_params, metrics = strategy.aggregate_fit(
            server_round = rnd,
            results      = fit_results,
            failures     = [],
        )

        selected_indices = metrics.get("fltrust_trusted_indices", [])
        dropped_indices  = list(metrics.get("fltrust_flagged_indices", []))
        trust_scores_round = metrics.get("trust_scores", [])

        st.session_state["fltrust_scores_hist"].append({
            "round":         rnd,
            "trust_scores":  [round(s, 4) for s in trust_scores_round],
            "selected":      selected_indices,
            "dropped":       dropped_indices,
        })

        # Set byzantine_org from whoever FLTrust flagged (low cosine trust)
        if dropped_indices:
            fltrust_byz_org = active_orgs[dropped_indices[0]]
            st.session_state["byzantine_org"] = fltrust_byz_org
        else:
            fltrust_byz_org = None

        # ── Server AI Auto-Quarantine ─────────────────────────────────────
        # Only quarantine when attack data was ACTUALLY injected AND FLTrust
        # flagged exactly that client. Never quarantine on honest-noise drops.
        _atk_idx     = st.session_state.get("attack_idx")
        _atk_injected = (_atk_idx is not None and isinstance(_atk_idx, int) and _atk_idx >= 0)
        fltrust_accurate = _atk_injected and (_atk_idx in dropped_indices)
        if fltrust_accurate and fltrust_byz_org and fltrust_byz_org not in quarantined:
            _write_readiness_server(fltrust_byz_org, under_attack=True)
            _log(f"  [SERVER-AI] \u26a1 Auto-quarantined {fltrust_byz_org.upper()} "
                 f"\u2014 FLTrust flagged low-trust (Byzantine) update. "
                 f"Org blocked from next FL run until issue resolved.")

        for i, org in enumerate(_ORGS):
            org_key = _ORG_ID_TO_KEY.get(org["id"], org["id"])
            is_quar = org_key in quarantined
            is_act  = org_key in active_orgs
            act_idx = active_orgs.index(org_key) if is_act else None
            is_sel  = is_act and (act_idx in selected_indices)
            if is_quar:
                new_status = "quarantined"  # preserve — never overwrite with offline
            elif is_sel:
                new_status = "selected"
            elif is_act:
                new_status = "dropped"
            else:
                new_status = "offline"
            st.session_state["client_cards"][org["id"]]["status"]   = new_status
            st.session_state["client_cards"][org["id"]]["selected"] = is_sel
        _render_clients(card_placeholders)

        sel_labels  = [active_orgs[i].capitalize() for i in selected_indices]
        drop_labels = [active_orgs[i].capitalize() for i in dropped_indices]
        if fltrust_byz_org:
            accuracy_tag = " ✓ CORRECT — real attacker caught" if fltrust_accurate else " ⚠ MISSED — flagged honest node"
            byz_note = f" ⚡ FLTrust flagged {fltrust_byz_org.upper()}{accuracy_tag}"
        else:
            byz_note = " ✓ no outlier detected"
        _log(f"  [FLTRUST] Trusted: {sel_labels}  |  Flagged: {drop_labels}{byz_note}")
        _log(f"  [FLTRUST] Trust scores → {['%.3f' % s for s in trust_scores_round]}")

        model_version = metrics.get("model_version", f"v{rnd}.{rnd}")
        model_hash    = metrics.get("model_hash", "")
        if new_params is not None:
            global_params = parameters_to_ndarrays(new_params)

        st.session_state["global_hash"]    = model_hash
        st.session_state["global_version"] = model_version

        _log(f"  [SERVER] Global model {model_version} aggregated from "
             f"{len(selected_indices)} updates")
        time.sleep(0.3)
        _set_pipe(2, "done")

        # ── STEP 3: Mint Hash ─────────────────────────────────────────────
        _set_pipe(3, "active"); _render_pipe(pipe_ph)

        if is_final:
            # aggregate_fit already minted hash + wrote registry via bc_module
            st.session_state["hash_ledger"].append({
                "round": rnd, "version": model_version,
                "hash":  model_hash, "time": time.strftime("%H:%M:%S"),
                "final": True,
            })
            _log(f"  [BLOCKCHAIN] ✅ Hash MINTED — {model_version}")
            _log(f"  [BLOCKCHAIN]   SHA-256: {model_hash[:32]}…")
        else:
            st.session_state["hash_local"].append({
                "round": rnd, "version": model_version,
                "hash":  model_hash, "time": time.strftime("%H:%M:%S"),
            })
            _log(f"  [HASH] Intermediate hash recorded (not minted) — {model_hash[:20]}…")

        _render_ledger(ledger_ph)
        time.sleep(0.35)
        _set_pipe(3, "done")

        # ── STEP 4: Broadcast + Verify ────────────────────────────────────
        _set_pipe(4, "active"); _render_pipe(pipe_ph)
        _log(f"  [SERVER] Broadcasting global model {model_version} to all clients …")
        time.sleep(0.2)

        if is_final:
            client_received_hash = hash_model_weights(global_params)
            for i, org in enumerate(_ORGS):
                org_key = _ORG_ID_TO_KEY.get(org["id"], org["id"])
                if org_key not in active_orgs:
                    continue
                match = (client_received_hash == model_hash)
                st.session_state["verify_results"][org["id"]] = match
                st.session_state["client_cards"][org["id"]]["verified"] = match
                status = "✓ MATCH — deployed" if match else "✗ MISMATCH — rejected"
                _log(f"  [{org['label']}] Verify: {status}")
            _render_clients(card_placeholders)
        else:
            _log(f"  [CLIENTS] Round {rnd} model received. "
                 f"Verification on final round only.")

        _set_pipe(4, "done"); _render_pipe(pipe_ph)

        # ── Persist round result ──────────────────────────────────────────
        st.session_state["round_results"].append({
            "round":                   rnd,
            "model_version":           model_version,
            "model_hash":              model_hash,
            "fltrust_trusted_indices": selected_indices,
            "fltrust_flagged_indices": dropped_indices,
            "trust_scores":            trust_scores_round,
            "is_final":                is_final,
        })

        _render_metrics(metrics_ph)
        _render_round_hist(round_hist_ph)
        time.sleep(0.4)

        # Reset pipe for next round
        if rnd < n_rounds:
            st.session_state["pipe_state"] = [0] * len(_PIPE_STEPS)

    st.session_state["fl_running"] = False
    st.session_state["fl_done"]    = True
    _log(f"✅ Federation complete — {n_rounds} rounds finished")
    _log(f"   Final hash: {st.session_state['global_hash'][:24] if st.session_state['global_hash'] else 'N/A'}…")


# ─────────────────────────────────────────────────────────────────────────────
# Renderers — write into placeholders
# ─────────────────────────────────────────────────────────────────────────────

def _set_pipe(idx: int, state: str):
    mapping = {"pending": 0, "active": 1, "done": 2}
    st.session_state["pipe_state"][idx] = mapping[state]


def _render_pipe(ph):
    states = st.session_state["pipe_state"]
    state_cls = ["pending", "active", "done"]
    state_ico = ["⏳", "🔄", "✅"]

    cols_html = ""
    for i, (icon, label) in enumerate(_PIPE_STEPS):
        cls = state_cls[states[i]]
        ico = state_ico[states[i]]
        arrow = " <span style='color:#8b949e; font-size:1.2em'>→</span> " if i < len(_PIPE_STEPS) - 1 else ""
        cols_html += (
            f"<span class='pipe-step {cls}' style='display:inline-block; "
            f"min-width:90px; margin:0 2px'>"
            f"{ico} {icon}<br><b>{label}</b></span>{arrow}"
        )

    bg, br = THEME["panel"], THEME["border"]
    ph.markdown(
        f"<div style='background:{bg}; border:1px solid {br}; border-radius:8px; "
        f"padding:0.65rem 1rem; text-align:center; font-size:0.8em'>"
        f"{cols_html}</div>",
        unsafe_allow_html=True,
    )


def _render_clients(card_phs):
    cards = st.session_state["client_cards"]
    status_label = {
        "idle":        ("Idle",                         THEME["dim"],    "idle"),
        "sending":     ("Sending\u2026",                THEME["yellow"], "idle"),
        "selected":    ("\u2713 Selected",              THEME["green"],  "selected"),
        "dropped":     ("\u2717 Dropped",               THEME["red"],    "dropped"),
        "offline":     ("\u23f8 Not Ready",             THEME["dim"],    "idle"),
        "quarantined": ("\U0001f6ab Quarantined",       THEME["red"],    "dropped"),
    }
    # Resolve Byzantine and quarantine dynamically from session state
    _fltrust_byz = st.session_state.get("byzantine_org")
    _quarantined = st.session_state.get("quarantined_orgs", [])
    for i, org in enumerate(_ORGS):
        c       = cards[org["id"]]
        raw     = c.get("status", "idle")
        org_key = _ORG_ID_TO_KEY.get(org["id"])
        lbl, color, css = status_label.get(raw, ("Idle", THEME["dim"], "idle"))
        if org_key in _quarantined:
            role_label = "🚨 Under Attack — Blocked"
            role_color = THEME["red"]
        elif _fltrust_byz and _fltrust_byz == org_key:
            role_label = "⚠ FLTrust-flagged"
            role_color = THEME["orange"]
        else:
            role_label = "✓ Normal"
            role_color = THEME["green"]
        vfy = c.get("verified")
        vfy_html = ""
        _vfy_grn = THEME["green"]
        _vfy_red = THEME["red"]
        if vfy is True:
            vfy_html = f"<div style='color:{_vfy_grn}; font-size:0.8em'>&#9939; Hash verified &#10003;</div>"
        elif vfy is False:
            vfy_html = f"<div style='color:{_vfy_red}; font-size:0.8em'>&#9939; Hash MISMATCH &#10007;</div>"

        card_phs[i].markdown(
            f"<div class='client-card {css}'>"
            f"<div style='font-size:1.0em; font-weight:bold; color:{THEME['cyan']}'>"
            f"{['🏥','🏦','🎓'][i]} {org['label']}</div>"
            f"<div style='font-size:0.77em; color:{THEME['dim']}'>{org['id']}</div>"
            f"<div style='font-size:0.77em; color:{THEME['dim']}'>🌐 {org['net']}</div>"
            f"<div style='font-size:0.78em; color:{role_color}; margin-top:2px'>"
            f"{role_label}</div>"
            f"<div style='font-size:0.85em; color:{color}; margin-top:4px'>"
            f"<b>{lbl}</b></div>"
            f"{vfy_html}"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_ledger(ph):
    entries = st.session_state["hash_ledger"]
    local   = st.session_state["hash_local"]
    bg, br = THEME["panel"], THEME["border"]

    rows = ""
    for e in reversed(entries):
        rows += (
            f"<div class='hash-row final'>"
            f"<span style='color:{THEME['green']}'>⛓ MINTED</span> "
            f"<b style='color:{THEME['cyan']}'>{e['version']}</b>  "
            f"<span style='color:{THEME['text']}'>{e['hash'][:28]}…</span>  "
            f"<span style='color:{THEME['dim']}'>[{e['time']}]</span>"
            f"</div>"
        )
    for e in reversed(local[-4:]):
        rows += (
            f"<div class='hash-row'>"
            f"<span style='color:{THEME['dim']}'>📋 local </span> "
            f"<b style='color:{THEME['dim']}'>{e['version']}</b>  "
            f"<span style='color:{THEME['dim']}'>{e['hash'][:28]}…</span>  "
            f"<span style='color:{THEME['dim']}'>[{e['time']}]</span>"
            f"</div>"
        )

    if not rows:
        rows = f"<div style='color:{THEME['dim']}; font-size:0.8em'>No hashes yet …</div>"

    ph.markdown(
        f"<div style='background:{bg}; border:1px solid {br}; border-radius:8px; "
        f"padding:0.65rem; min-height:80px'>"
        f"<div style='color:{THEME['dim']}; font-size:0.72em; margin-bottom:0.4rem'>"
        f"BLOCKCHAIN LEDGER</div>{rows}</div>",
        unsafe_allow_html=True,
    )


def _render_metrics(ph):
    rr = st.session_state["round_results"]
    if not rr:
        return
    last = rr[-1]
    c1, c2, c3, c4 = ph.columns(4)
    c1.metric("Rounds Done",  f"{len(rr)} / {st.session_state['total_rounds']}")
    _nt = len(last.get("fltrust_trusted_indices", []))
    _nf = len(last.get("fltrust_flagged_indices", []))
    c2.metric("FLTrust trusted", f"{_nt} / 3")
    c3.metric("FLTrust flagged", str(_nf))
    status = "✅ Done" if st.session_state["fl_done"] else "🔄 Running"
    c4.metric("Status", status)


def _render_round_hist(ph):
    rr = st.session_state["round_results"]
    bg, br = THEME["panel"], THEME["border"]
    if not rr:
        return

    header = (
        f"<tr style='color:{THEME['dim']}; border-bottom:1px solid {br}'>"
        f"<th style='padding:3px 8px'>Round</th>"
        f"<th>Version</th>"
        f"<th>✓ Kept</th>"
        f"<th>✗ Dropped</th>"
        f"<th>Trust scores</th>"
        f"<th>Hash (truncated)</th>"
        f"<th>On-Chain</th>"
        f"</tr>"
    )
    rows = ""
    for r in rr:
        chain_mark = (f"<span style='color:{THEME['green']}'>⛓ MINTED</span>"
                      if r["is_final"]
                      else f"<span style='color:{THEME['dim']}'>— local</span>")
        scores_str = "  ".join([f"{s:.3f}" for s in r.get("trust_scores", [])])
        rows += (
            f"<tr style='border-bottom:1px solid {br}; font-size:0.77em'>"
            f"<td style='padding:3px 8px; color:{THEME['cyan']}'>{r['round']}</td>"
            f"<td style='color:{THEME['text']}'>{r['model_version']}</td>"
            f"<td style='color:{THEME['green']}'>{len(r.get('fltrust_trusted_indices', []))}</td>"
            f"<td style='color:{THEME['red']}'>{len(r.get('fltrust_flagged_indices', []))}</td>"
            f"<td style='color:{THEME['dim']}; font-family:monospace'>{scores_str}</td>"
            f"<td style='color:{THEME['dim']}; font-family:monospace'>{r['model_hash'][:22]}…</td>"
            f"<td>{chain_mark}</td>"
            f"</tr>"
        )

    ph.markdown(
        f"<div style='background:{bg}; border:1px solid {br}; border-radius:8px; "
        f"padding:0.65rem; overflow-x:auto'>"
        f"<div style='color:{THEME['dim']}; font-size:0.72em; margin-bottom:0.4rem'>"
        f"ROUND HISTORY</div>"
        f"<table style='width:100%; border-collapse:collapse'>"
        f"<thead>{header}</thead><tbody>{rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _render_log(ph):
    lines = st.session_state["fl_log"][:24]
    bg, br = THEME["panel"], THEME["border"]
    body = "<br>".join(
        f"<span style='color:{THEME['dim']}'>{l}</span>" for l in lines
    ) or f"<span style='color:{THEME['dim']}'>Waiting for FL run…</span>"
    ph.markdown(
        f"<div style='background:{bg}; border:1px solid {br}; border-radius:8px; "
        f"padding:0.65rem; max-height:240px; overflow-y:auto; font-size:0.75em; "
        f"font-family:monospace'>{body}</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

# ── Header ───────────────────────────────────────────────────────────────────
rnd_cur   = st.session_state["current_round"]
rnd_total = st.session_state["total_rounds"]
run_state = ("🔄 RUNNING" if st.session_state["fl_running"]
             else ("✅ COMPLETE" if st.session_state["fl_done"] else "⏹ IDLE"))
run_color = (THEME["yellow"]  if st.session_state["fl_running"]
             else (THEME["green"] if st.session_state["fl_done"] else THEME["dim"]))

st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center;
            background:{THEME['panel']}; border:1px solid {THEME['border']};
            border-radius:10px; padding:0.8rem 1.5rem; margin-bottom:0.8rem">
  <div>
    <span style="font-size:1.5em; font-weight:bold; color:{THEME['cyan']}">
      ⚙️ AURA · FL Server Console
    </span>
    <span style="color:{THEME['dim']}; margin-left:0.8em; font-size:0.82em">
      FLTrust-Aggregated Federated Learning  ·  Blockchain-Audited
    </span>
  </div>
  <div style="text-align:right">
    <span style="color:{run_color}; font-weight:bold; font-size:1.0em">
      ● {run_state}
    </span>
    <span style="color:{THEME['dim']}; margin-left:1em; font-size:0.78em">
      Round {rnd_cur}/{rnd_total}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Metrics row (placeholder — updated during run) ───────────────────────────
metrics_ph = st.empty()

# Pre-fill initial metrics display
with metrics_ph.container():
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rounds Done",   f"0 / {rnd_total}")
    c2.metric("FLTrust trusted", "— / 3")
    c3.metric("FLTrust flagged", "—")
    c4.metric("Status",        run_state)

st.markdown("---")

# ── Live Org Readiness Panel ──────────────────────────────────────────────────
_readiness_hdr_col, _readiness_btn_col = st.columns([4, 1])
with _readiness_hdr_col:
    st.markdown(
        f"<h4 style='color:{THEME['green']}'>📡 Org Node Readiness</h4>",
        unsafe_allow_html=True,
    )
with _readiness_btn_col:
    st.button("🔄 Refresh", key="_refresh_readiness", use_container_width=True)

_ORG_ICONS = {"hospital": "🏥", "bank": "🏦", "university": "🎓"}
_readiness_data  = _read_readiness()
_byz_last        = st.session_state.get("byzantine_org")
_quarantined_now = st.session_state.get("quarantined_orgs", [])
_rd_cols = st.columns(3)
for _ri, _org_k in enumerate(["hospital", "bank", "university"]):
    _info         = _readiness_data.get(_org_k, {})
    _ready        = _info.get("ready", False)
    _is_quarantine= _info.get("under_attack", False)
    _icon         = _ORG_ICONS[_org_k]
    _label        = _org_k.capitalize()
    _net          = _info.get("net", "—")

    if _is_quarantine:
        _pill_color = THEME["red"]
        _pill_text  = "🚨 QUARANTINED — Under Attack"
    elif _ready:
        _pill_color = THEME["green"]
        _pill_text  = "🟢 READY"
    else:
        _pill_color = THEME["dim"]
        _pill_text  = "🔴 NOT READY"

    _byz_badge = (
        f"<span style='background:{THEME['yellow']};color:#000;border-radius:4px;"
        f"padding:1px 6px;font-size:0.75em;margin-left:6px;'>⚡ FLTrust-flagged</span>"
        if _byz_last == _org_k else ""
    )
    with _rd_cols[_ri]:
        st.markdown(
            f"<div style='border:2px solid {_pill_color};border-radius:8px;"
            f"padding:10px 14px;background:{THEME['panel']};margin-bottom:8px'>"
            f"<b style='font-size:1.05em'>{_icon} {_label}</b>{_byz_badge}<br>"
            f"<span style='color:{_pill_color};font-weight:600'>{_pill_text}</span><br>"
            f"<span style='color:{THEME['dim']};font-size:0.78em'>{_net}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

if _byz_last:
    st.markdown(
        f"<div style='color:{THEME['yellow']};font-size:0.85em;margin-bottom:8px'>"
        f"⚡ FLTrust-flagged suspicious node last run: <b>{_byz_last.upper()}</b>"
        f" — this client's gradient had low cosine trust vs the server root update and was down-weighted in aggregation."
        f"</div>",
        unsafe_allow_html=True,
    )

if _quarantined_now:
    st.markdown(
        f"<div style='color:{THEME['red']};font-size:0.85em;margin-bottom:8px;"
        f"border:1px solid {THEME['red']};border-radius:6px;padding:6px 10px'>"
        f"🚨 <b>Quarantined orgs blocked from FL this run: "
        f"{', '.join(k.upper() for k in _quarantined_now)}</b>"
        f" — attack must end before they can rejoin federation."
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Client Cards Row ─────────────────────────────────────────────────────────
st.markdown(f"<h4 style='color:{THEME['blue']}'>🖥 Participating Clients</h4>",
            unsafe_allow_html=True)
card_cols = st.columns(3)
card_placeholders = []
for col in card_cols:
    card_placeholders.append(col.empty())

# Initial render
_render_clients(card_placeholders)

st.markdown("---")

# ── Pipeline Animation ────────────────────────────────────────────────────────
st.markdown(f"<h4 style='color:{THEME['blue']}'>⚡ FL Pipeline</h4>",
            unsafe_allow_html=True)
pipe_ph = st.empty()
_render_pipe(pipe_ph)

st.markdown("---")

# ── Control ──────────────────────────────────────────────────────────────────
btn_col, info_col = st.columns([1, 3])
with btn_col:
    run_btn = st.button(
        "▶ Run Federated Learning",
        use_container_width=True,
        disabled=st.session_state["fl_running"],
        type="primary",
    )
with info_col:
    st.markdown(
        f"<div style='color:{THEME['dim']}; font-size:0.82em; padding-top:0.6rem'>"
        f"3 clients  ·  {rnd_total} rounds  ·  FLTrust (cosine trust vs server root)  "
        f"·  1 blockchain mint (final round only)"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Blockchain Ledger ─────────────────────────────────────────────────────────
st.markdown(f"<h4 style='color:{THEME['purple']}'>⛓ Blockchain Ledger</h4>",
            unsafe_allow_html=True)
ledger_ph = st.empty()
_render_ledger(ledger_ph)

st.markdown("---")

# ── Round History ─────────────────────────────────────────────────────────────
st.markdown(f"<h4 style='color:{THEME['cyan']}'>📊 Round History</h4>",
            unsafe_allow_html=True)
round_hist_ph = st.empty()
_render_round_hist(round_hist_ph)

st.markdown("---")

# ── FL Log ────────────────────────────────────────────────────────────────────
st.markdown(f"<h4 style='color:{THEME['dim']}'>📋 Server Log</h4>",
            unsafe_allow_html=True)
log_ph = st.empty()
_render_log(log_ph)

# ─────────────────────────────────────────────────────────────────────────────
# Run button handler
# ─────────────────────────────────────────────────────────────────────────────

if run_btn and not st.session_state["fl_running"]:
    run_fl_with_animation(
        pipe_ph            = pipe_ph,
        card_placeholders  = card_placeholders,
        log_ph             = log_ph,
        ledger_ph          = ledger_ph,
        metrics_ph         = metrics_ph,
        round_hist_ph      = round_hist_ph,
    )
    _render_log(log_ph)
    st.rerun()

# ── Auto-refresh log while idle ───────────────────────────────────────────────
if st.session_state["fl_done"] and st.session_state["round_results"]:
    _render_round_hist(round_hist_ph)
    _render_ledger(ledger_ph)
    _render_metrics(metrics_ph)
