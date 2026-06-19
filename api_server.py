"""
api_server.py — AURA REST API Backend for React+Vite Dashboard
==============================================================
Replaces Streamlit as the UI backend. Serves all dashboard state and actions.

Start with:  python api_server.py
Frontend:    cd frontend && npm run dev
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory

import sys
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from aura.dashboard_service import DashboardService, THEME, ORG_PROFILES
from aura.fl_dashboard_service import FLDashboardService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

_FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

# ─────────────────────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────────────────────

@app.after_request
def _add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    return response


@app.route("/api/<path:path>", methods=["OPTIONS"])
def _options(path):
    return jsonify({}), 200


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard state & actions
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok"})


@app.route("/api/state", methods=["GET"])
def api_state():
    return jsonify(DashboardService.get().get_state())


@app.route("/api/attack/<attack_type>", methods=["POST"])
def api_attack(attack_type: str):
    svc = DashboardService.get()
    valid = {"ddos", "portscan", "lateral", "exfil", "web"}
    if attack_type not in valid:
        return jsonify({"error": f"Unknown attack type: {attack_type}"}), 400
    result = svc.inject_attack(attack_type)
    return jsonify({"status": "ok", **result, "state": svc.get_state()})


@app.route("/api/normal", methods=["POST"])
def api_normal():
    svc = DashboardService.get()
    svc.inject_normal()
    return jsonify({"status": "ok", "state": svc.get_state()})


@app.route("/api/federation/run", methods=["POST"])
def api_federation():
    svc = DashboardService.get()
    result = svc.run_federation()
    return jsonify({**result, "state": svc.get_state()})


@app.route("/api/blockchain/register", methods=["POST"])
def api_blockchain_register():
    svc = DashboardService.get()
    entry = svc.register_test_hash()
    return jsonify({"status": "ok", "entry": entry, "state": svc.get_state()})


@app.route("/api/blockchain/verify", methods=["GET"])
def api_blockchain_verify():
    svc = DashboardService.get()
    result = svc.verify_chain()
    return jsonify(result)


@app.route("/api/logs/clear", methods=["POST"])
def api_clear_logs():
    svc = DashboardService.get()
    svc.clear_logs()
    return jsonify({"status": "ok", "state": svc.get_state()})


@app.route("/api/fl/ready", methods=["POST"])
def api_fl_ready():
    data = request.get_json(force=True, silent=True) or {}
    ready = bool(data.get("ready", True))
    svc = DashboardService.get()
    svc.set_fl_ready(ready)
    return jsonify({"status": "ok", "state": svc.get_state()})


@app.route("/api/fl/under-attack", methods=["POST"])
def api_fl_under_attack():
    svc = DashboardService.get()
    svc.set_under_attack(True)
    return jsonify({"status": "ok", "state": svc.get_state()})


@app.route("/api/fl/resolved", methods=["POST"])
def api_fl_resolved():
    svc = DashboardService.get()
    svc.set_under_attack(False)
    svc.set_fl_ready(False)
    return jsonify({"status": "ok", "state": svc.get_state()})


@app.route("/api/config", methods=["GET"])
def api_config():
    return jsonify({
        "theme": THEME,
        "org_profiles": ORG_PROFILES,
        "critical_allowlist": cfg.CRITICAL_ALLOWLIST,
        "num_nodes": cfg.NUM_SYNTHETIC_NODES,
        "refresh_ms": cfg.DASHBOARD_REFRESH_INTERVAL_MS,
    })


# ─────────────────────────────────────────────────────────────────────────────
# FL Server Console
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/fl-server/state", methods=["GET"])
def api_fl_server_state():
    return jsonify(FLDashboardService.get().get_state())


@app.route("/api/fl-server/run", methods=["POST"])
def api_fl_server_run():
    fl = FLDashboardService.get()
    fl.run_simulation()
    return jsonify({"status": "ok", "state": fl.get_state()})


# ─────────────────────────────────────────────────────────────────────────────
# Custom injection (existing endpoints)
# ─────────────────────────────────────────────────────────────────────────────

def _build_node_registry() -> list:
    nodes = []
    for i in range(cfg.NUM_SYNTHETIC_NODES):
        node_id = f"node_{i}"
        label = cfg.CRITICAL_ALLOWLIST.get(node_id, f"Host-{i:02d}")
        nodes.append({
            "id": node_id,
            "label": label,
            "index": i,
            "critical": node_id in cfg.CRITICAL_ALLOWLIST,
        })
    return nodes


NODE_REGISTRY = _build_node_registry()
_NODE_ID_SET = {n["id"] for n in NODE_REGISTRY}

BLOCKED_PATTERNS = ["os.system", "subprocess", "import os", "import sys"]
_AE_CACHE = None


def _check_script_safety(script: str):
    for pattern in BLOCKED_PATTERNS:
        if pattern in script:
            return False, pattern
    return True, None


def _get_autoencoder():
    global _AE_CACHE
    if _AE_CACHE is not None:
        return _AE_CACHE
    from aura.models import FlowAutoencoder, AURAModelBundle
    ae = FlowAutoencoder()
    bundle_path = Path(cfg.MODELS_DIR) / "aura_bundle.pth"
    if bundle_path.exists():
        try:
            bundle = AURAModelBundle()
            bundle.load_state_dict(torch.load(str(bundle_path), map_location="cpu"))
            ae = bundle.autoencoder
        except Exception as e:
            logger.warning(f"[AE] Bundle load failed: {e}")
    _AE_CACHE = ae.eval()
    return _AE_CACHE


def _run_inject_inference(target_node: str, node_index: int, attack_type: str = "custom") -> float:
    from aura.ae_explainer import explain_ae
    ae = _get_autoencoder()
    F = cfg.FEATURE_DIM
    n_e = 40
    profiles = cfg.ATTACK_CORRUPTION_PROFILES
    feat_map = cfg.FEATURE_INDEX_MAP
    norm_type = attack_type.lower().replace("-", "_")
    profile = profiles.get(norm_type, profiles["custom"])
    features = np.random.uniform(0.3, 0.5, (n_e, F)).astype(np.float32)
    for feat_name, (lo, hi) in profile.items():
        idx = feat_map.get(feat_name)
        if idx is None or idx >= F:
            continue
        features[:, idx] = np.random.uniform(lo, hi, n_e)
    edge_attr = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        x_hat, _ = ae(edge_attr)
        batch_mse = float(((edge_attr - x_hat) ** 2).mean())
        per_feat_sq = ((edge_attr - x_hat) ** 2).mean(dim=0).numpy()
    per_feat_abs = np.abs(edge_attr.numpy() - x_hat.numpy()).mean(axis=0)
    expl = explain_ae(per_feat_abs)
    feat_mean = edge_attr.mean(dim=0).numpy()
    xhat_mean = x_hat.mean(dim=0).detach().numpy()
    top_features_out = []
    for fname, fabs, fidx in expl["top_features"]:
        top_features_out.append({
            "name": fname,
            "error": round(float(per_feat_sq[fidx]), 4),
            "observed": round(float(feat_mean[fidx]), 4),
            "baseline": round(float(xhat_mean[fidx]), 4),
        })
    result = {
        "node": target_node,
        "attack_type": norm_type,
        "mse": round(batch_mse, 4),
        "inferred_attack": expl["inferred_attack"],
        "match_score": expl["match_score"],
        "top_features": top_features_out,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    expl_path = Path(cfg.LOGS_DIR) / "last_explanation.json"
    expl_path.parent.mkdir(parents=True, exist_ok=True)
    expl_path.write_text(json.dumps(result, indent=2))
    return batch_mse


@app.route("/api/nodes", methods=["GET"])
def api_nodes():
    return jsonify(NODE_REGISTRY)


@app.route("/api/inject_custom", methods=["POST"])
def api_inject_custom():
    data = request.get_json(force=True, silent=True) or {}
    script = str(data.get("script", "")).strip()
    target_node = str(data.get("target_node", "")).strip()
    attack_type = str(data.get("attack_type", "custom")).strip() or "custom"

    if target_node not in _NODE_ID_SET:
        return jsonify({"error": f"Node '{target_node}' not found."}), 400
    if not script:
        return jsonify({"error": "Script content cannot be empty."}), 400
    safe, blocked_pattern = _check_script_safety(script)
    if not safe:
        return jsonify({"error": f"Blocked: {blocked_pattern}"}), 400

    node_info = next((n for n in NODE_REGISTRY if n["id"] == target_node), {})
    event = {
        "tag": "CUSTOM_INJECT",
        "timestamp": time.time(),
        "window_id": f"CUSTOM_{target_node}_{int(time.time())}",
        "target_node": target_node,
        "node_label": node_info.get("label", "Unknown"),
        "severity": "MEDIUM",
        "confidence": 0.0,
        "ae_score": 0.0,
    }
    try:
        log_path = Path(cfg.ALERT_LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.error(f"[INJECT] Log write failed: {e}")

    _node_index = node_info.get("index", 0)
    pending_path = Path(cfg.LOGS_DIR) / "pending_inject.json"
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text(json.dumps({
        "target_node": target_node,
        "node_index": _node_index,
        "timestamp": time.time(),
        "mse": 0.0,
    }))

    try:
        mse = _run_inject_inference(target_node, _node_index, attack_type)
        pd = json.loads(pending_path.read_text())
        pd["mse"] = round(mse, 4)
        pending_path.write_text(json.dumps(pd))
    except Exception as e:
        logger.error(f"[INJECT] AE inference failed: {e}")
        mse = 0.0

    svc = DashboardService.get()
    svc.poll_pending_inject()

    return jsonify({
        "status": "ok",
        "message": f"Custom script queued for {target_node}",
        "target_node": target_node,
        "mse": mse,
        "state": svc.get_state(),
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# Production: serve built React static files (npm run build in frontend/)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path: str):
    if path.startswith("api"):
        return jsonify({"error": "Not found"}), 404
    if _FRONTEND_DIST.exists():
        target = _FRONTEND_DIST / path
        if path and target.is_file():
            return send_from_directory(_FRONTEND_DIST, path)
        return send_from_directory(_FRONTEND_DIST, "index.html")
    return jsonify({
        "message": "AURA API running. Dev UI: cd frontend && npm run dev",
        "endpoints": ["/api/state", "/api/fl-server/state"],
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    print("=" * 58)
    print("  AURA API Server - port 5001")
    print("  React UI: cd frontend && npm run dev")
    print("=" * 58)
    # Pre-warm ML pipeline so first UI poll is fast
    try:
        DashboardService.get()
        print("  Dashboard service: READY")
    except Exception as exc:
        print(f"  Dashboard service warmup warning: {exc}")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
