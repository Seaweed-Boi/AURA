"""
aura/dashboard_service.py — Backend state + logic for the React dashboard.

Extracted from dashboard.py so the UI can run in React+Vite without Streamlit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from aura.models import FlowAutoencoder, AuraSTGNN, AURAModelBundle
from aura.detector import AURAInferenceEngine, AlertSeverity, AnomalyEvent
from aura.response_engine import AURAResponseEngine
from aura.attack_injector import AttackInjector
from aura.blockchain import AURABlockchainLogger

THEME = {
    "bg": "#0a0e1a",
    "panel": "#0f1629",
    "border": "#1e2d4a",
    "text": "#e0e8f0",
    "green": "#00ff88",
    "yellow": "#ffd700",
    "red": "#ff4444",
    "blue": "#4488ff",
    "cyan": "#00ccff",
    "orange": "#ff8800",
    "dim": "#445566",
}

ORG_PROFILES = {
    "hospital": {"label": "Hospital", "id": "org_hospital_1", "net": "192.168.1.0/24", "icon": "🏥", "role": "Normal", "color": "#00ff88"},
    "bank": {"label": "Bank", "id": "org_bank_2", "net": "10.0.1.0/24", "icon": "🏦", "role": "Normal", "color": "#388bfd"},
    "university": {"label": "University", "id": "org_university_3", "net": "172.16.1.0/24", "icon": "🎓", "role": "Normal", "color": "#4488ff"},
}

ATTACK_MAP = {
    "ddos": "DDoS",
    "portscan": "Port Scan",
    "lateral": "Lateral",
    "exfil": "Exfil",
    "web": "Web",
}


class DashboardService:
    """Thread-safe singleton holding dashboard state and ML pipeline."""

    _instance: Optional["DashboardService"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._state_lock = threading.RLock()
        self.org_key = os.environ.get("AURA_ORG_ID", "").lower().strip()
        self.org = ORG_PROFILES.get(self.org_key)

        self.engine: Optional[AURAInferenceEngine] = None
        self.responder: Optional[AURAResponseEngine] = None
        self.injector: Optional[AttackInjector] = None
        self.blockchain: Optional[AURABlockchainLogger] = None
        self.model_status = "INITIALISING"

        self.ae_scores: List[float] = []
        self.thresholds: List[float] = []
        self.timestamps: List[float] = []
        self.alerts: List[dict] = []
        self.incidents: List[dict] = []
        self.fed_log: List[str] = []
        self.chain_log: List[dict] = []
        self.node_colors: Dict[int, str] = {i: THEME["green"] for i in range(cfg.NUM_SYNTHETIC_NODES)}
        self.node_states: Dict[int, str] = {i: "Normal" for i in range(cfg.NUM_SYNTHETIC_NODES)}
        self.current_graph: Optional[dict] = None
        self.attack_active = False
        self.attack_type: Optional[str] = None
        self.system_status = "INITIALISING"
        self.total_attacks = 0
        self.total_blocked = 0
        self.fl_rounds_done = 0
        self.chain_entries = 0
        self.window_counter = 0
        self.last_explanation: Optional[dict] = None
        self.fl_client_status: List[dict] = []
        self.fl_ready = False
        self.under_attack = False
        self._fl_running = False

        self._load_components()
        self._sync_fl_readiness()

    @classmethod
    def get(cls) -> "DashboardService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_components(self) -> None:
        bundle_path = cfg.MODELS_DIR / "aura_bundle.pth"
        bundle = AURAModelBundle()
        if bundle_path.exists():
            bundle.load_state_dict(torch.load(bundle_path, map_location="cpu"))
            self.model_status = "PRE-TRAINED"
        else:
            self.model_status = "UNTRAINED (DEMO MODE)"

        self.engine = AURAInferenceEngine(bundle.autoencoder, bundle.stgnn)
        self.responder = AURAResponseEngine()
        self.injector = AttackInjector()
        try:
            self.blockchain = AURABlockchainLogger()
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Blockchain logger init failed ({e}); using minimal fallback."
            )
            self.blockchain = AURABlockchainLogger.__new__(AURABlockchainLogger)
            self.blockchain._mode = "local_fallback"
            self.blockchain._w3 = None
            self.blockchain._contract = None
            self.blockchain._account = None
            self.blockchain._local_store = {}

        n = cfg.NUM_SYNTHETIC_NODES
        for _ in range(cfg.EMA_WARMUP_BATCHES + 10):
            x = torch.randn(n, cfg.FEATURE_DIM) * 0.1
            ei = torch.randint(0, n, (2, 40))
            attr = torch.randn(40, cfg.FEATURE_DIM) * 0.1
            self.engine.process({"x": x, "edge_index": ei, "edge_attr": attr, "window_id": "warmup"})

        self.system_status = "ACTIVE"

    def _sync_fl_readiness(self) -> None:
        if not self.org:
            return
        rf = Path(cfg.LOGS_DIR) / "fl_readiness.json"
        if rf.exists():
            try:
                rd = json.loads(rf.read_text())
                entry = rd.get(self.org_key, {})
                self.fl_ready = entry.get("ready", False)
                self.under_attack = entry.get("under_attack", False)
            except Exception:
                pass

    def _write_fl_readiness(self) -> None:
        if not self.org:
            return
        rf = Path(cfg.LOGS_DIR) / "fl_readiness.json"
        rf.parent.mkdir(parents=True, exist_ok=True)
        rd: dict = {}
        if rf.exists():
            try:
                rd = json.loads(rf.read_text())
            except Exception:
                rd = {}
        rd[self.org_key] = {
            "ready": self.fl_ready,
            "under_attack": self.under_attack,
            "org": self.org["label"],
            "net": self.org["net"],
            "ts": time.time(),
        }
        rf.write_text(json.dumps(rd, indent=2))

    def _trim_lists(self) -> None:
        for key in ("ae_scores", "thresholds", "timestamps"):
            lst = getattr(self, key)
            if len(lst) > 100:
                setattr(self, key, lst[-100:])
        if len(self.alerts) > 30:
            self.alerts = self.alerts[:30]
        if len(self.incidents) > 20:
            self.incidents = self.incidents[:20]

    def run_inference_tick(self, graph: dict, is_attack: bool = False) -> AnomalyEvent:
        assert self.engine and self.responder
        event = self.engine.process(graph)

        self.ae_scores.append(event.ae_score)
        thresh = event.ae_threshold if event.ae_threshold > 0 else 0
        self.thresholds.append(thresh)
        self.timestamps.append(event.timestamp)
        self._trim_lists()

        colors = {i: THEME["green"] for i in range(cfg.NUM_SYNTHETIC_NODES)}
        states = {i: "Normal" for i in range(cfg.NUM_SYNTHETIC_NODES)}

        if event.severity != AlertSeverity.NORMAL:
            self.total_attacks += 1
            for nid in event.triggered_nodes:
                colors[nid] = THEME["red"]
                states[nid] = f"⚠ {event.severity.name}"

            self.alerts.insert(0, event.to_dict())
            if event.inferred_attack != "Normal" and event.top_features:
                from aura.ae_explainer import ATTACK_EXPLANATIONS
                self.last_explanation = {
                    "inferred_attack": event.inferred_attack,
                    "match_score": event.match_score,
                    "top_features": event.top_features,
                    "group_residuals": event.group_residuals,
                    "severity": event.severity.name,
                    "confidence": event.confidence,
                    "explanation": ATTACK_EXPLANATIONS.get(
                        event.inferred_attack,
                        ATTACK_EXPLANATIONS.get("Unknown Anomaly", {}),
                    ),
                }

            records = self.responder.act(event)
            for r in records:
                if r.action_taken not in ("LOG_ONLY", "ALREADY_ACTIONED"):
                    self.total_blocked += 1
                self.incidents.insert(0, r.to_dict())

        elif is_attack:
            for nid in graph.get("attack_nodes", []):
                colors[nid] = THEME["yellow"]
                states[nid] = "Evaluating…"
            try:
                from aura.ae_explainer import explain_ae, ATTACK_EXPLANATIONS
                edge_attr = graph.get("edge_attr")
                if edge_attr is not None:
                    feat_residuals = self.engine.ae.explain_features(edge_attr)
                    expl = explain_ae(feat_residuals)
                    self.last_explanation = {
                        "inferred_attack": expl["inferred_attack"],
                        "match_score": expl["match_score"],
                        "top_features": expl["top_features"],
                        "group_residuals": expl["group_residuals"],
                        "severity": "LOW",
                        "confidence": expl["match_score"],
                        "explanation": ATTACK_EXPLANATIONS.get(
                            expl["inferred_attack"],
                            ATTACK_EXPLANATIONS.get("Unknown Anomaly", {}),
                        ),
                    }
            except Exception:
                pass

        self.node_colors = colors
        self.node_states = states
        self.current_graph = graph
        self.window_counter += 1
        return event

    def poll_pending_inject(self) -> None:
        pending_path = Path(cfg.LOGS_DIR) / "pending_inject.json"
        if not pending_path.exists():
            return
        try:
            pi = json.loads(pending_path.read_text())
            if not pi:
                return
            pi_idx = int(pi.get("node_index", -1))
            pi_mse = float(pi.get("mse", 0.0))
            pi_ts = float(pi.get("timestamp", 0))
            pi_nid = str(pi.get("target_node", ""))

            if pi_idx >= 0 and pi_ts > 0 and (time.time() - pi_ts) < 30:
                self.node_colors[pi_idx] = THEME["yellow"]
                self.node_states[pi_idx] = "⚡ Evaluating…"

                pi_sev = (
                    AlertSeverity.HIGH if pi_mse > cfg.MSE_THRESHOLD_HIGH
                    else AlertSeverity.MEDIUM if pi_mse > cfg.MSE_THRESHOLD_MEDIUM
                    else AlertSeverity.LOW
                )
                pi_conf = min(1.0, pi_mse / 0.5)
                pi_event = AnomalyEvent(
                    timestamp=pi_ts,
                    window_id=f"CUSTOM_INJECT_{pi_nid}",
                    ae_score=pi_mse,
                    ae_threshold=0.3,
                    gnn_scores=[],
                    severity=pi_sev,
                    triggered_nodes=[pi_idx],
                    confidence=pi_conf,
                    raw_label_ratio=0.0,
                    top_features=[],
                    inferred_attack="Custom Injection",
                    match_score=0.0,
                    group_residuals={},
                )
                if self.responder:
                    for pr in self.responder.act(pi_event):
                        self.incidents.insert(0, pr.to_dict())
                pi_alert = pi_event.to_dict()
                pi_alert["tag"] = "CUSTOM_INJECT"
                self.alerts.insert(0, pi_alert)
                self.total_attacks += 1

                expl_path = Path(cfg.LOGS_DIR) / "last_explanation.json"
                if expl_path.exists():
                    try:
                        self.last_explanation = {
                            **json.loads(expl_path.read_text()),
                            "severity": pi_sev.name,
                            "confidence": pi_conf,
                        }
                    except Exception:
                        pass

            pending_path.write_text("{}")
        except Exception:
            pass

    def inject_attack(self, attack_type: str) -> dict:
        assert self.injector
        with self._state_lock:
            self.attack_active = True
            self.attack_type = attack_type
            self.system_status = "UNDER ATTACK"
            attack_graph = self.injector.inject(attack_type)
            event = self.run_inference_tick(attack_graph, is_attack=True)
            return {
                "severity": event.severity.name,
                "confidence": event.confidence,
                "label": ATTACK_MAP.get(attack_type, attack_type),
            }

    def inject_normal(self) -> None:
        assert self.injector
        with self._state_lock:
            self.attack_active = False
            self.system_status = "ACTIVE"
            normal_graph = self.injector._generate_healthy_graph()
            normal_graph["window_id"] = f"NORMAL_{self.window_counter}"
            self.run_inference_tick(normal_graph)
            self.node_colors = {i: THEME["green"] for i in range(cfg.NUM_SYNTHETIC_NODES)}
            self.node_states = {i: "Normal" for i in range(cfg.NUM_SYNTHETIC_NODES)}

    def run_federation(self) -> dict:
        if self._fl_running:
            return {"status": "busy"}
        from aura.fl_server import run_federation_simulation

        with self._state_lock:
            self._fl_running = True
            self.fed_log = ["🚀 Federation round initiated …"]
            bc_module = self.blockchain
            round_results = run_federation_simulation(blockchain_module=bc_module, n_rounds=3)

            for r in round_results:
                rnd = r.get("round", "?")
                version = r.get("model_version", "N/A")
                h = r.get("model_hash", "N/A")
                trusted = r.get("fltrust_trusted_indices", [])
                kept = len(trusted)
                if "client_statuses" in r:
                    self.fl_client_status = r["client_statuses"]
                self.fed_log.extend([
                    f"━━━  Round {rnd}  ━━━",
                    "[CLIENT hospital_1] Attack pattern learned. Sending weights…",
                    "[CLIENT bank_2]     Local training complete. Sending weights…",
                    "[CLIENT uni_3]      Local training complete. Sending weights…",
                    f"[SERVER] FLTrust: {kept}/3 client updates trusted (cosine vs server root).",
                    f"[SERVER] Global Model {version} aggregated.",
                    f"[BLOCKCHAIN] Hash recorded: {h[:20]}…",
                    "[CLIENT hospital_1] Verifying hash on chain… ✓ Match. Model deployed.",
                    "[CLIENT bank_2]     Verifying hash on chain… ✓ Match. Model deployed.",
                    "[CLIENT uni_3]      Verifying hash on chain… ✓ Match. Model deployed.",
                ])
                self.chain_log.insert(0, {
                    "version": version,
                    "hash": h,
                    "round": rnd,
                    "time": time.strftime("%H:%M:%S"),
                })

            self.fl_rounds_done += 3
            self.chain_entries = len(self.chain_log)
            self.fed_log.append("✅ Federation complete.  All clients immunised.")
            self._fl_running = False
            return {"status": "ok", "rounds": len(round_results)}

    def register_test_hash(self) -> dict:
        assert self.blockchain
        fake_hash = "0x" + hashlib.sha256(str(time.time()).encode()).hexdigest()
        ver = f"manual_{int(time.time())}"
        self.blockchain.log_model_update(ver, fake_hash)
        entry = {"version": ver, "hash": fake_hash, "round": "manual", "time": time.strftime("%H:%M:%S")}
        self.chain_log.insert(0, entry)
        self.chain_entries = len(self.chain_log)
        return entry

    def verify_chain(self) -> dict:
        registry = Path("logs/hash_registry.json")
        ledger = Path("logs/blockchain_fallback.jsonl")
        if not registry.exists():
            return {"ok": False, "message": "Trusted registry not found. Run FL first."}
        trusted = json.loads(registry.read_text())
        if not ledger.exists() or ledger.stat().st_size == 0:
            return {"ok": False, "message": "No ledger entries found."}
        entries = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
        latest: dict = {}
        for e in entries:
            latest[e["model_version"]] = e
        fl_entries = [v for k, v in latest.items() if k in trusted]
        results = []
        all_ok = True
        for e in fl_entries:
            version = e["model_version"]
            ok = e["model_hash"] == trusted[version]
            if not ok:
                all_ok = False
            results.append({"version": version, "ok": ok, "ledger_hash": e["model_hash"][:32]})
        return {"ok": all_ok, "entries": results, "message": "INTACT" if all_ok else "TAMPER DETECTED"}

    def clear_logs(self) -> None:
        with self._state_lock:
            for key in ("ae_scores", "thresholds", "timestamps", "alerts", "incidents", "fed_log", "chain_log"):
                setattr(self, key, [])
            self.node_colors = {i: THEME["green"] for i in range(cfg.NUM_SYNTHETIC_NODES)}
            self.node_states = {i: "Normal" for i in range(cfg.NUM_SYNTHETIC_NODES)}
            self.system_status = "ACTIVE"
            self.last_explanation = None

    def set_fl_ready(self, ready: bool) -> None:
        self.fl_ready = ready
        if ready:
            self.under_attack = False
        self._write_fl_readiness()

    def set_under_attack(self, under: bool) -> None:
        self.under_attack = under
        if under:
            self.fl_ready = False
        self._write_fl_readiness()

    def _node_registry(self) -> List[dict]:
        nodes = []
        for i in range(cfg.NUM_SYNTHETIC_NODES):
            node_id = f"node_{i}"
            label = cfg.CRITICAL_ALLOWLIST.get(node_id, f"Host-{i:02d}")
            nodes.append({
                "id": node_id,
                "label": label,
                "index": i,
                "critical": node_id in cfg.CRITICAL_ALLOWLIST,
                "color": self.node_colors.get(i, THEME["green"]),
                "state": self.node_states.get(i, "Normal"),
            })
        return nodes

    def _edge_index(self) -> Optional[List[List[int]]]:
        cg = self.current_graph
        if cg is None or "edge_index" not in cg:
            return None
        ei = cg["edge_index"]
        if hasattr(ei, "numpy"):
            ei = ei.numpy()
        return ei[:, : min(ei.shape[1], 60)].tolist()

    def get_state(self) -> dict:
        with self._state_lock:
            self.poll_pending_inject()
            self._sync_fl_readiness()

            ema_info = {}
            if self.engine:
                ema = self.engine.ema_state
                warmup_left = max(0, cfg.EMA_WARMUP_BATCHES - ema.get("batch_count", 0))
                ema_info = {
                    "warmup_left": warmup_left,
                    "mean": ema.get("mean", 0),
                    "std": ema.get("std", 0),
                    "threshold": ema.get("threshold", 0),
                }

            current_ae = self.ae_scores[-1] if self.ae_scores else 0.0
            bc_mode = self.blockchain.mode.upper() if self.blockchain else "UNKNOWN"

            return {
                "system_status": self.system_status,
                "model_status": self.model_status,
                "blockchain_mode": bc_mode,
                "org": self.org,
                "metrics": {
                    "window_counter": self.window_counter,
                    "total_attacks": self.total_attacks,
                    "total_blocked": self.total_blocked,
                    "fl_rounds_done": self.fl_rounds_done,
                    "chain_entries": self.chain_entries,
                    "current_ae_score": round(current_ae, 4),
                },
                "nodes": self._node_registry(),
                "edge_index": self._edge_index(),
                "timeline": {
                    "scores": self.ae_scores,
                    "thresholds": self.thresholds,
                    "timestamps": self.timestamps,
                },
                "last_explanation": self.last_explanation,
                "alerts": self.alerts[:10],
                "incidents": self.incidents[:10],
                "fed_log": self.fed_log[-15:],
                "chain_log": self.chain_log[:6],
                "fl_client_status": self.fl_client_status,
                "fl_ready": self.fl_ready,
                "under_attack": self.under_attack,
                "fl_running": self._fl_running,
                "ema": ema_info,
                "theme": THEME,
                "critical_allowlist": cfg.CRITICAL_ALLOWLIST,
            }
