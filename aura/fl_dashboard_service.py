"""
aura/fl_dashboard_service.py — Backend for FL Server Console (React).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

THEME = {
    "bg": "#0d1117",
    "panel": "#161b22",
    "panel2": "#1c2430",
    "border": "#30363d",
    "cyan": "#58d1e8",
    "green": "#3fb950",
    "red": "#f85149",
    "orange": "#d29922",
    "blue": "#388bfd",
    "purple": "#bc8cff",
    "yellow": "#e3b341",
    "dim": "#8b949e",
    "text": "#c9d1d9",
}

ORGS = [
    {"id": "org_hospital_1", "key": "hospital", "label": "Hospital", "net": "192.168.1.0/24", "icon": "🏥"},
    {"id": "org_bank_2", "key": "bank", "label": "Bank", "net": "10.0.1.0/24", "icon": "🏦"},
    {"id": "org_university_3", "key": "university", "label": "University", "net": "172.16.1.0/24", "icon": "🎓"},
]

PIPE_STEPS = [
    ("📥", "Collect Weights"),
    ("🔬", "FLTrust Filter"),
    ("🧮", "Aggregate"),
    ("⛓", "Mint Hash"),
    ("📡", "Broadcast + Verify"),
]

_READINESS_FILE = Path(cfg.LOGS_DIR) / "fl_readiness.json"


class FLDashboardService:
    _instance: Optional["FLDashboardService"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._state_lock = threading.RLock()
        self.fl_running = False
        self.fl_done = False
        self.round_results: List[dict] = []
        self.hash_ledger: List[dict] = []
        self.client_cards: Dict[str, dict] = {
            o["id"]: {"status": "idle", "round": 0, "selected": None, "verified": None}
            for o in ORGS
        }
        self.pipe_state = [0] * len(PIPE_STEPS)
        self.current_round = 0
        self.total_rounds = cfg.FL_NUM_ROUNDS
        self.fl_log: List[str] = []
        self.fltrust_scores_hist: List[dict] = []
        self.global_hash: Optional[str] = None
        self.global_version: Optional[str] = None
        self.verify_results: Dict[str, bool] = {}
        self.byzantine_org: Optional[str] = None
        self.quarantined_orgs: List[str] = []

    @classmethod
    def get(cls) -> "FLDashboardService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _read_readiness(self) -> dict:
        if not _READINESS_FILE.exists():
            return {}
        try:
            return json.loads(_READINESS_FILE.read_text())
        except Exception:
            return {}

    def run_simulation(self) -> None:
        if self.fl_running:
            return

        def _worker():
            from aura.blockchain import AURABlockchainLogger
            from aura.fl_server import run_federation_simulation

            with self._state_lock:
                self.fl_running = True
                self.fl_done = False
                self.fl_log = ["🚀 FL Server: starting networked simulation…"]
                self.pipe_state = [1] + [0] * (len(PIPE_STEPS) - 1)
                self.current_round = 0
                self.round_results = []

            try:
                bc = AURABlockchainLogger()
                results = run_federation_simulation(blockchain_module=bc, n_rounds=self.total_rounds)

                for step_idx in range(len(PIPE_STEPS)):
                    with self._state_lock:
                        self.pipe_state = [2 if i < step_idx else (1 if i == step_idx else 0) for i in range(len(PIPE_STEPS))]

                for r in results:
                    rnd = r.get("round", 0)
                    with self._state_lock:
                        self.current_round = rnd
                        self.round_results.append(r)
                        version = r.get("model_version", "N/A")
                        h = r.get("model_hash", "N/A")
                        self.global_hash = h
                        self.global_version = version
                        trusted = r.get("fltrust_trusted_indices", [])
                        scores = r.get("trust_scores", {})
                        self.fltrust_scores_hist.append({"round": rnd, "scores": scores})
                        self.fl_log.append(f"Round {rnd}: FLTrust trusted {len(trusted)}/3 clients")
                        self.fl_log.append(f"Hash minted: {h[:24]}…")
                        self.hash_ledger.insert(0, {
                            "version": version,
                            "hash": h,
                            "round": rnd,
                            "time": time.strftime("%H:%M:%S"),
                        })
                        if "client_statuses" in r:
                            for cs in r["client_statuses"]:
                                oid = cs.get("org_id", "")
                                if oid in self.client_cards:
                                    self.client_cards[oid]["status"] = cs.get("status", "idle")
                                    self.client_cards[oid]["round"] = rnd
                                    self.client_cards[oid]["selected"] = cs.get("selected")
                                    self.client_cards[oid]["verified"] = cs.get("verified")

                with self._state_lock:
                    self.pipe_state = [2] * len(PIPE_STEPS)
                    self.fl_log.append("✅ Federation complete — all clients verified.")
                    self.fl_done = True
            except Exception as e:
                with self._state_lock:
                    self.fl_log.append(f"❌ FL error: {e}")
            finally:
                with self._state_lock:
                    self.fl_running = False

        threading.Thread(target=_worker, daemon=True).start()

    def get_state(self) -> dict:
        readiness = self._read_readiness()
        org_readiness = []
        for o in ORGS:
            info = readiness.get(o["key"], {})
            org_readiness.append({
                **o,
                "ready": info.get("ready", False),
                "under_attack": info.get("under_attack", False),
                "net_live": info.get("net", o["net"]),
            })

        run_state = "RUNNING" if self.fl_running else ("COMPLETE" if self.fl_done else "IDLE")

        with self._state_lock:
            return {
                "theme": THEME,
                "run_state": run_state,
                "fl_running": self.fl_running,
                "fl_done": self.fl_done,
                "current_round": self.current_round,
                "total_rounds": self.total_rounds,
                "pipe_steps": [{"icon": s[0], "label": s[1], "state": self.pipe_state[i]} for i, s in enumerate(PIPE_STEPS)],
                "orgs": org_readiness,
                "client_cards": self.client_cards,
                "round_results": self.round_results,
                "hash_ledger": self.hash_ledger[:8],
                "fl_log": self.fl_log[-24:],
                "fltrust_scores_hist": self.fltrust_scores_hist,
                "global_hash": self.global_hash,
                "global_version": self.global_version,
                "byzantine_org": self.byzantine_org,
                "quarantined_orgs": self.quarantined_orgs,
            }
