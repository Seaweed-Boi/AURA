"""
policy_engine.py — AURA Operator-Configurable Script Execution Engine
======================================================================

Replaces hardcoded iptables/tc subprocess calls with a YAML-driven,
hot-swappable policy table. Operators can change response behaviour by
editing response_policy.yaml without touching any Python code.

Decision Tree (per spec)
------------------------
  LOW    → Log autonomously.  No human involvement.
  MEDIUM → Throttle autonomously.  HITL notification sent (inform only).
  HIGH   → HITL approval gate presented.
             y   → Isolate (scripts/isolate.sh)
             else→ Throttle as fallback (scripts/throttle.sh)
                   Log the rejection + fallback action with timestamp.

A node ALWAYS exits the response engine in a controlled state.
HITL rejection of the highest tier is not permission to do nothing —
it is permission to execute the next tier down.

HITL Gate (Upgrade 1 requirement)
----------------------------------
  Only fires when the matched script path contains 'isolate'.
  Prints a prompt to stdout.  On 'y': runs isolate.sh.
  On any other input: runs throttle.sh and logs the fallback.
"""

import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_BASE_DIR    = Path(__file__).parent.resolve()
_POLICY_FILE = _BASE_DIR / "response_policy.yaml"
_FALLBACK_SCRIPT = str(_BASE_DIR / "scripts" / "log_only.sh")
_THROTTLE_SCRIPT = str(_BASE_DIR / "scripts" / "throttle.sh")

IS_WINDOWS = platform.system() == "Windows"


# ─────────────────────────────────────────────────────────────────────────────
# Load Policy at Import Time
# ─────────────────────────────────────────────────────────────────────────────

def _load_policy(path: Path) -> list:
    """
    Parse response_policy.yaml into the RULES list.
    Returns an empty list on failure (causes all events to fall back to log_only).
    """
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        rules = data.get("rules", [])
        logger.info(f"[POLICY] Loaded {len(rules)} rules from {path}")
        return rules
    except FileNotFoundError:
        logger.error(f"[POLICY] Policy file not found: {path}. All actions default to LOG_ONLY.")
        return []
    except yaml.YAMLError as e:
        logger.error(f"[POLICY] YAML parse error in {path}: {e}. All actions default to LOG_ONLY.")
        return []


RULES: list = _load_policy(_POLICY_FILE)


# ─────────────────────────────────────────────────────────────────────────────
# Rule Matching
# ─────────────────────────────────────────────────────────────────────────────

def _match_rule(severity: str, asset_class: str) -> Optional[str]:
    """
    Find the first rule where severity matches AND asset_class matches
    (or asset_class is 'ANY').  Returns the script path string or None.
    """
    sev_upper   = severity.upper()
    asset_upper = asset_class.upper()

    for rule in RULES:
        rule_sev   = str(rule.get("severity", "")).upper()
        rule_asset = str(rule.get("asset_class", "")).upper()
        action     = rule.get("action", "")

        if rule_sev == sev_upper and (rule_asset == asset_upper or rule_asset == "ANY"):
            return str(_BASE_DIR / action)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# HITL Gate
# ─────────────────────────────────────────────────────────────────────────────

def _hitl_gate(node_id: str, node_label: str, confidence: float) -> bool:
    """
    Present an operator confirmation prompt for isolation actions.

    Returns True  → proceed with isolate.sh
    Returns False → degrade to throttle.sh (logged)

    This is non-blocking on the happy path (y).  On any other input,
    the system degrades gracefully — a node is never left uncontrolled.
    """
    print("\n" + "=" * 62)
    print("  [!]  AURA - HUMAN-IN-THE-LOOP ISOLATION APPROVAL REQUIRED")
    print("=" * 62)
    print(f"  Node     : {node_id} ({node_label})")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Action   : FULL NETWORK ISOLATION (iptables DROP)")
    print("=" * 62)
    try:
        answer = input("  Approve isolation? [y / anything else = throttle]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        # Non-interactive environment (e.g. dashboard simulation) → degrade
        answer = ""

    if answer == "y":
        logger.info(f"[HITL] Isolation APPROVED by operator for {node_id}")
        return True
    else:
        logger.warning(
            f"[HITL] Isolation REJECTED at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"for {node_id}. Fallback: THROTTLE applied automatically."
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Script Execution
# ─────────────────────────────────────────────────────────────────────────────

def _run_script(
    script_path: str,
    node_id:     str,
    simulated_ip: str,
    severity:    str,
    confidence:  float,
) -> str:
    """
    Execute the given shell script, injecting context via environment variables.
    Returns the command string for audit logging.

    On Windows, scripts are not executed — the command is logged as SIMULATED.
    """
    env = {
        **os.environ,
        "NODE_ID":      node_id,
        "SIMULATED_IP": simulated_ip,
        "SEVERITY":     severity,
        "CONFIDENCE":   f"{confidence:.4f}",
    }

    # Normalise path separators
    script_path = str(Path(script_path))
    command_str = f"[SCRIPT] {script_path} NODE_ID={node_id} IP={simulated_ip}"

    if IS_WINDOWS:
        logger.info(f"[POLICY-SIM] {command_str}")
        print(f"[POLICY ENGINE — WINDOWS SIMULATION] Would run: {script_path}")
        return f"[SIMULATED] {command_str}"

    if not Path(script_path).exists():
        logger.error(f"[POLICY] Script not found: {script_path}. Action skipped.")
        return f"[SCRIPT-MISSING] {script_path}"

    try:
        result = subprocess.run(
            ["bash", script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        if result.stdout:
            logger.info(f"[SCRIPT-OUT] {result.stdout.strip()}")
        return command_str
    except subprocess.CalledProcessError as e:
        logger.error(f"[POLICY] Script failed (exit {e.returncode}): {e.stderr.strip()}")
        return f"[SCRIPT-ERROR] {command_str}"
    except subprocess.TimeoutExpired:
        logger.error(f"[POLICY] Script timed out: {script_path}")
        return f"[SCRIPT-TIMEOUT] {command_str}"
    except Exception as exc:
        logger.error(f"[POLICY] Unexpected error running {script_path}: {exc}")
        return f"[SCRIPT-EXCEPTION] {command_str}"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def execute_response(
    severity:     str,
    asset_class:  str,
    node_id:      str,
    node_label:   str,
    simulated_ip: str,
    confidence:   float,
) -> str:
    """
    Main entry point called by response_engine.py.

    Matches severity + asset_class against YAML rules, applies HITL gate
    for isolation actions, executes the appropriate script.

    Parameters
    ----------
    severity     : "HIGH" | "MEDIUM" | "LOW"
    asset_class  : "CRITICAL" | "STANDARD"
    node_id      : e.g. "node_5"
    node_label   : human-readable label from allowlist or "Standard Asset"
    simulated_ip : derived IP string (e.g. "10.0.0.5")
    confidence   : fused anomaly confidence ∈ [0, 1]

    Returns
    -------
    Command string for audit logging in IncidentRecord.
    """
    script_path = _match_rule(severity, asset_class)

    if script_path is None:
        logger.warning(
            f"[POLICY] No rule matched for severity={severity} "
            f"asset_class={asset_class}. Defaulting to LOG_ONLY."
        )
        script_path = _FALLBACK_SCRIPT

    # ── HITL Gate (only for isolate actions) ─────────────────────────────────
    if "isolate" in Path(script_path).name.lower():
        approved = _hitl_gate(node_id, node_label, confidence)
        if not approved:
            # Degrade to throttle — node must exit in a controlled state
            print(f"[POLICY ENGINE] HITL rejected isolation. Auto-applying THROTTLE on {node_id}.")
            script_path = _THROTTLE_SCRIPT

    return _run_script(script_path, node_id, simulated_ip, severity, confidence)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=== policy_engine.py — Sanity Check ===\n")
    print(f"Loaded {len(RULES)} rules from response_policy.yaml")
    for i, r in enumerate(RULES):
        print(f"  Rule {i+1}: severity={r['severity']} asset_class={r['asset_class']} → {r['action']}")

    print("\n--- Test: LOW / ANY ---")
    cmd = execute_response("LOW", "STANDARD", "node_7", "Standard Asset", "10.0.0.7", 0.25)
    print(f"  Returned: {cmd}")

    print("\n--- Test: MEDIUM / ANY ---")
    cmd = execute_response("MEDIUM", "STANDARD", "node_9", "Standard Asset", "10.0.0.9", 0.65)
    print(f"  Returned: {cmd}")

    print("\n--- Test: HIGH / CRITICAL (should throttle, no HITL gate) ---")
    cmd = execute_response("HIGH", "CRITICAL", "node_0", "Domain Controller (AD)", "10.0.0.1", 0.91)
    print(f"  Returned: {cmd}")

    print("\n--- Test: HIGH / STANDARD (HITL gate will prompt) ---")
    cmd = execute_response("HIGH", "STANDARD", "node_12", "Standard Asset", "10.0.0.12", 0.88)
    print(f"  Returned: {cmd}")

    print("\n✓ policy_engine sanity check complete.")
