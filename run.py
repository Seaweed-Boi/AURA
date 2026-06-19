"""
run.py — AURA Quick-Start Launcher
===================================
Convenience script for the hackathon demo.  Handles all startup sequencing.

Commands:
  python run.py train       # Train models on CICIDS2017 data
  python run.py train --quick  # 5-epoch quick sanity check
  python run.py dashboard   # Launch Streamlit dashboard
  python run.py demo        # Full pipeline demo without dashboard
  python run.py test        # Sanity-test all modules
  python run.py federation  # Run FL simulation standalone
"""

import sys
import subprocess
from pathlib import Path

BASE = Path(__file__).parent
PYTHON = sys.executable


def cmd_train(quick=False):
    args = [PYTHON, str(BASE / "train.py")]
    if quick or "--quick" in sys.argv:
        args.append("--quick")
    subprocess.run(args)


def cmd_dashboard():
    """Launch React+Vite UI with Flask API backend (replaces Streamlit)."""
    import webbrowser
    import time as _time

    frontend = BASE / "frontend"
    if not (frontend / "package.json").exists():
        print("Frontend not found. Expected:", frontend)
        sys.exit(1)

    # Free port 5001 if a stale API server is still running (Windows)
    if sys.platform == "win32":
        subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-NetTCPConnection -LocalPort 5001 -ErrorAction SilentlyContinue | "
                "ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }",
            ],
            capture_output=True,
        )

    print("\n[!] Launching AURA Dashboard (React + Vite) ...")
    print("   API:  http://localhost:5001")
    print("   UI:   http://localhost:5173\n")

    api_proc = subprocess.Popen(
        [PYTHON, str(BASE / "api_server.py")],
        cwd=str(BASE),
    )

    # Install deps on first run if node_modules missing
    if not (frontend / "node_modules").exists():
        print("Installing frontend dependencies (first run)…")
        subprocess.run(["npm", "install"], cwd=str(frontend), shell=True, check=True)

    ui_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(frontend),
        shell=True,
    )

    _time.sleep(2)
    webbrowser.open("http://localhost:5173")

    try:
        ui_proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        api_proc.terminate()
        ui_proc.terminate()
        print("\nDashboard stopped.\n")


def cmd_test():
    """Quick sanity test of all modules."""
    print("\n=== AURA Full System Sanity Test ===\n")
    errors = []

    tests = [
        ("Data Loader",      BASE / "aura" / "data_loader.py"),
        ("Models",           BASE / "aura" / "models.py"),
        ("Detector",         BASE / "aura" / "detector.py"),
        ("Response Engine",  BASE / "aura" / "response_engine.py"),
        ("Attack Injector",  BASE / "aura" / "attack_injector.py"),
        ("Blockchain",       BASE / "aura" / "blockchain.py"),
        ("FL Client",        BASE / "aura" / "fl_client.py"),
    ]

    for name, script in tests:
        args = [PYTHON, str(script)]
        if name == "FL Client":
            args.append("--help")
        result = subprocess.run(args, capture_output=True, text=True)
        status = "[PASS]" if result.returncode == 0 else "[FAIL]"
        color  = "\033[92m" if result.returncode == 0 else "\033[91m"
        print(f"  {color}{status}\033[0m  {name}")
        if result.returncode != 0:
            errors.append((name, result.stderr[-300:]))

    if errors:
        print("\nFailed modules:")
        for name, err in errors:
            print(f"\n[{name}]\n{err}")
        sys.exit(1)
    else:
        print("\n✓ All modules passed. AURA is ready for demo.\n")


def cmd_demo():
    """Run a full pipeline demo in-process."""
    print("\n=== AURA Full Pipeline Demo ===\n")
    import torch
    from aura.models import FlowAutoencoder, AuraSTGNN
    from aura.detector import AURAInferenceEngine, AlertSeverity
    from aura.response_engine import AURAResponseEngine
    from aura.attack_injector import AttackInjector
    from aura.blockchain import AURABlockchainLogger
    from aura.fl_server import run_federation_simulation
    import config as cfg

    ae       = FlowAutoencoder()
    gnn      = AuraSTGNN()
    engine   = AURAInferenceEngine(ae, gnn)
    responder = AURAResponseEngine()
    injector  = AttackInjector()
    bc        = AURABlockchainLogger()

    print("Phase 1: Processing 60 normal traffic windows (EMA warmup) …")
    for i in range(60):
        normal = injector._generate_healthy_graph()
        normal["window_id"] = f"normal:w{i}"
        engine.process(normal)

    print("\nPhase 2: Injecting DDoS attack …")
    attack = injector.inject("ddos")
    event  = engine.process(attack)
    print(f"  → Severity: {event.severity.name}  Confidence: {event.confidence:.1%}")
    records = responder.act(event)
    for r in records:
        print(f"  → Response: {r.action_taken}  Node: {r.node_id} ({r.node_label})")

    print("\nPhase 3: Federated Learning simulation …")
    run_federation_simulation(blockchain_module=bc, n_rounds=2)

    print("\nPhase 4: Blockchain verification …")
    import hashlib
    ver  = "demo_v1.0"
    h    = "0x" + hashlib.sha256(b"test_model_weights").hexdigest()
    bc.log_model_update(ver, h)
    ok, src = bc.verify_model(ver, h)
    print(f"  → Hash verified: {ok}  (source: {src})")

    print("\n✓ Full demo pipeline complete!\n")


def cmd_federation():
    from aura.fl_server import run_federation_simulation
    run_federation_simulation(n_rounds=3)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "train":
        cmd_train()
    elif cmd == "dashboard":
        cmd_dashboard()
    elif cmd == "test":
        cmd_test()
    elif cmd == "demo":
        cmd_demo()
    elif cmd == "federation":
        cmd_federation()
    else:
        print(__doc__)
