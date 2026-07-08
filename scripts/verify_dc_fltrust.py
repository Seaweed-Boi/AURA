import sys
from pathlib import Path
import torch
import torch.optim as optim
import copy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aura.models import FlowAutoencoder, AttackHead
from scripts.benchmark_byzantine import _run_local_training_dual
from aura.fl_server import dc_fltrust_aggregate

def run_verify():
    print("=== DC-FLTrust Synthetic Verification ===")
    
    # Global models
    global_ae = FlowAutoencoder()
    global_head = AttackHead()
    
    # Create clients: 0=honest, 1=under attack, 2=byzantine
    client_aes = [copy.deepcopy(global_ae) for _ in range(3)]
    client_heads = [copy.deepcopy(global_head) for _ in range(3)]
    client_ae_opts = [optim.Adam(ae.parameters(), lr=1e-3) for ae in client_aes]
    client_head_opts = [optim.Adam(head.parameters(), lr=1e-3) for head in client_heads]
    
    # Root models (server)
    root_ae = copy.deepcopy(global_ae)
    root_head = copy.deepcopy(global_head)
    root_ae_opt = optim.Adam(root_ae.parameters(), lr=1e-3)
    root_head_opt = optim.Adam(root_head.parameters(), lr=1e-3)
    
    # Synthetic data generator
    def get_flows(client_id):
        # honest
        if client_id == 0:
            return torch.randn(200, 47) * 0.1
        # under attack
        elif client_id == 1:
            benign = torch.randn(100, 47) * 0.1
            attacks = torch.randn(100, 47) * 5.0
            return torch.cat([benign, attacks])
        # byzantine (data doesn't matter, we'll submit random gradients anyway)
        else:
            return torch.randn(200, 47) * 0.1

    def run_pass(model_ae, model_head, opt_ae, opt_head, flows, g_ae_weights, g_head_weights):
        return _run_local_training_dual(
            ae=model_ae,
            attack_head=model_head,
            all_flows=flows,
            ae_optimizer=opt_ae,
            head_optimizer=opt_head,
            global_ae_weights=g_ae_weights,
            global_head_weights=g_head_weights,
            mse_threshold_high=0.5,
            head_epochs=3
        )

    # 5 rounds
    for round_idx in range(1, 6):
        print(f"\n--- Round {round_idx} ---")
        g_ae_w = {k: v.clone() for k, v in global_ae.state_dict().items()}
        g_head_w = {k: v.clone() for k, v in global_head.state_dict().items()}
        
        # 1. Sync clients to global weights
        for i in range(3):
            client_aes[i].load_state_dict(g_ae_w)
            client_heads[i].load_state_dict(g_head_w)
            
        root_ae.load_state_dict(g_ae_w)
        root_head.load_state_dict(g_head_w)
        
        # 2. Server root training
        root_flows = get_flows(1) # root trains on representative mix
        r_ae_delta, r_head_delta, _, _, _ = run_pass(
            root_ae, root_head, root_ae_opt, root_head_opt, root_flows, g_ae_w, g_head_w
        )
        
        # 3. Client training
        c_ae_deltas = []
        c_head_deltas = []
        
        for i in range(3):
            if i == 2:
                # Byzantine client
                ae_delta = {k: torch.randn_like(v) for k, v in g_ae_w.items()}
                head_delta = {k: torch.randn_like(v) for k, v in g_head_w.items()}
                c_ae_deltas.append(ae_delta)
                c_head_deltas.append(head_delta)
            else:
                flows = get_flows(i)
                ae_d, head_d, _, _, _ = run_pass(
                    client_aes[i], client_heads[i], client_ae_opts[i], client_head_opts[i],
                    flows, g_ae_w, g_head_w
                )
                c_ae_deltas.append(ae_d)
                c_head_deltas.append(head_d)
                
        # 4. Aggregation
        # Set warmup rounds to 0 so we see ch2 immediately
        client_round_counts = [round_idx] * 3
        agg_ae, agg_head, ch1, ch2, classes = dc_fltrust_aggregate(
            c_ae_deltas, c_head_deltas, r_ae_delta, r_head_delta, client_round_counts, ch2_warmup_rounds=0
        )
        
        for i in range(3):
            print(f"Client {i}: ch1={ch1[i]:.4f} ch2={ch2[i]:.4f} -> {classes[i]}")

if __name__ == "__main__":
    run_verify()
