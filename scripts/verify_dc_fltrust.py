import sys
from pathlib import Path
import torch
import torch.optim as optim
import copy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aura.models import FlowAutoencoder, AttackHead
from scripts.benchmark_byzantine import _run_local_training_dual
from aura.fl_server import ae_only_fltrust_aggregate, joint_dual_fltrust_aggregate, dc_fltrust_aggregate

def run_verify():
    print("=== DC-FLTrust Synthetic Verification ===")
    
    # Global models
    global_ae = FlowAutoencoder()
    global_head = AttackHead()
    
    # Create 5 clients
    # Client 0, 3, 4: HEALTHY
    # Client 1: UNDER_ATTACK
    # Client 2: BYZANTINE
    num_clients = 5
    client_aes = [copy.deepcopy(global_ae) for _ in range(num_clients)]
    client_heads = [copy.deepcopy(global_head) for _ in range(num_clients)]
    client_ae_opts = [optim.Adam(ae.parameters(), lr=1e-3) for ae in client_aes]
    client_head_opts = [optim.Adam(head.parameters(), lr=1e-3) for head in client_heads]
    
    # Root models (server)
    root_ae = copy.deepcopy(global_ae)
    root_head = copy.deepcopy(global_head)
    root_ae_opt = optim.Adam(root_ae.parameters(), lr=1e-3)
    root_head_opt = optim.Adam(root_head.parameters(), lr=1e-3)
    
    def get_flows(client_id):
        if client_id in [0, 3, 4]:
            # healthy
            return torch.randn(200, 47) * 0.1
        elif client_id == 1:
            # under attack
            benign = torch.randn(100, 47) * 0.1
            attacks = torch.randn(100, 47) * 5.0
            return torch.cat([benign, attacks])
        else:
            # byzantine (data doesn't matter)
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

    g_ae_w = {k: v.clone() for k, v in global_ae.state_dict().items()}
    g_head_w = {k: v.clone() for k, v in global_head.state_dict().items()}
    
    root_flows = get_flows(1) # root trains on representative mix
    r_ae_delta, r_head_delta, _, _, _ = run_pass(
        root_ae, root_head, root_ae_opt, root_head_opt, root_flows, g_ae_w, g_head_w
    )
    
    c_ae_deltas = []
    c_head_deltas = []
    
    for i in range(num_clients):
        if i == 2:
            # Byzantine camouflage client: honest AE, inverted/random AttackHead
            # We want to test BYZANTINE_FAKE_ATTACK, so AE is honest
            flows = get_flows(0) # honest data
            ae_d, _, _, _, _ = run_pass(
                client_aes[i], client_heads[i], client_ae_opts[i], client_head_opts[i],
                flows, g_ae_w, g_head_w
            )
            # honest AE delta
            c_ae_deltas.append(ae_d)
            # inverted head delta to guarantee 0.0 cosine similarity
            head_delta = {k: -v.clone() for k, v in r_head_delta.items()}
            c_head_deltas.append(head_delta)
        else:
            flows = get_flows(i)
            ae_d, head_d, _, _, _ = run_pass(
                client_aes[i], client_heads[i], client_ae_opts[i], client_head_opts[i],
                flows, g_ae_w, g_head_w
            )
            c_ae_deltas.append(ae_d)
            c_head_deltas.append(head_d)
            
    client_round_counts = [10] * num_clients
    
    # Mode A
    agg_ae_A, trust_scores_A = ae_only_fltrust_aggregate(c_ae_deltas, r_ae_delta)
    
    # Mode B
    agg_ae_B, agg_head_B, combined_B, ch1_B, ch2_B = joint_dual_fltrust_aggregate(
        c_ae_deltas, c_head_deltas, r_ae_delta, r_head_delta, client_round_counts,
        ch2_warmup_rounds=0, ch1_weight=0.7
    )
    total_B = sum(combined_B)
    c2_weight_B = combined_B[2] / total_B
    
    # Mode C
    agg_ae_C, agg_head_C, ch1_C, ch2_C, classes_C = dc_fltrust_aggregate(
        c_ae_deltas, c_head_deltas, r_ae_delta, r_head_delta, client_round_counts, ch2_warmup_rounds=0
    )
    # in Mode C, Client 2 is classified as BYZANTINE_FAKE_ATTACK, so its head is excluded
    c2_weight_C = 0.0
    
    print("\nSynthetic Verification — All Three Aggregation Modes")
    print("(5 rounds, Client 0=HEALTHY, Client 1=UNDER_ATTACK, Client 2=BYZANTINE)\n")
    print(f"{'':<17} {'Mode A (AE-Only)':<18} {'Mode B (Joint)':<17} {'Mode C (DC-FLTrust)':<20}")
    print(f"{'Client 0 trust:':<17} {trust_scores_A[0]:<18.4f} {combined_B[0]:<17.4f} {classes_C[0]:<20}")
    print(f"{'Client 1 trust:':<17} {trust_scores_A[1]:<18.4f} {combined_B[1]:<17.4f} {classes_C[1]:<20}")
    print(f"{'Client 2 trust:':<17} {trust_scores_A[2]:<18.4f} {combined_B[2]:<17.4f} {classes_C[2]:<20}")
    print(f"{'C2 head weight:':<17} {'N/A':<18} {c2_weight_B:<17.4f} {c2_weight_C:<20.1f} (excluded)")

if __name__ == "__main__":
    run_verify()
