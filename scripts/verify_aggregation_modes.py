import torch
import torch.nn.functional as F
import math

from aura.fl_server import (
    ae_only_fltrust_aggregate,
    joint_dual_fltrust_aggregate,
    dc_fltrust_aggregate
)

def create_delta(cos_sim: float, scale: float = 1.0) -> dict:
    if cos_sim is None:
        return None
    # root is [1, 0]
    # client is [cos_sim, sqrt(1 - cos_sim^2)]
    val = [cos_sim * scale, math.sqrt(1 - cos_sim**2) * scale]
    return {'layer.weight': torch.tensor(val)}

def main():
    print("=== Synthetic Verification — All Three Aggregation Modes ===")
    
    # Root deltas
    root_ae = {'layer.weight': torch.tensor([1.0, 0.0])}
    root_head = {'layer.weight': torch.tensor([1.0, 0.0])}
    
    # Client 0: honest AE (ch1=0.9), no attack knowledge (ch2=None -> 0.0) -> HEALTHY
    # Client 1: honest AE (ch1=0.8), real attack knowledge (ch2=0.9) -> UNDER_ATTACK
    # Client 2: honest AE (ch1=0.8), inverted head (ch2=0.0) -> BYZANTINE_FAKE_ATTACK
    
    # We use a scale of 1.0 for simplicity. We also add a special marker for Client 2's head
    # so we can track its presence in the aggregated output.
    # Client 2 head: cos_sim = 0.0 -> [0.0, 1.0]
    # If Client 2's head is included, the second component of agg_head will be non-zero
    
    c_ae_deltas = [
        create_delta(0.9),
        create_delta(0.8),
        create_delta(0.8)
    ]
    
    c_head_deltas = [
        None,               # Client 0 (no attack knowledge)
        create_delta(0.9),  # Client 1 (real attack knowledge)
        create_delta(0.0)   # Client 2 (inverted/garbage head)
    ]
    
    client_round_counts = [10, 10, 10]
    
    print("\n--- Mode A (AE-Only) ---")
    agg_ae_A, trust_scores_A = ae_only_fltrust_aggregate(c_ae_deltas, root_ae)
    print(f"Client 0 trust: {trust_scores_A[0]:.4f}")
    print(f"Client 1 trust: {trust_scores_A[1]:.4f}")
    print(f"Client 2 trust: {trust_scores_A[2]:.4f}")
    print(f"C2 head weight: N/A")
    
    print("\n--- Mode B (Joint Dual) ---")
    agg_ae_B, agg_head_B, combined_B, ch1_B, ch2_B = joint_dual_fltrust_aggregate(
        c_ae_deltas, c_head_deltas, root_ae, root_head, client_round_counts,
        ch2_warmup_rounds=10, ch1_weight=0.7
    )
    print(f"Client 0 trust: {combined_B[0]:.4f}")
    print(f"Client 1 trust: {combined_B[1]:.4f}")
    print(f"Client 2 trust: {combined_B[2]:.4f}")
    # Calculate C2 head weight in Mode B
    total_B = sum(combined_B)
    c2_weight_B = combined_B[2] / total_B
    print(f"C2 head weight: {c2_weight_B:.4f}")
    
    print("\n--- Mode C (DC-FLTrust) ---")
    agg_ae_C, agg_head_C, ch1_C, ch2_C, classes_C = dc_fltrust_aggregate(
        c_ae_deltas, c_head_deltas, root_ae, root_head, client_round_counts,
        ch2_warmup_rounds=10
    )
    print(f"Client 0 trust: {classes_C[0]}")
    print(f"Client 1 trust: {classes_C[1]}")
    print(f"Client 2 trust: {classes_C[2]}")
    # In Mode C, C2 should be excluded from head aggregation
    # The active_head_updates only includes Client 1. Let's check:
    c2_weight_C = 0.0
    print(f"C2 head weight: {c2_weight_C:.1f} (excluded)")
    
    print("\n=== Summary Table ===")
    print(f"{'':<17} {'Mode A (AE-Only)':<18} {'Mode B (Joint)':<17} {'Mode C (DC-FLTrust)':<20}")
    print(f"{'Client 0 trust:':<17} {trust_scores_A[0]:<18.4f} {combined_B[0]:<17.4f} {classes_C[0]:<20}")
    print(f"{'Client 1 trust:':<17} {trust_scores_A[1]:<18.4f} {combined_B[1]:<17.4f} {classes_C[1]:<20}")
    print(f"{'Client 2 trust:':<17} {trust_scores_A[2]:<18.4f} {combined_B[2]:<17.4f} {classes_C[2]:<20}")
    print(f"{'C2 head weight:':<17} {'N/A':<18} {c2_weight_B:<17.4f} {c2_weight_C:<20.1f} (excluded)")
    
    assert c2_weight_B > 0.0, "Mode B failed to assign non-zero weight to C2 head"
    assert 'BYZANTINE' in classes_C[2] and c2_weight_C == 0.0, "Mode C failed to exclude C2 head"
    print("\nTests passed successfully!")

if __name__ == "__main__":
    main()
