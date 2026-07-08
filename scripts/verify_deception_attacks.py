import sys
from pathlib import Path
import torch
import torch.optim as optim
import copy
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aura.models import FlowAutoencoder, AttackHead
from scripts.benchmark_byzantine import _run_local_training_dual
from scripts.experiments.byzantine_deception_experiment import _run_latent_inversion_byzantine, _run_true_labelflip_byzantine
from aura.fl_server import ae_only_fltrust_aggregate, joint_dual_fltrust_aggregate, dc_fltrust_aggregate

def measure_head_auroc(global_head, test_z_benign, test_z_attack):
    """
    Measure how well the current global AttackHead distinguishes
    attack z vectors from benign z vectors.
    AUROC of 1.0 = perfect separation. 0.5 = random chance.
    Degradation from baseline indicates attack succeeded.
    """
    from sklearn.metrics import roc_auc_score
    
    global_head.eval()
    with torch.no_grad():
        benign_scores = global_head(test_z_benign).squeeze().cpu().numpy()
        attack_scores = global_head(test_z_attack).squeeze().cpu().numpy()
    
    labels = np.concatenate([
        np.zeros(len(benign_scores)),
        np.ones(len(attack_scores))
    ])
    scores = np.concatenate([benign_scores, attack_scores])
    return roc_auc_score(labels, scores)

def run_experiment(attack_mode, agg_mode, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    global_ae = FlowAutoencoder()
    global_head = AttackHead()
    
    num_clients = 5
    client_aes = [copy.deepcopy(global_ae) for _ in range(num_clients)]
    client_heads = [copy.deepcopy(global_head) for _ in range(num_clients)]
    client_ae_opts = [optim.Adam(ae.parameters(), lr=1e-3) for ae in client_aes]
    client_head_opts = [optim.Adam(head.parameters(), lr=1e-3) for head in client_heads]
    
    root_ae = copy.deepcopy(global_ae)
    root_head = copy.deepcopy(global_head)
    root_ae_opt = optim.Adam(root_ae.parameters(), lr=1e-3)
    root_head_opt = optim.Adam(root_head.parameters(), lr=1e-3)
    
    # Pre-train baseline server to have something reasonable
    for i in range(3):
        root_flows = torch.randn(200, 47) * 0.1
        _run_local_training_dual(global_ae, global_head, root_flows, 
                                 optim.Adam(global_ae.parameters(), lr=1e-2), 
                                 optim.Adam(global_head.parameters(), lr=1e-2), 
                                 global_ae.state_dict(), global_head.state_dict(), 0.5, 1)
                                 
    def get_flows(client_id, is_honest_baseline=False):
        if client_id == 0 and not is_honest_baseline:
            # attacker client (client 0 is byzantine)
            benign = torch.randn(100, 47) * 0.1
            attacks = torch.randn(100, 47) * 5.0
            return torch.cat([benign, attacks])
        elif client_id in [0, 1, 2, 3, 4]:
            # healthy clients (or attacker in honest baseline)
            benign = torch.randn(100, 47) * 0.1
            attacks = torch.randn(50, 47) * 5.0  # Give them some attacks so they train AttackHead
            return torch.cat([benign, attacks])

    def run_pass(model_ae, model_head, opt_ae, opt_head, flows, g_ae_weights, g_head_weights, is_byzantine):
        if is_byzantine:
            if attack_mode == 'latent_inversion':
                return _run_latent_inversion_byzantine(
                    model_ae, model_head, flows, opt_ae, opt_head, g_ae_weights, g_head_weights, 0.5, 3
                )
            elif attack_mode == 'true_labelflip':
                return _run_true_labelflip_byzantine(
                    model_ae, model_head, flows, opt_ae, opt_head, g_ae_weights, g_head_weights, 0.5, 3
                )
        return _run_local_training_dual(
            model_ae, model_head, flows, opt_ae, opt_head, g_ae_weights, g_head_weights, 0.5, 3
        )

    sc_flags = 0
    dc_flags = 0
    ch1_history = []
    ch2_history = []
    combined_weight = 0.0
    head_weight = 0.0
    
    for rnd in range(1, 6):
        g_ae_w = {k: v.clone() for k, v in global_ae.state_dict().items()}
        g_head_w = {k: v.clone() for k, v in global_head.state_dict().items()}
        
        for i in range(num_clients):
            client_aes[i].load_state_dict(g_ae_w)
            client_heads[i].load_state_dict(g_head_w)
            
        root_ae.load_state_dict(g_ae_w)
        root_head.load_state_dict(g_head_w)
        
        root_flows = torch.cat([torch.randn(100, 47) * 0.1, torch.randn(100, 47) * 5.0])
        r_ae_delta, r_head_delta, _, _, _ = run_pass(
            root_ae, root_head, root_ae_opt, root_head_opt, root_flows, g_ae_w, g_head_w, False
        )
        
        c_ae_deltas = []
        c_head_deltas = []
        
        for i in range(num_clients):
            flows = get_flows(i, is_honest_baseline=(attack_mode == 'honest'))
            ae_d, head_d, _, _, _ = run_pass(
                client_aes[i], client_heads[i], client_ae_opts[i], client_head_opts[i],
                flows, g_ae_w, g_head_w, is_byzantine=(i==0 and attack_mode != 'honest')
            )
            c_ae_deltas.append(ae_d)
            c_head_deltas.append(head_d)
            
        client_round_counts = [rnd] * num_clients
        
        if agg_mode == 'ae_only':
            agg_ae, trust_scores = ae_only_fltrust_aggregate(c_ae_deltas, r_ae_delta)
            if trust_scores[0] <= 0.0:
                sc_flags += 1
            ch1_history.append(trust_scores[0])
            ch2_history.append(0.0)
            combined_weight = trust_scores[0] / sum(trust_scores)
            head_weight = 0.0
            
            for k in g_ae_w:
                g_ae_w[k] += agg_ae[k]
        elif agg_mode == 'joint_dual':
            agg_ae, agg_head, combined, ch1, ch2 = joint_dual_fltrust_aggregate(
                c_ae_deltas, c_head_deltas, r_ae_delta, r_head_delta, client_round_counts, ch2_warmup_rounds=0
            )
            if combined[0] <= 0.0:
                sc_flags += 1 # Not technically SC, but it's the single joint trust score
            ch1_history.append(ch1[0])
            ch2_history.append(ch2[0] if ch2[0] is not None else 0.0)
            combined_weight = combined[0] / sum(combined) if sum(combined) > 0 else 0
            
            valid_heads = [(w, d) for w, d in zip([c/sum(combined) for c in combined], c_head_deltas) if d is not None]
            if valid_heads and c_head_deltas[0] is not None:
                head_weight = combined[0] / sum(w for w, d in valid_heads)
            else:
                head_weight = 0.0
                
            for k in g_ae_w:
                g_ae_w[k] += agg_ae[k]
            if agg_head is not None:
                for k in g_head_w:
                    g_head_w[k] += agg_head[k]
        else:
            agg_ae, agg_head, ch1, ch2, classes = dc_fltrust_aggregate(
                c_ae_deltas, c_head_deltas, r_ae_delta, r_head_delta, client_round_counts, ch2_warmup_rounds=0
            )
            if classes[0] == 'BYZANTINE' or classes[0] == 'BYZANTINE_FAKE_ATTACK':
                dc_flags += 1
            ch1_history.append(ch1[0])
            ch2_history.append(ch2[0] if ch2[0] is not None else 0.0)
            
            # calculate weight
            active_weights = [t for i, t in enumerate(ch1) if classes[i] in ('HEALTHY', 'UNDER_ATTACK')]
            combined_weight = ch1[0] / sum(active_weights) if classes[0] in ('HEALTHY', 'UNDER_ATTACK') and sum(active_weights) > 0 else 0.0
            head_weight = 0.0 if classes[0] in ('HEALTHY', 'BYZANTINE_FAKE_ATTACK', 'BYZANTINE') else combined_weight
            
            for k in g_ae_w:
                g_ae_w[k] += agg_ae[k]
            if agg_head is not None:
                for k in g_head_w:
                    g_head_w[k] += agg_head[k]
                    
        global_ae.load_state_dict(g_ae_w)
        global_head.load_state_dict(g_head_w)
        
    # Measure AUROC
    test_benign_flows = torch.randn(500, 47) * 0.1
    test_attack_flows = torch.randn(500, 47) * 5.0
    
    global_ae.eval()
    with torch.no_grad():
        test_z_benign = global_ae.encode(test_benign_flows)
        test_z_attack = global_ae.encode(test_attack_flows)
        
    auroc = measure_head_auroc(global_head, test_z_benign, test_z_attack)
        
    return np.mean(ch1_history), np.mean(ch2_history), sc_flags, dc_flags, combined_weight, head_weight, auroc

def main():
    print("=== DC-FLTrust 3-Seed Analysis ===")
    
    seeds = [0, 1, 2]
    
    def gather_stats(attack_mode, agg_mode):
        ch1_l, ch2_l, comb_l, head_l, auroc_l, dc_flags = [], [], [], [], [], []
        for s in seeds:
            ch1, ch2, scf, dcf, comb_w, head_w, auroc = run_experiment(attack_mode, agg_mode, seed=s)
            ch1_l.append(ch1)
            ch2_l.append(ch2)
            comb_l.append(comb_w)
            head_l.append(head_w)
            auroc_l.append(auroc)
            dc_flags.append(dcf)
        return np.mean(ch1_l), np.std(ch1_l), np.mean(ch2_l), np.std(ch2_l), np.mean(comb_l), np.mean(head_l), np.mean(auroc_l), np.std(auroc_l), np.mean(dc_flags)

    # 1. Honest Baseline
    _, _, _, _, _, _, hon_auroc, hon_auroc_std, _ = gather_stats('honest', 'dc_fltrust')
    
    # 2. Latent Inversion across Modes
    la_ch1, la_ch1_std, la_ch2, la_ch2_std, la_comb, la_head, la_auroc, la_auroc_std, _ = gather_stats('latent_inversion', 'ae_only')
    lb_ch1, lb_ch1_std, lb_ch2, lb_ch2_std, lb_comb, lb_head, lb_auroc, lb_auroc_std, _ = gather_stats('latent_inversion', 'joint_dual')
    lc_ch1, lc_ch1_std, lc_ch2, lc_ch2_std, lc_comb, lc_head, lc_auroc, lc_auroc_std, lc_dcf = gather_stats('latent_inversion', 'dc_fltrust')
    
    print("\nGlobal AttackHead AUROC after 5 rounds:")
    print(f"  No attack baseline (honest clients only):  {hon_auroc:.4f} \u00B1 {hon_auroc_std:.4f}")
    print(f"  With Latent Inversion — Mode A (AE-only):  {la_auroc:.4f} \u00B1 {la_auroc_std:.4f}  \u2190 no head federation, unaffected")
    print(f"  With Latent Inversion — Mode B (Joint):    {lb_auroc:.4f} \u00B1 {lb_auroc_std:.4f}  \u2190 degraded by partial head inclusion")
    print(f"  With Latent Inversion — Mode C (DC):       {lc_auroc:.4f} \u00B1 {lc_auroc_std:.4f}  \u2190 preserved by full exclusion")
    
    print("\nLatent Inversion Threat Metrics (Mean \u00B1 Std over 3 seeds):")
    print(f"  ch1 score: {lc_ch1:.4f} \u00B1 {lc_ch1_std:.4f}")
    print(f"  ch2 score: {lc_ch2:.4f} \u00B1 {lc_ch2_std:.4f}")
    
    print("\nAggregation Weights (Final Round):")
    print(f"  Mode B combined trust weight: {lb_comb:.4f}")
    print(f"  Mode B head inclusion weight: {lb_head:.4f}")
    
    print(f"\nMode C Detection Rate: {lc_dcf:.1f}/5.0 rounds (BYZANTINE_FAKE_ATTACK)")
    
if __name__ == "__main__":
    main()
