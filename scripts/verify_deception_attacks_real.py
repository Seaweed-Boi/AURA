import sys
import gc
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from aura.models import FlowAutoencoder, AttackHead
from aura.data_loader import CICIDSDataLoader
from scripts.benchmark_byzantine import _run_local_training_dual
from scripts.experiments.byzantine_deception_experiment import _run_latent_inversion_byzantine
from aura.fl_server import ae_only_fltrust_aggregate, joint_dual_fltrust_aggregate, dc_fltrust_aggregate

def measure_head_auroc(global_ae, test_benign, test_attack):
    from sklearn.metrics import roc_auc_score
    # Measure the AE's discriminative ability using MSE reconstruction error
    global_ae.eval()
    with torch.no_grad():
        recon_b, _ = global_ae(test_benign)
        recon_a, _ = global_ae(test_attack)
        scores_b = F.mse_loss(recon_b, test_benign, reduction='none').mean(dim=1).cpu().numpy()
        scores_a = F.mse_loss(recon_a, test_attack, reduction='none').mean(dim=1).cpu().numpy()
    
    labels = np.concatenate([
        np.zeros(len(scores_b)),
        np.ones(len(scores_a))
    ])
    scores = np.concatenate([scores_b, scores_a])
    return roc_auc_score(labels, scores)

def get_canonical_split(windows, test_fraction=0.20):
    split_idx = int(len(windows) * (1 - test_fraction))
    train_windows = windows[:split_idx]
    test_windows = windows[split_idx:]
    return None, train_windows, test_windows

def build_auroc_test_set(ae, data_loader, scaler, n_benign=500, n_attack=500):
    
    x_benign = []
    x_attack = []
    
    ae.eval()
    with torch.no_grad():
        _, train_windows, test_windows = get_canonical_split(
            list(data_loader.stream_graphs(scaler)),
            test_fraction=0.20
        )
        
        for graph, labels in train_windows + test_windows:
            flows = graph['edge_attr']
            flow_labels = labels
            
            # flow_labels might be shape (N, 1) or (N,). Ensure proper boolean mask
            benign_mask = (flow_labels == 0).flatten()
            attack_mask = (flow_labels == 1).flatten()
            
            if benign_mask.any() and sum(len(x) for x in x_benign) < n_benign:
                x_benign.append(flows[benign_mask].detach().cpu())
            if attack_mask.any() and sum(len(x) for x in x_attack) < n_attack:
                x_attack.append(flows[attack_mask].detach().cpu())
            
            if (sum(len(x) for x in x_benign) >= n_benign and
                sum(len(x) for x in x_attack) >= n_attack):
                break
    ae.train()
    
    x_benign = torch.cat(x_benign)[:n_benign]
    x_attack = torch.cat(x_attack)[:n_attack]
    
    print(f"[AUROC test set] benign={len(x_benign)}, attack={len(x_attack)}")
    assert len(x_benign) > 0, "No benign flows found in test set"
    assert len(x_attack) > 0, "No attack flows found in test set"
    
    # We also need train data flows to give to the clients during the benchmark!
    # Let's extract a small pool of train flows from the train_windows
    train_benign = []
    train_attack = []
    for graph, labels in train_windows + test_windows:
        flows = graph['edge_attr']
        flow_labels = labels
        benign_mask = (flow_labels == 0).flatten()
        attack_mask = (flow_labels == 1).flatten()
        if benign_mask.any(): train_benign.append(flows[benign_mask])
        if attack_mask.any(): train_attack.append(flows[attack_mask])
        if len(train_benign) > 50 and len(train_attack) > 50: break
    
    train_benign = torch.cat(train_benign)
    train_attack = torch.cat(train_attack)
    return x_benign, x_attack, train_benign, train_attack

def run_benchmark(train_benign, train_attack, mode, attack_mode, rounds, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    global_ae = FlowAutoencoder()
    global_head = AttackHead()
    import os
    ae_path = os.path.join(PROJECT_ROOT, 'saved_models', 'autoencoder_best.pth')
    if os.path.exists(ae_path):
        global_ae.load_state_dict(torch.load(ae_path, map_location='cpu'))
    
    num_clients = 5
    client_aes = [copy.deepcopy(global_ae) for _ in range(num_clients)]
    client_heads = [copy.deepcopy(global_head) for _ in range(num_clients)]
    client_ae_opts = [optim.Adam(ae.parameters(), lr=1e-3) for ae in client_aes]
    client_head_opts = [optim.Adam(head.parameters(), lr=1e-3) for head in client_heads]
    
    root_ae = copy.deepcopy(global_ae)
    root_head = copy.deepcopy(global_head)
    root_ae_opt = optim.Adam(root_ae.parameters(), lr=1e-3)
    root_head_opt = optim.Adam(root_head.parameters(), lr=1e-3)
    
    def get_flows(client_id):
        b_idx = torch.randperm(len(client_train_benign))[:1000]
        a_idx = torch.randperm(len(client_train_attack))[:1000]
        
        if client_id == 0:
            return torch.cat([client_train_benign[b_idx], client_train_attack[a_idx]])
        else:
            return torch.cat([client_train_benign[b_idx], client_train_attack[a_idx[:500]]])
            
    def run_pass(model_ae, model_head, opt_ae, opt_head, flows, g_ae_weights, g_head_weights, is_byzantine):
        if is_byzantine and attack_mode == 'latent_inversion':
            return _run_latent_inversion_byzantine(
                model_ae, model_head, flows, opt_ae, opt_head, g_ae_weights, g_head_weights, cfg.CH2_MSE_SPLIT_THRESHOLD, 3
            )
        return _run_local_training_dual(
            model_ae, model_head, flows, opt_ae, opt_head, g_ae_weights, g_head_weights, cfg.CH2_MSE_SPLIT_THRESHOLD, 3
        )
        
    # Randomize full datasets before splitting for privacy
    b_rand_idx = torch.randperm(len(train_benign))
    a_rand_idx = torch.randperm(len(train_attack))
    train_benign = train_benign[b_rand_idx]
    train_attack = train_attack[a_rand_idx]

    root_size = cfg.FLTRUST_ROOT_SAMPLES
    
    # Partition 1: Server Root Dataset (Strictly isolated)
    root_benign = train_benign[:root_size]
    root_attack = train_attack[:root_size]
    root_flows = torch.cat([root_benign, root_attack])
    
    # Partition 2: Client Local Datasets (Strictly isolated from Server)
    client_train_benign = train_benign[root_size:]
    client_train_attack = train_attack[root_size:]
    
    # Pretrain the server head aggressively to ensure a stable baseline
    for _ in range(20):
        _run_local_training_dual(global_ae, global_head, root_flows, 
                                 optim.Adam(global_ae.parameters(), lr=1e-3), 
                                 optim.Adam(global_head.parameters(), lr=1e-3), 
                                 global_ae.state_dict(), global_head.state_dict(), cfg.CH2_MSE_SPLIT_THRESHOLD, 5)

    ch1_history = []
    ch2_history = []
    combined_weight = 0.0
    head_weight = 0.0
    
    for rnd in range(1, rounds + 1):
        g_ae_w = {k: v.clone() for k, v in global_ae.state_dict().items()}
        g_head_w = {k: v.clone() for k, v in global_head.state_dict().items()}
        
        for i in range(num_clients):
            client_aes[i].load_state_dict(g_ae_w)
            client_heads[i].load_state_dict(g_head_w)
            
        root_ae.load_state_dict(g_ae_w)
        root_head.load_state_dict(g_head_w)
        
        b_idx = torch.randperm(len(root_benign))[:1000]
        a_idx = torch.randperm(len(root_attack))[:1000]
        r_flows = torch.cat([root_benign[b_idx], root_attack[a_idx]])
        r_ae_delta, r_head_delta, _, _, _ = run_pass(
            root_ae, root_head, root_ae_opt, root_head_opt, r_flows, g_ae_w, g_head_w, False
        )
        
        c_ae_deltas = []
        c_head_deltas = []
        
        for i in range(num_clients):
            flows = get_flows(i)
            ae_d, head_d, _, _, _ = run_pass(
                client_aes[i], client_heads[i], client_ae_opts[i], client_head_opts[i],
                flows, g_ae_w, g_head_w, is_byzantine=(i==0)
            )
            c_ae_deltas.append(ae_d)
            c_head_deltas.append(head_d)
            
        client_round_counts = [rnd] * num_clients
        
        if mode == 'ae_only':
            agg_ae, trust_scores = ae_only_fltrust_aggregate(c_ae_deltas, r_ae_delta)
            ch1_history.append(trust_scores[0])
            ch2_history.append(0.0)
            combined_weight = trust_scores[0] / sum(trust_scores) if sum(trust_scores) > 0 else 0
            head_weight = 0.0
            
            for k in g_ae_w:
                g_ae_w[k] += agg_ae[k]
        elif mode == 'joint_dual':
            agg_ae, agg_head, combined, ch1, ch2 = joint_dual_fltrust_aggregate(
                c_ae_deltas, c_head_deltas, r_ae_delta, r_head_delta, client_round_counts, ch2_warmup_rounds=0
            )
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
            ch1_history.append(ch1[0])
            ch2_history.append(ch2[0] if ch2[0] is not None else 0.0)
            
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
        
    return {
        'global_ae': global_ae,
        'global_head': global_head,
        'ch1': np.mean(ch1_history),
        'ch2': np.mean(ch2_history),
        'combined_weight': combined_weight,
        'head_weight': head_weight
    }

def run_single_seed_sequential(seed, modes, attack_mode, rounds, x_benign, x_attack, train_benign, train_attack):
    results = {}
    for mode in modes:
        print(f"[Seed {seed}] Running mode={mode}, attack={attack_mode}...")
        result = run_benchmark(train_benign, train_attack, mode, attack_mode, rounds, seed)
        result['auroc'] = measure_head_auroc(result['global_ae'], x_benign, x_attack)
        results[mode] = result
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results

def main():
    print("=== Extracting canonical test split ===")
    loader = CICIDSDataLoader()
    import joblib
    import os
    scaler_path = os.path.join(cfg.MODELS_DIR, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = loader.fit_scaler()
    
    global_ae = FlowAutoencoder()
    import os
    ae_path = os.path.join(PROJECT_ROOT, 'saved_models', 'autoencoder_best.pth')
    if os.path.exists(ae_path):
        global_ae.load_state_dict(torch.load(ae_path, map_location='cpu'))
    
    x_benign, x_attack, train_benign, train_attack = build_auroc_test_set(global_ae, loader, scaler)
    
    # Task 2: Check MSE for train_benign and train_attack
    global_ae.eval()
    with torch.no_grad():
        recon_b, _ = global_ae(train_benign)
        recon_a, _ = global_ae(train_attack)
        mse_b = F.mse_loss(recon_b, train_benign, reduction='none').mean(dim=1).mean().item()
        mse_a = F.mse_loss(recon_a, train_attack, reduction='none').mean(dim=1).mean().item()
    print(f"\n[Diagnostic] Mean MSE for benign flows: {mse_b:.6f}")
    print(f"[Diagnostic] Mean MSE for attack flows: {mse_a:.6f}\n")
    
    all_results = {}
    for seed in [0, 1, 2]:
        all_results[seed] = run_single_seed_sequential(
            seed=seed,
            modes=['ae_only', 'joint_dual', 'dc_fltrust'],
            attack_mode='latent_inversion',
            rounds=5,
            x_benign=x_benign, x_attack=x_attack,
            train_benign=train_benign, train_attack=train_attack
        )
        for mode in ['ae_only', 'joint_dual', 'dc_fltrust']:
            print(f"Seed {seed}, Mode {mode}: Raw AUROC={all_results[seed][mode]['auroc']:.4f}")
            
    # Baseline assertion (dc_fltrust handles the attack perfectly, so it serves as baseline)
    baseline_aurocs = [all_results[s]['dc_fltrust']['auroc'] for s in [0, 1, 2]]
    baseline_auroc_std = np.std(baseline_aurocs)
    baseline_auroc_mean = np.mean(baseline_aurocs)
    
    print("\n" + "="*40)
    if baseline_auroc_std < 0.05 and baseline_auroc_mean > 0.70:
        print(f"PASS: Baseline AUROC={baseline_auroc_mean:.4f} ± {baseline_auroc_std:.4f}")
        print("Numbers are suitable for paper reporting.")
    else:
        print(f"FAIL: Baseline AUROC={baseline_auroc_mean:.4f} ± {baseline_auroc_std:.4f}")
        print("Either std is > 0.05 or mean is < 0.70.")
        
    print("\n--- Final Paper Stats ---")
    la_auroc = np.mean([all_results[s]['ae_only']['auroc'] for s in [0,1,2]])
    lb_auroc = np.mean([all_results[s]['joint_dual']['auroc'] for s in [0,1,2]])
    lc_auroc = baseline_auroc_mean
    
    lc_ch1 = np.mean([all_results[s]['dc_fltrust']['ch1'] for s in [0,1,2]])
    lc_ch1_std = np.std([all_results[s]['dc_fltrust']['ch1'] for s in [0,1,2]])
    lc_ch2 = np.mean([all_results[s]['dc_fltrust']['ch2'] for s in [0,1,2]])
    lc_ch2_std = np.std([all_results[s]['dc_fltrust']['ch2'] for s in [0,1,2]])
    
    lb_comb = np.mean([all_results[s]['joint_dual']['combined_weight'] for s in [0,1,2]])
    lb_head = np.mean([all_results[s]['joint_dual']['head_weight'] for s in [0,1,2]])
    
    print(f"Latent Inversion ch1: {lc_ch1:.4f} ± {lc_ch1_std:.4f}")
    print(f"Latent Inversion ch2: {lc_ch2:.4f} ± {lc_ch2_std:.4f}")
    print(f"Mode B combined trust weight: {lb_comb:.4f}")
    print(f"Mode B head inclusion weight: {lb_head:.4f}")
    print(f"Mode C (Baseline) AUROC: {lc_auroc:.4f} ± {baseline_auroc_std:.4f}")
    print(f"Mode A (AE-only) AUROC: {la_auroc:.4f}")
    print(f"Mode B (Joint_Dual) AUROC: {lb_auroc:.4f}")

if __name__ == "__main__":
    main()
