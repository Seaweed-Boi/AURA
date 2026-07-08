"""
Byzantine Deception Experiment — Paper Evidence Collection
Implements the deceptive Byzantine attack variants used to validate
DC-FLTrust's channel 2 discrimination capability.

Run: python scripts/experiments/byzantine_deception_experiment.py --seeds 0 1 2

This is NOT part of the production benchmark. It is a standalone
experiment script for generating paper figures.
"""

import torch
import torch.nn.functional as F

def _run_latent_inversion_byzantine(
    ae, attack_head, all_flows, ae_optimizer, head_optimizer,
    global_ae_weights, global_head_weights, mse_threshold_high,
    head_epochs=3
):
    """
    Latent Inversion Byzantine Attack.

    The attacker trains its AE honestly on benign traffic (produces a 
    legitimate ch1 gradient indistinguishable from an honest client) but 
    trains its AttackHead with inverted pseudo-labels on high-MSE latent 
    representations (z vectors from the AE bottleneck).

    Specifically: honest clients label high-MSE z vectors as attack 
    (pseudo_label=1). This attacker labels them as benign (pseudo_label=0),
    producing an AttackHead gradient that is anti-aligned with genuine 
    attack-pattern learning.

    Effect on global model:
      - AE channel: unaffected (honest training)
      - AttackHead channel: progressively corrupted toward classifying 
        attack latent representations as benign

    Detectability:
      - Single-channel FLTrust: UNDETECTABLE
        The full-bundle gradient is dominated by the honest AE component 
        (5,487 parameters vs 145 AttackHead parameters = 97% honest).
        Cosine similarity with root is positive → client accepted.
      - DC-FLTrust channel 2: DETECTED
        AttackHead gradient is anti-aligned with the server's attack 
        reference (negative cosine similarity → ch2=0 after ReLU).
        Classification: BYZANTINE_FAKE_ATTACK.

    Threat model realism: An adversary who has compromised a federated 
    client and can observe which flows produce high reconstruction error 
    (a reasonable assumption given access to the local model) can execute 
    this attack without any knowledge of other clients' data or the 
    server's aggregation weights.
    """
    # Honest AE training
    ae.eval()
    with torch.no_grad():
        recon, _ = ae(all_flows)
        mse_per_flow = F.mse_loss(
            recon, all_flows, reduction='none'
        ).mean(dim=1)
    ae.train()
    
    benign_mask = mse_per_flow < mse_threshold_high
    high_mse_mask = ~benign_mask
    
    benign_flows = all_flows[benign_mask]
    if len(benign_flows) > 0:
        ae_optimizer.zero_grad()
        recon_b, _ = ae(benign_flows)
        ae_loss = F.mse_loss(recon_b, benign_flows)
        ae_loss.backward()
        ae_optimizer.step()
        
    ae_delta = {k: ae.state_dict()[k].clone() - global_ae_weights[k]
                for k in ae.state_dict()}
                
    # Latent Inversion on AttackHead
    attack_flows = all_flows[high_mse_mask]
    z_buffer = []
    
    if len(attack_flows) > 0:
        ae.eval()
        with torch.no_grad():
            for i in range(0, len(attack_flows), 256):
                batch = attack_flows[i:i+256]
                z = ae.encode(batch)
                z_buffer.append(z.detach().cpu())
        ae.train()
        
        z_tensor = torch.cat(z_buffer)
        for _ in range(head_epochs):
            head_optimizer.zero_grad()
            preds = attack_head(z_tensor).squeeze()
            # LATENT INVERSION: Label high-MSE z vectors as BENIGN (0) instead of ATTACK (1)
            inverted_labels = torch.zeros(len(z_tensor))
            head_loss = F.binary_cross_entropy(preds, inverted_labels)
            head_loss.backward()
            head_optimizer.step()
            
    if len(attack_flows) > 0:
        head_delta = {k: attack_head.state_dict()[k].clone() - global_head_weights[k]
                      for k in attack_head.state_dict()}
    else:
        head_delta = None

    return ae_delta, head_delta, z_buffer, len(benign_flows), len(attack_flows)


def _run_true_labelflip_byzantine(
    ae, attack_head, all_flows, ae_optimizer, head_optimizer,
    global_ae_weights, global_head_weights, mse_threshold_high,
    head_epochs=3
):
    """
    True Label-Flip Byzantine Attack.
    
    Flips the benign/attack classification at the raw data level.
    The AE trains on HIGH-MSE flows (attack traffic) as if they were 
    benign — corrupting the AE's reconstruction boundary.
    The AttackHead trains on LOW-MSE flows (benign traffic) with 
    pseudo_label=1 — corrupting attack detection in latent space.
    
    This is a stronger attack than Latent Inversion because it corrupts
    the AE's representation of normality, affecting both channels.
    However it is also more detectable because the AE gradient diverges
    significantly from an honest client's AE gradient (ch1 drops).
    
    Use this as a comparison point to show that Latent Inversion is 
    the harder attack — it achieves AttackHead corruption while 
    maintaining AE gradient legitimacy.
    """
    ae.eval()
    with torch.no_grad():
        recon, _ = ae(all_flows)
        mse_per_flow = F.mse_loss(
            recon, all_flows, reduction='none'
        ).mean(dim=1)
    ae.train()
    
    benign_mask = mse_per_flow < mse_threshold_high
    high_mse_mask = ~benign_mask
    
    # TRUE LABEL FLIP: train AE on attack flows (treating them as benign)
    attack_flows = all_flows[high_mse_mask]
    if len(attack_flows) > 0:
        ae_optimizer.zero_grad()
        recon_a, _ = ae(attack_flows)
        ae_loss = F.mse_loss(recon_a, attack_flows)
        ae_loss.backward()
        ae_optimizer.step()
    
    ae_delta = {
        k: ae.state_dict()[k].clone() - global_ae_weights[k]
        for k in ae.state_dict()
    }
    
    # Train AttackHead on benign z vectors with pseudo_label=1
    benign_flows = all_flows[benign_mask]
    z_buffer = []
    ae.eval()
    with torch.no_grad():
        if len(benign_flows) > 0:
            for i in range(0, len(benign_flows), 256):
                batch = benign_flows[i:i+256]
                z = ae.encode(batch)
                z_buffer.append(z.detach().cpu())
    ae.train()
    
    if z_buffer:
        z_tensor = torch.cat(z_buffer)
        for _ in range(head_epochs):
            head_optimizer.zero_grad()
            preds = attack_head(z_tensor).squeeze()
            # Label benign z vectors as attacks
            flipped_labels = torch.ones(len(z_tensor))
            head_loss = F.binary_cross_entropy(preds, flipped_labels)
            head_loss.backward()
            head_optimizer.step()
    
    if z_buffer:
        head_delta = {
            k: attack_head.state_dict()[k].clone() - global_head_weights[k]
            for k in attack_head.state_dict()
        }
    else:
        head_delta = None
        
    return ae_delta, head_delta, z_buffer, len(attack_flows), len(benign_flows)
