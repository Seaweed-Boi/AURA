"""
Byzantine Deception Experiment — Paper Evidence Collection
Implements the deceptive Byzantine attack variants used to validate
DC-FLTrust's channel 2 discrimination capability.

Run: python scripts/experiments/byzantine_deception_experiment.py --seeds 0 1 2

This is NOT part of the production benchmark. It is a standalone
experiment script for generating paper figures.
"""

def _run_deceptive_byzantine(ae, attack_head, benign_flows,
                              global_ae_weights, global_head_weights):
    """
    Deceptive Byzantine: trains AE honestly on benign flows
    (produces plausible ch1 gradient) but submits random AttackHead
    weights (produces incoherent ch2 gradient).
    
    This is undetectable by single-channel FLTrust.
    DC-FLTrust catches it because ch2 reveals the AttackHead
    gradient is not consistent with having seen real attack traffic.
    """
    import torch, torch.nn.functional as F
    # Honest AE training — gradient looks legitimate
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    optimizer.zero_grad()
    recon, _ = ae(benign_flows)
    loss = F.mse_loss(recon, benign_flows)
    loss.backward()
    optimizer.step()
    
    ae_delta = {k: ae.state_dict()[k].clone() - global_ae_weights[k]
                for k in ae.state_dict()}
    
    # Random AttackHead — gradient is incoherent garbage
    # Single-channel FLTrust never sees this
    # DC-FLTrust channel 2 will score this near zero
    head_delta = {k: torch.randn_like(v) * 0.1
                  for k, v in attack_head.state_dict().items()}
    
    return ae_delta, head_delta, [], len(benign_flows), 0
