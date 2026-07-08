import torch
import numpy as np
from collections import deque

class AttackReferenceBuffer:
    """
    Dynamic server-side buffer of confirmed attack z vectors.
    Updated when clients are classified UNDER ATTACK (high ch1, low ch2 trust 
    in single-channel, or via dual-channel disambiguation).
    Uses reservoir sampling to prevent old attack geometry from dominating.
    """
    
    def __init__(self, max_size=5000, min_size_to_use=50, device='cpu'):
        self.max_size = max_size
        self.min_size_to_use = min_size_to_use
        self.device = device
        self._buffer = deque(maxlen=max_size)
        self.rounds_updated = []
        self.total_submitted = 0
    
    def update(self, z_vectors: torch.Tensor, round_num: int):
        """Add z vectors from a client classified as UNDER ATTACK this round."""
        z_np = z_vectors.detach().cpu().numpy()
        for z in z_np:
            self._buffer.append(z)
        self.rounds_updated.append(round_num)
        self.total_submitted += len(z_vectors)
    
    def get_reference_tensor(self) -> torch.Tensor:
        """Returns current buffer as tensor. Returns None if buffer too small."""
        if len(self._buffer) < self.min_size_to_use:
            return None  # caller must fall back to static reference
        return torch.tensor(np.array(list(self._buffer)), 
                           dtype=torch.float32).to(self.device)
    
    def is_ready(self) -> bool:
        return len(self._buffer) >= self.min_size_to_use
    
    def stats(self) -> dict:
        return {
            'buffer_size': len(self._buffer),
            'total_submitted': self.total_submitted,
            'rounds_updated': self.rounds_updated[-5:],  # last 5 rounds
            'is_ready': self.is_ready()
        }
