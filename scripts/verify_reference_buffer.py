import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aura.attack_reference import AttackReferenceBuffer
import torch

buf = AttackReferenceBuffer(max_size=100, min_size_to_use=10)

# Before minimum size — should return None
assert buf.get_reference_tensor() is None
assert not buf.is_ready()

# Add enough vectors
for i in range(15):
    buf.update(torch.randn(5, 16), round_num=i)

# Should now be ready
assert buf.is_ready()
ref = buf.get_reference_tensor()
assert ref.shape[1] == 16  # z vectors are 16-dim, not 47
assert ref.shape[0] <= 100  # respects max_size

# Test reservoir — add beyond max_size
for i in range(200):
    buf.update(torch.randn(1, 16), round_num=100+i)
assert len(buf._buffer) == 100  # capped at max_size

print("PASS: buffer mechanics correct")
print(f"Stats: {buf.stats()}")
