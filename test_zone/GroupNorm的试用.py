import torch.nn as nn
import torch

input = torch.randn(16, 6, 8, 8)
group_norm = nn.GroupNorm(3, 6)
output = group_norm(input)
print(f'output-shape:{output.shape}')