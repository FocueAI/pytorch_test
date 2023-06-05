import torch

x = torch.randn(2, 3, 4)
y = x.flatten(1)
print(f'y-shape:{y.shape}') # y-shape:torch.Size([2, 12])