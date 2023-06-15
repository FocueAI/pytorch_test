import torch

x = torch.tensor([[1, 2, 0], [0, 0, 0], [3, 4, 5]])
mask = torch.all(torch.eq(x, 0), dim=1)
indices = torch.nonzero(mask)
print(mask)
print(indices)