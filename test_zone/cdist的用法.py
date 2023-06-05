import torch

x1 = torch.randn(2, 3)
x2 = torch.randn(4, 3)

dist1 = torch.cdist(x1, x2, p=2) # torch.Size([2, 4])
dist2 = torch.cdist(x2, x1, p=2) # torch.Size([4, 2])
print(dist1.shape, dist2.shape) # torch.Size([2, 4]), torch.Size([4, 2])
