import torch
import torch.nn as nn
nn.Conv2d()




a = torch.ones((2, 3))
print('======原始的a===========')
print(a)
a1 = torch.sum(a)
a2 = torch.sum(a, dim=0)
a3 = torch.sum(a, dim=1)
print('======torch.sum(a)===========')
print(a1, a1.shape)
print('======torch.sum(a, dim=0)===========')
print(a2, a2.shape)
print('======torch.sum(a, dim=1)===========')
print(a3, a3.shape)
