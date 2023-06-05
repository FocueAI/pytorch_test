import torch

x = torch.randn(3, 4)
print('before-shape:',x.shape)
print('before:',x)

print('='*6)
y = x[..., None]
print('after-shape:',y.shape)
print('after:',y)
print('-'*6)
print(x.unsqueeze(-1))