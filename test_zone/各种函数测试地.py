import numpy as np
import torch
subsequent_mask = np.triu(np.ones([5,5]), k=1)
subsequent_mask = torch.from_numpy(subsequent_mask).byte()
print(subsequent_mask)
#
# import torch.nn as nn
# nn.ReLU()
# x = torch.ones(1)
# x.add_(1)
# x.add(1)

# x = torch.randn(3, requires_grad=True)
# y = torch.randn(3, requires_grad=True)
# x = x.add(1)
# y +=x ** 2
# y.backward()

import torch

x = torch.randn(3, requires_grad=True)
y = torch.randn(3)

# In-place operation
# x.add_(y)
# x +=y
x = x+y

# Backward pass
x.sum().backward()

import torch.nn as nn
a = nn.Softmax(dim=0)(torch.tensor([-1e9, -1e9, -1e9, -1e9]))
# a = np.Softmax(dim=0)([1.8, 2.9])
print(f'a:{a}')


# -----------------------------------------

# a = torch.tensor([[1, 2], [3, 4]])
# b = torch.tensor([[5, 6], [7, 8]])
# c = torch.mm(a,b)
# d = torch.matmul(a,b)
# print('=-'*5)
# print(f'c:{c}')
# print(f'd:{d}')








