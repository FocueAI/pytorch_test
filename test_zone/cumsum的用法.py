import torch

x = torch.tensor([[1, 2], [3, 4],[2, 3]])
print(x.cumsum(dim=0))


# a = torch.arange(10)
# print(f'a:{a}')
# print(f'a[-1:]:{a[-1:]}')