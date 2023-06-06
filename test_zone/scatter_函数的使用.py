import torch

a = torch.rand(2,5)
print(f'a:{a}')
b = torch.zeros(3, 5)
print(f'b:{b}')
b.scatter_(
    dim = 0,
    index = torch.LongTensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]),
    src = a
)
print(f'b:{b}')
