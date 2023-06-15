import torch

a = torch.tensor([[1.,2.,3.,4.,5.],[11.,22,33.,44.,55.]])
print(f'a:{a}')
b = torch.zeros(3, 5)
print('--------原始的 b -----------')
print(f'b:{b}')
b.scatter_(
    dim = 0,
    index = torch.LongTensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]),
    src = a
)
print('--------修改后的 b -----------')
print(f'b:{b}')


