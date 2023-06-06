import torch

a = torch.arange(0,9)
b = torch.arange(10,19)
c = torch.arange(20,29)

tot = torch.stack((a,b,c),dim=1).transpose(1,0)
print(tot)

tot1 = torch.cat((a[None], b[None], c[None]))
print(tot1)

print('*'*10)
choice = tot1[ [[0,2]]]
choice_ = tot1[ [0,2]]
print(f'choice:{choice}')
print(f'choice_:{choice_}')
print('='*10)
choice1 = tot1[ [[0,2],[3,4]]]
print(f'choice1:{choice1}')
