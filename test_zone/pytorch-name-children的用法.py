import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear_1 = nn.Linear(1, 2)
        self.linear_2 = nn.Linear(2, 1)
        self.linear_3 = nn.Linear(1, 3)
        self.conv1 = nn.Conv2d(1,2,2)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        x = self.conv1(x)
        return x

net = LinearModel()
print(f'-'*6,'net.state_dict()','-'*6)
print(net.state_dict())


print(f'-'*6,'net.parameters()','-'*6)
for i in net.parameters():
    print(f'i:{i}')

print(f'-*'*6,'net.named_parameters()','*-'*6)
for i in net.named_parameters():
    print(f'i:{i}')

print('='*6,'net.named_children()','='*6)
for i in net.named_children():
    print(f'i:{i}')
print('*' * 6,'综合嵌套信息','*' * 6)
for i,j in zip(net.parameters(),net.named_children()):
    print(f'i:{i}')
    print(f'j:{j}')

