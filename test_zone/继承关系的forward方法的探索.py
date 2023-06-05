import torch
import torch.nn as nn


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        print('I am in base class __init__')

    def forward(self):
        print('I am in Base-forward')


class Net(Base):
    def __init__(self):
        super(Net, self).__init__()
        print(f'I am in Net __init__')

    def forward(self):
        print('I am in Net forward...')


if __name__ == '__main__':
    net = Net()
    net()
