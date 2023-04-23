import torch
import torch.nn as nn
import torch.nn.functional as F

# https://blog.csdn.net/Hushenghan12/article/details/108143816

# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1,padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(12 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.reshape(-1, 12 * 7 * 7)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    model = Net()
    print(model)
    input = torch.randn(1, 1, 24, 24)
    output = model(input)
    print(f'output:{output}')
