"""
使用torchvision集成的现成代码
"""
import torch
from torch import nn
from torchvision.models import resnet18

# print(f'resnet18:{resnet18()}')
# resnet = resnet18()
# backbone = nn.Sequential(
#     resnet.conv1,
#     resnet.bn1,
#     resnet.relu,
#     resnet.maxpool,
#     resnet.layer1,
#     resnet.layer2,
#     # resnet.layer3,
# )
# input_tensor = torch.zeros(8,3,224,224)
# out_tensor = backbone(input_tensor)
# print(f'input_tensor.shape:{input_tensor.shape}')
# print(f'out_tensor.shape:{out_tensor.shape}')

mask = (torch.triu(torch.ones(5, 5)) == 1).transpose(0, 1)
print(mask)
print('='*10)
print(torch.triu(torch.ones(5, 5).transpose(0, 1)))
print('-'*10)
print(torch.triu(torch.ones(5, 5)))