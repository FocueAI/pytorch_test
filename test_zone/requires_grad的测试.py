import torch

x = torch.randn(1, 2)
print(x.requires_grad) # False-可见默认建立的tensor是不可求导,不可反向传播的
print(x)

