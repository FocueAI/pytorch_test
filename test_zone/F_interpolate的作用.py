import torch.nn.functional as F
import torch

a=torch.arange(12,dtype=torch.float32).reshape(1,2,2,3)
b=F.interpolate(a,size= (4,4),mode='bilinear')
print(a)
print(b)
print('a.shape:',a.shape)
print('b.shape:',b.shape)