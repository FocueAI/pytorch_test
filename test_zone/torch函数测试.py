import numpy as np
import torch

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



x = torch.randn(5, 5)
triu = torch.triu(x, diagonal=0)
print(f'x:{x}')
print(f'triu:{triu}')

tril = torch.tril(x,diagonal=0)
print(f'tril:{tril}')