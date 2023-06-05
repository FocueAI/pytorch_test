import torch
import torch.nn.functional as F

input = torch.randn(1, 2, 4, 4)
grid = torch.tensor([
                      [ [[0.5, 0.5], [0.5, 0.5]],
                        [[0.5, 0.5], [0.5, 0.5]]
                      ]
                    ])
output = F.grid_sample(input, grid)
print(f'input.shape:{input.shape},grid.shape:{grid.shape},output.shape:{output.shape}')
print(output)

