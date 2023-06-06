import torch
#                                 a1                    a2                    a3
out_bbox = torch.tensor([[0.5, 0.5, 1.0, 1.0], [0.2, 0.2, 0.8, 0.8], [0.1, 0.1, 0.3, 0.3]])
#                                 b1                    b2                    b3
tgt_bbox = torch.tensor([[0.4, 0.4, 1.0, 1.0], [0.3, 0.3, 0.7, 0.7], [0.2, 0.2, 0.4, 0.4]])

cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

print(f'cost_bbox:{cost_bbox}')