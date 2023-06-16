import torch
from torch import nn


d_model = 128
nhead = 8
dim_feedforward = 256
dropout = 0.5
num_decoder_layers = 2
nn.Dropout()
nn.Linear
transformer_decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_decoder_layers)
# print(transformer_decoder)
for i in range(num_decoder_layers):
    transformer_decoder_layer()


nn.MultiheadAttention()

# a = torch.tensor([1, 2, 3, 4])
# b = torch.tensor([2, 1, 3, 4])
# res = torch.gt(a,b)
# print(f'res:{res}')
# res:tensor([False,  True, False, False])
