from torch import nn
import torch

num_layer = 6

decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layer)
memory = torch.rand(10, 32, 512)
tgt = torch.rand(20, 32, 512)
out = transformer_decoder(tgt, memory)

print(f'out.shape:{out.shape}') # torch.Size([20, 32, 512]) 感觉这就是最后一层的decode的输出


for i in range(num_layer):
    tgt = decoder_layer(tgt, memory)
    print(f'i:{i}, tgt.shape:{tgt.shape}')