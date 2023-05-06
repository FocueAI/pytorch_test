import torch
import torch.nn as nn


def transformer_decoder_sig():
    decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    memory = torch.rand(10, 32, 512)
    tgt = torch.rand(20, 32, 512)

    out = transformer_decoder(tgt, memory)
    print(f'out.shape:{out.shape}')


def transformer_encoder_sig():
    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    src = torch.rand(10, 32, 512)

    out = transformer_encoder(src)
    print(f'out.shape:{out.shape}')


def transformer_encoder_decoder():
    # 编码器
    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    # 解码器
    decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    src = torch.rand(10, 32, 512)  # 10: 序列长度, 32: batch_size, 512: d_model(transformer模型的维度)
    tgt = torch.rand(20, 32, 512)

    memory = transformer_encoder(src)
    out = transformer_decoder(tgt, memory)
    print(f'out.shape:{out.shape}')


if __name__ == '__main__':
    transformer_encoder_decoder()
