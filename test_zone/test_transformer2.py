# 使用huggingface的transformer库
from transformers import Decoder
decoder = Decoder(num_layers=6, d_model=512, num_heads=8, d_ff=2048)
print(f'decoder:{decoder}')