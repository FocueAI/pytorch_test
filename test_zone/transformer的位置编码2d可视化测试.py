import torch.nn as nn
import torch,math
import torch as Tensor

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class PositionalEncoding1D(nn.Module):
    """Classic Attention-is-all-you-need positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = self.make_pe(d_model, max_len)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    @staticmethod
    def make_pe(d_model: int, max_len: int) -> Tensor:  # d_model=256, max_len=2000
        """Compute positional encoding."""
        pe = torch.zeros(max_len, d_model)  # shape:[max_len=2000, d_model=256]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # shape=[2000,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        return pe  # [max_len=2000, 1, d_model=256]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (S, B, d_model)

        Returns:
            (B, d_model, H, W)
        """
        assert x.shape[2] == self.pe.shape[2]  # type: ignore
        x = x + self.pe[: x.size(0)]  # type: ignore
        return self.dropout(x)

class PositionalEncoding2D(nn.Module):
    """2-D positional encodings for the feature maps produced by the encoder.

    Following https://arxiv.org/abs/2103.06450 by Sumeet Singh.

    Reference:
    https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/main/lab9/text_recognizer/models/transformer_util.py
    """

    def __init__(self, d_model: int, max_h: int = 2000, max_w: int = 2000) -> None:
        super().__init__()
        self.d_model = d_model
        assert d_model % 2 == 0, f"Embedding depth {d_model} is not even"
        pe = self.make_pe(d_model, max_h, max_w)  # (d_model, max_h, max_w)
        self.register_buffer("pe", pe)

    @staticmethod
    def make_pe(d_model: int, max_h: int, max_w: int) -> Tensor:
        """Compute positional encoding."""
        pe_h = PositionalEncoding1D.make_pe(d_model=d_model // 2, max_len=max_h)  # (max_h=2000, 1 d_model // 2=256)
        pe_h = pe_h.permute(2, 0, 1).expand(-1, -1, max_w)  # (d_model // 2=256, max_h=2000, max_w=2000)

        pe_w = PositionalEncoding1D.make_pe(d_model=d_model // 2, max_len=max_w)  # (max_w, 1, d_model // 2)
        pe_w = pe_w.permute(2, 1, 0).expand(-1, max_h, -1)  # (d_model // 2=256, max_h=2000, max_w=2000)

        pe = torch.cat([pe_h, pe_w], dim=0)  # (d_model=512, max_h=2000, max_w=2000)
        return pe

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, d_model, H, W)

        Returns:
            (B, d_model, H, W)
        """
        assert x.shape[1] == self.pe.shape[0]  # type: ignore  断定2者的d_model一定相等
        x = x + self.pe[:, : x.size(2), : x.size(3)]  # type: ignore
        return x
if __name__ == '__main__':
    import os,time,cv2
    import numpy as np
    from PIL import Image
    # posencoding = PositionalEncoding1D(d_model=512, max_len=1000)
    # #                   (seq_len=32, batch_size=16, d_model=512)
    # input_x = torch.ones(32, 16, 512)
    # output_y = posencoding(input_x)
    # print(f'output_y:{output_y.shape}')

    # 2D位置编码的可视化
    posencoding = PositionalEncoding2D(d_model=512)
    input_x = torch.zeros(1, 512, 512, 512)
    out_x = posencoding(input_x)
    out_x = out_x.permute(1,0,2,3)
    for i in range(512):
        print(f'out_x.shape:{out_x.shape}')
        out_x_shape = out_x[i][0]

        out_x_shape = out_x_shape.numpy()
        out_x_shape = (out_x_shape*255).astype(np.uint8)
        pil_img = Image.fromarray(out_x_shape)

        cv_img = np.array(pil_img)
        cv2.imshow('pos-encoding',cv_img)
        cv2.waitKey(3)