
import torch
import torch.nn.functional as F

def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    ##value:                 shape=[bs=2, seq_len=10428, n_heads=8, d_k=32]
    ##value_spatial_shapes:  shape=[2,2] ---> value=[[76,103],[38,52],...] 第一张特征图的h=76,w=103. 第二张特征图的h=38,w=52 ....
    ##sampling_locations:    shape=[bs=2,seq_len=10428/cross-attn=300,n_heads=8,特征图数=4,参考点数=4,坐标点xy=2]
    ##attention_weights:     shape=[bs=2,seq_len=10428/cross-attn=300,n_heads=8,特征图数=2,参考点数=2]

    # for debug and test only,
    # need to use cuda version instead
    batch_size, seq_len, n_heads, d_k = value.shape
    _, Lq_, n_heads, level, sp_num, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],dim=1)  # [value1=[bs=2, h1*w1=76*103=7828, n_heads=8, d_k=32],  --->第1张特征图 =======>下面重点分析该特征图
    sampling_grids = 2 * sampling_locations - 1  # value1=[bs=2,  h2*w2=38*52=1976, n_heads=8, d_k=32]   --->第2张特征图
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # batch_size, H_*W_, n_heads, d_k -> batch_size, H_*W_, n_heads*d_k -> batch_size, n_heads*d_k, H_*W_ -> [batch_size*n_heads, d_k, H_, W_] #### [bs=1, hn*wn, 多头数=2, 隐变量的维度=2]-->[bs=1, hn*wn, 多头数*隐变量的维度=4]-->[bs=1, 多头数*隐变量的维度=4,hn*wn]=>[bs*多头数=2,  隐变量的维度=2,  hn=6,    wn=4]
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(batch_size * n_heads, d_k, H_,
                                                                       W_)  # shape=[batch_size*n_heads=16, d_k=32, H_=76, W_=103]
        # batch_size, Lq_, n_heads, sp_num, 2 -> batch_size, n_heads, Lq_, sp_num, 2 -> [batch_size*n_heads, Lq_, sp_num, 2]   #### [bs=1,seq_len=2,n_heads=2,参考点数=2,坐标点xy=2]-->[bs=1, n_heads=2, seq_len=2, 参考点数=2, 坐标点xy=2]--------------------------->[bs*n_heads=2, seq_len=2, 参考点数=2, 坐标点xy=2]
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0,1)  # shape=[bs*n_heads=16,seq_len=10428 ,参考点数=4,坐标点xy=2]
        # batch_size*n_heads, d_k, Lq_, sp_num
        sampling_value_l_ = F.grid_sample(value_l_,  # -->  shape=[batch_size*n_heads=16, d_k=32, H_=76, W_=103]
                                          sampling_grid_l_,# -->  shape=[batch_size*n_heads=16, seq_len=10428 ,参考点数=4,坐标点xy=2]
                                          mode='bilinear',
                                          padding_mode='zeros',
                                          align_corners=False)  # --->[16,d_k=32,10428,4]
        sampling_value_list.append(sampling_value_l_) # [[batch_size*n_heads=16,d_k=32,10428,4],[16,32,10428,4],[16,32,10428,4][16,32,10428,4]]
    # (batch_size, Lq_, n_heads, level, sp_num) -> (batch_size, n_heads, Lq_, level, sp_num) -> (batch_size*n_heads, 1, Lq_, level*sp_num)
    attention_weights = attention_weights.transpose(1, 2).reshape(batch_size * n_heads, 1, Lq_, level * sp_num) # [bs*n_heads=16, 1, 10428, 4*4=16]
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(batch_size,n_heads * d_k, Lq_)
    #                  [batch_size*n_heads=16,d_k=32,10428,4*4=16] *
    #                  [batch_size*n_heads=16, 1,    10428,4*4=16] ---sum(-1).shape=[16,32,10428] ---->最后---->[bs=2,n_heads=8*d_k=32 =256, 10428]
    return output.transpose(1,2).contiguous()  # shape=[batch_size, Lq_=10428/300, n_heads*d_k]

output_pytorch = ms_deform_attn_core_pytorch(value=torch.randn(size=(2,10428,8,32)),
                                             value_spatial_shapes=torch.tensor([[76,103],[38,52],[19,26],[10,13]]),
                                             sampling_locations=torch.randn(size=[2,10428,8,4,4,2]),
                                             attention_weights=torch.randn(size=[2,10428,8,4,4])
                                             ).detach().cpu()
print(f'output_pytorch.shape:{output_pytorch.shape}')


