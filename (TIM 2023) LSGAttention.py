import torch
from einops import rearrange, repeat
import torch.nn as nn

# 论文题目：Light Self-Gaussian-Attention Vision Transformer for Hyperspectral Image Classification
# 中文题目：用于高光谱图像分类的轻量自高斯注意力视觉Transformer
# 论文链接：https://ieeexplore.ieee.org/document/10135126
# 官方github：https://github.com/machao132/LSGA-VIT
# 所属机构：南京理工大学电子与光电子工程学院等
# 代码整理：微信公众号《AI缝合术》

class LSGAttention(nn.Module):
    def __init__(self, dim, att_inputsize, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.att_inputsize = att_inputsize[0]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        totalpixel = self.att_inputsize * self.att_inputsize
        gauss_coords_h = torch.arange(totalpixel) - int((totalpixel - 1) / 2)
        gauss_coords_w = torch.arange(totalpixel) - int((totalpixel - 1) / 2)
        gauss_x, gauss_y = torch.meshgrid([gauss_coords_h, gauss_coords_w])
        sigma = 10
        gauss_pos_index = torch.exp(torch.true_divide(-(gauss_x ** 2 + gauss_y ** 2), (2 * sigma ** 2)))
        self.register_buffer("gauss_pos_index", gauss_pos_index)
        self.token_wA = nn.Parameter(torch.empty(1, self.att_inputsize * self.att_inputsize, dim),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, dim, dim),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        wa = repeat(self.token_wA, '() n d -> b n d', b=B_)  # wa (bs 4 64)
        wa = rearrange(wa, 'b h w -> b w h')  # Transpose # wa (bs 64 4)
        A = torch.einsum('bij,bjk->bik', x, wa)  # A (bs 81 4)
        A = rearrange(A, 'b h w -> b w h')  # Transpose # A (bs 4 81)
        A = A.softmax(dim=-1)
        VV = repeat(self.token_wV, '() n d -> b n d', b=B_)  # VV(bs,64,64)
        VV = torch.einsum('bij,bjk->bik', x, VV)  # VV(bs,81,64)
        x = torch.einsum('bij,bjk->bik', A, VV)  # T(bs,4,64)
        absolute_pos_bias = self.gauss_pos_index.unsqueeze(0)
        q = self.qkv(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = x.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn + absolute_pos_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 参数配置
    batch_size = 1
    channels = 32
    height = 16
    width = 16  # 输入特征图的高度和宽度
    # 初始化 LSGAttention 模块
    attention_module = LSGAttention(
        dim=channels,
        att_inputsize=(height, width),
        num_heads=4
    ).to(device)
    # 构造输入张量 (B, N, C)，其中 N 为 height * width
    input_tensor = torch.rand(batch_size, height * width, channels).to(device)
    # 前向传播
    output_tensor = attention_module(input_tensor)
    # 打印输入和输出的形状
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")
