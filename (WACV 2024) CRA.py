import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import math

# 论文题目：MetaSeg: MetaFormer-based Global Contexts-aware Network for Efficient Semantic Segmentation
# 中文题目:  MetaSeg: 基于MetaFormer的全局上下文感知网络，用于高效的语义分割
# 论文链接：http://bmvc2018.org/contents/papers/0092.pdfhttps://openaccess.thecvf.com/content/WACV2024/papers/Kang_MetaSeg_MetaFormer-Based_Global_Contexts-Aware_Network_for_Efficient_Semantic_Segmentation_WACV_2024_paper.pdf
# 官方github：https://github.com/hyunwoo137/MetaSeg
# 所属机构：韩国首尔高丽大学
# 关键词：MetaFormer, 语义分割, 全局上下文, 自注意力, 计算效率

# 公众号：AI缝合术
class ChannelReductionAttention(nn.Module):
    def __init__(self, dim1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        # self.dim2 = dim2
        self.pool_ratio = pool_ratio
        self.num_heads = num_heads
        head_dim = dim1 // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.k = nn.Linear(dim1, self.num_heads, bias=qkv_bias)  
        self.v = nn.Linear(dim1, dim1, bias=qkv_bias)  
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)  
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
        self.sr = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim1)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, h, w):
        B, N, C = x.shape  
        q = self.q(x).reshape(B, N, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        x_ = x.permute(0, 2, 1).reshape(B, C, h, w)
        x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.norm(x_)
        x_ = self.act(x_)

        k = self.k(x_).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def test_channel_reduction_attention():

    h, w = 64, 64   # 输入图像的高和宽
    input_tensor = torch.randn(1, 64, h * w)  # 模拟输入张量，形状为 (batch_size, dim1, N)
    input_tensor = input_tensor.permute(0, 2, 1) # 转换输入形状 (B, N, C)
    
    # 初始化 ChannelReductionAttention 模块
    cra = ChannelReductionAttention(dim1=64, num_heads=8, pool_ratio=16)
    
    # 执行前向传播
    output = cra(input_tensor, h, w)
    
    # 输出结果的形状
    print(f"输入张量的形状: {input_tensor.shape}")
    print(f"输出张量的形状: {output.shape}")

if __name__ == "__main__":
    test_channel_reduction_attention()
