import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum

# 论文题目：Rethinking Transformer-Based Blind-Spot Network for Self-Supervised Image Denoising
# 中文题目：基于Transformer的盲点网络自监督图像去噪的再思考
# 论文链接：https://arxiv.org/pdf/2404.07846
# 官方github：https://github.com/nagejacob/TBSN
# 所属机构：哈尔滨工业大学
# 代码整理：微信公众号《AI缝合术》
   
class GCSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(GCSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out
    
if __name__ == "__main__":

    # 将模块移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建测试输入张量 (batch_size, channels, height, width)
    x = torch.randn(1, 32, 256, 256).to(device)

    # 初始化 GCSA 模块
    gcsa = GCSA(dim=32, num_heads=4, bias=True)
    print(gcsa)
    gcsa = gcsa.to(device)

    # 前向传播
    output = gcsa(x)

    # 打印输入和输出张量的形状
    print("输入张量形状:", x.shape)
    print("输出张量形状:", output.shape)
