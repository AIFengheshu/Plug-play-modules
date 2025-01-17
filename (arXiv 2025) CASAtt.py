import torch
import torch.nn as nn

# 论文题目：CAS-ViT: Convolutional Additive Self-attention Vision Transformers for Efficient Mobile Applications
# 中文题目：CAS-ViT：用于高效移动应用的卷积加性自注意力视觉Transformer
# 论文链接：https://arxiv.org/pdf/2408.03703?
# 官方github：https://github.com/Tianfang-Zhang/CAS-ViT
# 所属机构：商汤科技研究院，清华大学自动化系，华盛顿大学电气与计算机工程系，哥本哈根大学计算机科学系
# 代码整理：微信公众号《AI缝合术》
# 全部即插即用模块代码：https://github.com/AIFengheshu/Plug-play-modules

class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)
    
class CASAtt(nn.Module):

    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out


if __name__ == "__main__":

    # 将模块移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建测试输入张量 (batch_size, channels, height, width)
    x = torch.randn(1, 32, 256, 256).to(device)

    # 初始化 casatt 模块
    casatt=CASAtt(dim=32)
    print(casatt)
    print("微信公众号:AI缝合术")
    casatt = casatt.to(device)

    # 前向传播
    output = casatt(x)
    
    # 打印输入和输出张量的形状
    print("输入张量形状:", x.shape)
    print("输出张量形状:", output.shape)
