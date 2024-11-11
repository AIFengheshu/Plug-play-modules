# 论文题目：SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution
# 论文链接：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06713.pdf
# 官方github：https://github.com/Zheng-MJ/SMFANet

# 代码整理与注释：公众号：AI缝合术
# AI缝合术github：https://github.com/AIFengheshu/Plug-play-modules
# https://github.com/AIFengheshu/Plug-play-modules/edit/main/(ECCV%202024)%20SMFA.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim,hidden_dim,3,1,1,groups=dim),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0)
        )
        self.act =nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x
    
class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.lde = DMlp(dim, 2)

        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w), mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)

def main():
    # 设置输入张量的尺寸
    batch_size = 1
    dim = 32  # 通道数
    height = 64  # 高度
    width = 64   # 宽度
    # 创建一个随机的输入张量
    input_tensor = torch.randn(batch_size, dim, height, width)
    # 初始化 SMFA 模块
    smfa = SMFA(dim=dim)
    # 前向传播
    output_tensor = smfa(input_tensor)
    # 输出结果的形状
    print(f"输入张量的形状: {input_tensor.shape}")
    print(f"输出张量的形状: {output_tensor.shape}")

if __name__ == "__main__":
    main()
