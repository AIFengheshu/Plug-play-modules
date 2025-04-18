# 论文题目：SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution
# 论文链接：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06713.pdf
# 官方github：https://github.com/Zheng-MJ/SMFANet

# 代码整理与注释：公众号：AI缝合术
# AI缝合术github：https://github.com/AIFengheshu/Plug-play-modules
# https://github.com/AIFengheshu/Plug-play-modules/edit/main/(ECCV%202024)%20SMFA.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# DMlp 类：一个多层感知机模块，使用卷积层和激活函数进行特征处理
class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)  # 计算隐藏层的维度，growth_rate 控制增长比例
        # 定义卷积层序列，首先对输入做深度卷积，再进行逐点卷积
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),  # 深度卷积，逐通道卷积
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)  # 逐点卷积，通道数不变
        )
        self.act = nn.GELU()  # GELU 激活函数，非线性变换
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)  # 最终卷积，输出通道数为 dim

    def forward(self, x):
        x = self.conv_0(x)  # 通过卷积层 conv_0
        x = self.act(x)  # 激活函数处理
        x = self.conv_1(x)  # 通过 conv_1
        return x  # 返回处理后的特征

# SMFA 类：包含了多种特征融合与处理的操作
class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        # 1x1 卷积，增加通道数，将通道数从 dim 扩展为 dim * 2
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        # 1x1 卷积，保持通道数为 dim
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        # 1x1 卷积，保持通道数为 dim
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        # 引入 DMlp 模块，用于进行特征处理
        self.lde = DMlp(dim, 2)
        # 深度卷积，通道数不变，进行空间卷积处理
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        # GELU 激活函数
        self.gelu = nn.GELU()
        self.down_scale = 8  # 下采样比例因子
        # alpha 和 belt 是可学习的参数，控制模型的加权
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))  # 乘法因子
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))  # 加法因子

    def forward(self, f):
        # 获取输入张量的尺寸：batch_size, channels, height, width
        _, _, h, w = f.shape
        # 将输入通过卷积层 linear_0 分割成 y 和 x 两部分
        y, x = self.linear_0(f).chunk(2, dim=1)
        # 对 x 部分进行下采样，并使用深度卷积处理
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        # 计算 x 的方差，作为额外的统计信息
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        # 对 x 部分做加权和调整，并通过线性卷积进行处理
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w), mode='nearest')
        # 对 y 部分通过 DMlp 进行处理
        y_d = self.lde(y)
        # 将处理后的 x_l 和 y_d 进行加和，并通过 linear_2 进行最终卷积
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
