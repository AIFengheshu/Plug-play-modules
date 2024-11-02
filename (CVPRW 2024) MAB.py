import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文题目：Multi-scale Attention Network for Single Image Super-Resolution
# 中文题目:  单图像超分辨率的多尺度注意力网络
# 论文链接：https://arxiv.org/abs/2209.14145
# 官方github：https://github.com/icandle/MAN
# 代码整理与注释：公众号：AI缝合术
# AI缝合术github：https://github.com/AIFengheshu/Plug-play-modules

# 自定义的LayerNorm层，用于对输入数据进行标准化
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        # 创建权重参数，初始化为1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # 创建偏置参数，初始化为0
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        # 判断data_format是否符合预期
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        # 设置标准化形状
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # 如果数据格式是"channels_last"，直接使用layer_norm
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # 如果是"channels_first"，手动计算均值和方差
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# Gated Spatial Attention Unit (GSAU)，用于空间注意力机制
class GSAU(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2

        # 第一个1x1卷积层，用于特征扩展
        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        # 深度可分离卷积层，用于提取空间信息
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        # 第二个1x1卷积层
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        # 标准化层
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        # 可学习的尺度参数
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        # 保留输入的副本，作为残差连接
        shortcut = x.clone()

        # 经过标准化和第一个卷积层
        x = self.Conv1(self.norm(x))
        # 将通道分为两部分，a用于门控操作，x继续处理
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)  # 空间注意力机制
        x = self.Conv2(x)  # 再次卷积

        # 加上残差连接并返回
        return x * self.scale + shortcut

# Multi-scale Large Kernel Attention (MLKA)，多尺度大核卷积注意力机制
class MLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        if n_feats % 3 != 0:
            raise ValueError("n_feats必须能被3整除，才能用于MLKA")

        i_feats = 2 * n_feats

        # 标准化层
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        # 可学习的尺度参数
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        # 三个不同尺度的卷积核分别构建7x7、5x5和3x3的LKA模块
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats // 3, dilation=4),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // 3, dilation=3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, stride=1, padding=(5 // 2) * 2, groups=n_feats // 3, dilation=2),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))

        # 三个不同尺度的额外卷积层
        self.X3 = nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3)
        self.X5 = nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3)
        self.X7 = nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3)

        # 第一个1x1卷积层用于通道扩展
        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        # 最后的1x1卷积层用于恢复通道数量
        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x):
        # 保存输入作为残差连接
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        # 将特征分为a和x两部分
        a, x = torch.chunk(x, 2, dim=1)
        # a的三部分通过不同尺度的卷积操作
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)
        a = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3)],
                      dim=1)
        # 合并后的结果再经过最后的卷积层
        x = self.proj_last(x * a) * self.scale + shortcut
        return x

# Multi-scale Attention Blocks (MAB)，多尺度注意力块
class MAB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        # 大核卷积注意力模块
        self.LKA = MLKA(n_feats)
        # 门控空间注意力单元
        self.LFE = GSAU(n_feats)

    def forward(self, x):
        # 先通过大核卷积注意力
        x = self.LKA(x)
        # 然后通过门控空间注意力
        x = self.LFE(x)
        return x


if __name__ == '__main__':
    #通道数需要被3整除
    n_feats = 3 
    mab = MAB(3)
    input = torch.randn(1, 3, 256, 256)
    output = mab(input)
    print(input.size())
    print(output.size())