import torch
import torch.nn as nn

# 论文题目：SCConv: Spatial and Channel Reconstruction Convolution for Feature Redundancy
# 中文题目:  SCConv: 用于特征冗余的空间和通道重建卷积
# 论文链接：https://arxiv.org/pdf/2012.11879https://openaccess.thecvf.com/content/CVPR2023/papers/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.pdf
# 官方github：无
# 所属机构：华东师范大学，同济大学
# 关键词：卷积神经网络，特征冗余，空间重建单元，通道重建单元，模型压缩

class SRU(nn.Module):
    """
    空间重建单元（Spatial Reconstruction Unit）

    减少空间维度中的冗余

    主要部分：
    1. 分离（Separate）
        - 从空间内容中分离出有信息的特征图和信息较少的特征图，
        这样我们可以重建低冗余的特征。

    2. 重建（Reconstruction）
        - 在不同通道之间（有信息的通道和信息较少的通道）进行交互，
        加强这些通道之间的信息流动，因此可能提高准确度，
        减少冗余特征并增强CNN的特征表示能力。
    """

    def __init__(
            self,
            channels: int,
            group_num: int = 4,
            gate_threshold: float = 0.5,
    ):
        super(SRU, self).__init__()
        self.gn = nn.GroupNorm(group_num, channels)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        # 从空间内容中分离出有信息的特征图和信息较少的特征图
        gn_x = self.gn(x)
        # 使用 gn.weight 测量每个批次和通道的空间像素方差
        w = (self.gn.weight / torch.sum(self.gn.weight)).view(1, -1, 1, 1)
        w = self.sigmoid(w * gn_x)
        infor_mask = w >= self.gate_threshold
        less_infor_maks = w < self.gate_threshold
        x1 = infor_mask * gn_x
        x2 = less_infor_maks * gn_x

        # 使用有信息的特征和信息较少的特征重建特征
        x11, x12 = torch.split(x1, x1.size(1) // 2, dim=1)
        x21, x22 = torch.split(x2, x2.size(1) // 2, dim=1)
        out = torch.cat([x11 + x22, x12 + x21], dim=1)
        return out


class CRU(nn.Module):
    """
    空间重建单元（Spatial Reconstruction Unit）

    CRU 通过轻量级卷积操作提取丰富的代表性特征，
    同时通过廉价操作和特征重用方案处理冗余特征。

    主要部分：
    1. 分割（Split）
        - 分割和压缩，将空间特征划分为 Xup（上层转换阶段）
          和 Xlow（下层转换阶段）
        Xup 用作 '丰富特征提取器'。
        Xlow 用作 '细节信息补充'

        Xup 使用 GWC（组卷积）和 PWC（点卷积）替代昂贵的标准 k x k
        卷积来提取高级代表性信息并减少计算成本。

        GWC 可以减少参数和计算量，但会切断通道组之间的信息流动，
        因此另一条路径使用 PWC 来帮助信息流跨通道流动，然后将 GWC 和 PWC 的输出相加
        形成 Y2，用于提取丰富的代表性信息。

        Xlow 重用前馈特征图并利用 1x1 的 PWC 作为对丰富特征提取器的补充，
        然后将它们连接起来形成 Y2。

    2. 融合（Fuse）
        - 类似 SKNet，使用 GAP（全局平均池化）和通道维度上的 Soft-Attention 来重构
        新的特征。
    """

    def __init__(
            self,
            channels: int,
            alpha: float = 0.5,
            squeeze_ratio: int = 2,
            groups: int = 2,
            stride: int = 1,
    ):
        super(CRU, self).__init__()
        self.upper_channel = int(channels * alpha)
        self.low_channel = channels - self.upper_channel
        s_up_c, s_low_c = self.upper_channel // squeeze_ratio, self.low_channel // squeeze_ratio
        self.squeeze_up = nn.Conv2d(self.upper_channel, s_up_c, 1, stride=stride, bias=False)
        self.squeeze_low = nn.Conv2d(self.low_channel, s_low_c, 1, stride=stride, bias=False)

        # 上层 -> GWC + PWC
        self.gwc = nn.Conv2d(s_up_c, channels, 3, stride=1, padding=1, groups=groups)
        self.pwc1 = nn.Conv2d(s_up_c, channels, 1, bias=False)

        # 下层 -> 连接（前馈特征，PWC）
        self.pwc2 = nn.Conv2d(s_low_c, channels - s_low_c, 1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        up, low = torch.split(x, [self.upper_channel, self.low_channel], dim=1)
        up, low = self.squeeze_up(up), self.squeeze_low(low)

        # 上层 -> GWC + PWC
        y1 = self.gwc(up) + self.pwc1(up)
        # 下层 -> 连接（前馈特征，PWC）
        y2 = torch.cat((low, self.pwc2(low)), dim=1)

        out = torch.cat((y1, y2), dim=1)
        # 增强包含大量信息的特征图
        out_s = self.softmax(self.gap(out))
        out = out * out_s
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        # 减少冗余信息
        return out1 + out2

class SC(nn.Module):
    def __init__(
            self,
            channels: int,
            group_num: int = 4,
            gate_threshold: int = 0.5,
            alpha: float = 0.5,
            squeeze_ratio: int = 2,
            groups: int = 2,
            stride: int = 1,
    ):
        super(SC, self).__init__()
        self.sru = SRU(channels, group_num, gate_threshold)
        self.cru = CRU(channels, alpha, squeeze_ratio, groups, stride)

    def forward(self, x: torch.Tensor):
        x = self.sru(x)
        x = self.cru(x)
        return x

if __name__ == '__main__':
    x       = torch.randn(1,32,16,16)
    model   = SC(32)
    print(model(x).shape)
