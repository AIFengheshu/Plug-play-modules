import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文题目：BAM: Bottleneck Attention Module
# 中文题目:  BAM：瓶颈注意力模块
# 论文链接：http://bmvc2018.org/contents/papers/0092.pdf
# 官方github：https://github.com/Jongchan/attention-module
# 所属机构：Lunit Inc., 韩国; 韩国科学技术院(KAIST), 韩国; Adobe Research, 美国
# 关键词： 瓶颈注意力模块, 深度神经网络, 注意力机制, 图像分类, 目标检测
# 微信公众号：AI缝合术

class Flatten(nn.Module):
    def forward(self,x):
        return x.view(x.shape[0],-1)

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        gate_channels = [channel]
        gate_channels += [channel // reduction] * num_layers
        gate_channels += [channel]

        self.ca = nn.Sequential()
        self.ca.add_module('flatten', Flatten())
        for i in range(len(gate_channels) - 2):
            self.ca.add_module('fc%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.ca.add_module('relu%d' % i, nn.ReLU())
        self.ca.add_module('last_fc', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        res = self.avgpool(x)
        res = self.ca(res)
        res = res.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return res

class SpatialAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3, dia_val=2):
        super().__init__()
        self.sa = nn.Sequential()
        self.sa.add_module('conv_reduce1', nn.Conv2d(kernel_size=1, in_channels=channel, out_channels=channel // reduction))
        self.sa.add_module('bn_reduce1', nn.BatchNorm2d(channel // reduction))
        self.sa.add_module('relu_reduce1', nn.ReLU())
        for i in range(num_layers):
            self.sa.add_module('conv_%d' % i, nn.Conv2d(
                kernel_size=3,
                in_channels=channel // reduction,
                out_channels=channel // reduction,
                padding=1,
                dilation=dia_val
            ))
            self.sa.add_module('bn_%d' % i, nn.BatchNorm2d(channel // reduction))
            self.sa.add_module('relu_%d' % i, nn.ReLU())
        self.sa.add_module('last_conv', nn.Conv2d(channel // reduction, 1, kernel_size=1))

    def forward(self, x):
        res = self.sa(x)
        # 使用插值调整大小
        res = F.interpolate(res, size=x.shape[2:], mode='bilinear', align_corners=False)
        res = res.expand_as(x)
        return res

class BAMBlock(nn.Module):
    def __init__(self, channel=512,reduction=16,dia_val=2):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(channel=channel,reduction=reduction,dia_val=dia_val)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        sa_out=self.sa(x)
        ca_out=self.ca(x)
        weight=self.sigmoid(sa_out+ca_out)
        out=(1+weight)*x
        return out

if __name__ == "__main__": 
    bam = BAMBlock(channel=32)
    # 随机生成输入张量 (B, C, H, W)
    input_tensor = torch.rand(1, 32, 256, 256)
    # 打印输入张量的形状
    print(f"输入张量的形状: {input_tensor.shape}")
    # 前向传播
    output_tensor = bam(input_tensor)
    # 打印输出张量的形状
    print(f"输出张量的形状: {output_tensor.shape}")
