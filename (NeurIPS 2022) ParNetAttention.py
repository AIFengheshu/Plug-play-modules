import numpy as np
import torch
from torch import nn
from torch.nn import init

# 论文地址：https://arxiv.org/pdf/2110.07641
# 论文：Non-deep Networks
# 微信公众号：AI缝合术
"""
2024年全网最全即插即用模块,全部免费!包含各种卷积变种、最新注意力机制、特征融合模块、上下采样模块，
适用于人工智能(AI)、深度学习、计算机视觉(CV)领域，适用于图像分类、目标检测、实例分割、语义分割、
单目标跟踪(SOT)、多目标跟踪(MOT)、红外与可见光图像融合跟踪(RGBT)、图像去噪、去雨、去雾、去模糊、超分等任务，
模块库持续更新中......
https://github.com/AIFengheshu/Plug-play-modules
"""

class ParNetAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.Sigmoid()
        )

        self.conv1x1=nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.BatchNorm2d(channel)
        )
        self.conv3x3=nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(channel)
        )
        self.silu=nn.SiLU()
        

    def forward(self, x):
        b, c, _, _ = x.size()
        x1=self.conv1x1(x)
        x2=self.conv3x3(x)
        x3=self.sse(x)*x
        y=self.silu(x1+x2+x3)
        return y


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    pna = ParNetAttention(channel=512)
    output=pna(input)
    print(output.shape)

    