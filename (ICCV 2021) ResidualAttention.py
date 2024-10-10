import numpy as np
import torch
from torch import nn
from torch.nn import init

# 论文地址：https://arxiv.org/pdf/2108.02456
# 论文：Residual Attention: A Simple but Effective Method for Multi-Label Recognition
# 微信公众号：AI缝合术
"""
2024年全网最全即插即用模块,全部免费!包含各种卷积变种、最新注意力机制、特征融合模块、上下采样模块，
适用于人工智能(AI)、深度学习、计算机视觉(CV)领域，适用于图像分类、目标检测、实例分割、语义分割、
单目标跟踪(SOT)、多目标跟踪(MOT)、红外与可见光图像融合跟踪(RGBT)、图像去噪、去雨、去雾、去模糊、超分等任务，
模块库持续更新中......
https://github.com/AIFengheshu/Plug-play-modules
"""

class ResidualAttention(nn.Module):

    def __init__(self, channel=512 , num_class=1000,la=0.2):
        super().__init__()
        self.la=la
        self.fc=nn.Conv2d(in_channels=channel,out_channels=num_class,kernel_size=1,stride=1,bias=False)

    def forward(self, x):
        b,c,h,w=x.shape
        y_raw=self.fc(x).flatten(2) #b,num_class,hxw
        y_avg=torch.mean(y_raw,dim=2) #b,num_class
        y_max=torch.max(y_raw,dim=2)[0] #b,num_class
        score=y_avg+self.la*y_max
        return score

        


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    resatt = ResidualAttention(channel=512,num_class=1000,la=0.2)
    output=resatt(input)
    print(output.shape)

    