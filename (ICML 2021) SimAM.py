# ---------------------------------------
# Simam: A simple, parameter-free attention module for convolutional neural networks (ICML 2021)
# Github:https://github.com/ZjjConan/SimAM
# 微信公众号：AI缝合术
"""
2024年全网最全即插即用模块,全部免费!包含各种卷积变种、最新注意力机制、特征融合模块、上下采样模块，
适用于人工智能(AI)、深度学习、计算机视觉(CV)领域，适用于图像分类、目标检测、实例分割、语义分割、
单目标跟踪(SOT)、多目标跟踪(MOT)、红外与可见光图像融合跟踪(RGBT)、图像去噪、去雨、去雾、去模糊、超分等任务，
模块库持续更新中......
https://github.com/AIFengheshu/Plug-play-modules
"""
# ---------------------------------------
import torch
import torch.nn as nn
from thop import profile


class Simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(Simam_module, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.act(y)


# 无参注意力机制    输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    model = Simam_module().cuda()
    x = torch.randn(1, 3, 64, 64).cuda()
    y = model(x)
    print(y.size())
    flops, params = profile(model, (x,))
    print(flops / 1e9)
    print(params)
