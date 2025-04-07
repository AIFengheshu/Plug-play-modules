import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward  # 导入离散小波变换模块

# 运行报错ModuleNotFoundError: No module named 'pytorch_wavelets'
# 说明缺少pytorch_wavelets包, 用 pip install pytorch-wavelets 即可安装

# GitHub地址 ：https://github.com/apple1986/HWD
# 论文地址：https://www.sciencedirect.com/science/article/pii/S0031320323005174
# 微信公众号：AI缝合术

"""

2024年全网最全即插即用模块,全部免费!包含各种卷积变种、最新注意力机制、特征融合模块、上下采样模块，
适用于人工智能(AI)、深度学习、计算机视觉(CV)领域，适用于图像分类、目标检测、实例分割、语义分割、
单目标跟踪(SOT)、多目标跟踪(MOT)、红外与可见光图像融合跟踪(RGBT)、图像去噪、去雨、去雾、去模糊、超分等任务，
模块库持续更新中......
https://github.com/AIFengheshu/Plug-play-modules

"""

class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        # 初始化离散小波变换，J=1表示变换的层数，mode='zero'表示填充模式，使用'Haar'小波
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        # 定义卷积、批归一化和ReLU激活的顺序组合
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),  # 1x1卷积层，通道数由in_ch*4变为out_ch
            nn.BatchNorm2d(out_ch),  # 批归一化层
            nn.ReLU(inplace=True),  # ReLU激活函数
        )

    def forward(self, x):
        # 对输入x进行离散小波变换，得到低频部分yL和高频部分yH
        yL, yH = self.wt(x)
        # 提取高频部分的不同分量
        y_HL = yH[0][:, :, 0, ::]  # 水平高频
        y_LH = yH[0][:, :, 1, ::]  # 垂直高频
        y_HH = yH[0][:, :, 2, ::]  # 对角高频
        # 将低频部分和高频部分拼接在一起
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        # 通过卷积、批归一化和ReLU激活处理拼接后的特征
        x = self.conv_bn_relu(x)
        return x

if __name__ == '__main__':
    block = Down_wt(64, 64)  # 创建 Down_wt 模块，输入和输出通道数均为64
    input = torch.rand(3, 64, 64, 64)  # 创建输入张量，形状为 (B, C, H, W)
    output = block(input)  # 通过模块处理输入
    print(input.size())  # 打印输入的尺寸
    print(output.size())  # 打印输出的尺寸
