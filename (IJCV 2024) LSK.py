import torch
import torch.nn as nn

#Github地址：https://github.com/zcablii/Large-Selective-Kernel-Network
#论文地址：https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Large_Selective_Kernel_Network_for_Remote_Sensing_Object_Detection_ICCV_2023_paper.pdf
# 微信公众号：AI缝合术
"""
2024年全网最全即插即用模块,全部免费!包含各种卷积变种、最新注意力机制、特征融合模块、上下采样模块，
适用于人工智能(AI)、深度学习、计算机视觉(CV)领域，适用于图像分类、目标检测、实例分割、语义分割、
单目标跟踪(SOT)、多目标跟踪(MOT)、红外与可见光图像融合跟踪(RGBT)、图像去噪、去雨、去雾、去模糊、超分等任务，
模块库持续更新中......
https://github.com/AIFengheshu/Plug-play-modules
"""

class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 定义各个卷积层
        # 深度可分离卷积，保持输入和输出通道数一致，卷积核大小为5
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # 空间卷积，卷积核大小为7，膨胀率为3，增加感受野
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # 1x1卷积，用于降维
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        # 结合平均和最大注意力的卷积
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        # 最后的1x1卷积，将通道数恢复到原始维度
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        # 对输入进行两种不同的卷积操作以生成注意力特征
        attn1 = self.conv0(x)  # 第一个卷积特征
        attn2 = self.conv_spatial(attn1)  # 空间卷积特征

        # 对卷积特征进行1x1卷积以降维
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        # 将两个特征在通道维度上拼接
        attn = torch.cat([attn1, attn2], dim=1)
        # 计算平均注意力特征
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        # 计算最大注意力特征
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        # 拼接平均和最大注意力特征
        agg = torch.cat([avg_attn, max_attn], dim=1)
        # 通过卷积生成注意力权重，并应用sigmoid激活函数
        sig = self.conv_squeeze(agg).sigmoid()
        # 根据注意力权重调整特征
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + \
               attn2 * sig[:, 1, :, :].unsqueeze(1)
        # 最终卷积恢复到原始通道数
        attn = self.conv(attn)
        # 通过注意力特征加权原输入
        return x * attn


if __name__ == '__main__':
    block = LSKblock(64).cuda()  # 实例化LSK模块并转移到GPU
    input = torch.rand(3, 64, 32, 32).cuda()  # 创建一个随机输入张量，形状为 (batch_size, channels, height, width)
    output = block(input)  # 通过模块前向传播获取输出
    print(input.size())  # 打印输入的尺寸
    print(output.size())  # 打印输出的尺寸
