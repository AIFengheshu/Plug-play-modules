# 论文题目：InceptionNeXt: When Inception Meets ConvNeXt
# 论文链接：https://arxiv.org/pdf/2303.16900
# 官方github：https://github.com/sail-sg/inceptionnext
# 代码块github：https://github.com/AIFengheshu/Plug-play-modules/blob/main/(CVPR%202024)%20IDC.py

# 代码整理与注释：公众号：AI缝合术
# AI缝合术github：https://github.com/AIFengheshu/Plug-play-modules

import torch
import torch.nn as nn

class InceptionDWConv2d(nn.Module):
    """ 
    Inception深度可分离卷积模块
    这个模块使用多个分支来对输入进行不同类型的卷积操作，类似于Inception模块的结构。
    每个分支使用深度卷积（depthwise convolution），以便减少计算量。
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        # 根据branch_ratio计算每个卷积分支的输出通道数
        gc = int(in_channels * branch_ratio)  # branch的通道数
        # 定义三个分支的卷积操作
        # 1. square_kernel_size 为正方形卷积核大小的深度卷积（3x3）
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 2. band_kernel_size 为带状卷积核的深度卷积（横向卷积：1x11，纵向卷积：11x1）
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        # 计算每个分支的输入输出通道数：idc (原始通道数 - 3个分支的通道数)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        """
        前向传播：对输入x进行拆分并通过各分支卷积处理，最后合并结果
        """
        # 根据计算的split_indexes拆分输入x
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        
        # 将原始通道x_id与经过3个卷积分支处理后的x_hw、x_w、x_h合并
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,  # 在通道维度上拼接
        )

# 测试部分
if __name__ == '__main__':
    # 创建一个InceptionDWConv2d模块实例，输入通道数为64
    idc = InceptionDWConv2d(64)  # 输入通道数64
    # 随机生成一个大小为(1, 64, 256, 256)的输入张量，形状为[B, C, H, W]
    input = torch.randn(1, 64, 256, 256)  # 输入形状为(1, 64, 256, 256)
    # 通过InceptionDWConv2d模块处理输入
    output = idc(input)
    # 打印输入和输出张量的形状
    print(input.size())  # 输出输入张量的形状
    print(output.size())  # 输出输出张量的形状
