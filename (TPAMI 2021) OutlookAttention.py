# ---------------------------------------
# 论文: VOLO: Vision Outlooker for Visual Recognition (TPAMI 2021)
# 论文链接：https://arxiv.org/pdf/2106.13112
# Github地址: https://github.com/sail-sg/volo
# 微信公众号：AI缝合术
"""
2024年全网最全即插即用模块,全部免费!包含各种卷积变种、最新注意力机制、特征融合模块、上下采样模块，
适用于人工智能(AI)、深度学习、计算机视觉(CV)领域，适用于图像分类、目标检测、实例分割、语义分割、
单目标跟踪(SOT)、多目标跟踪(MOT)、红外与可见光图像融合跟踪(RGBT)、图像去噪、去雨、去雾、去模糊、超分等任务，
模块库持续更新中......
https://github.com/AIFengheshu/Plug-play-modules
"""
import torch
from torch import nn
import math
from torch.nn import functional as F

class OutlookAttention(nn.Module):
    """
    实现 Outlook Attention 模块
    参数说明：
    --dim: 隐藏层维度
    --num_heads: 注意力头的数量
    --kernel_size: 注意力窗口的卷积核大小
    返回值：经过 Outlook Attention 的 token 特征
    """

    def __init__(self, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        # 计算每个注意力头的维度
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        # 若 qk_scale 未指定则使用 head_dim 的倒数的平方根作为缩放因子
        self.scale = qk_scale or head_dim**-0.5

        # 定义用于获取V特征的全连接层
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        # 定义获取注意力权重的全连接层，输出维度与 kernel_size^4 * num_heads 相同
        self.attn = nn.Linear(dim, kernel_size**4 * num_heads)

        # 定义注意力和投影的dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 定义用于展开输入的 Unfold 操作
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        # 定义用于池化的平均池化层
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        # 获取输入的 batch 大小和特征图的高、宽及通道数
        B, H, W, C = x.shape

        # 通过全连接层获取V特征，并调整形状为 (B, C, H, W)
        v = self.v(x).permute(0, 3, 1, 2)

        # 计算经过stride后的特征图的高、宽
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        # 对V特征进行unfold展开，并调整形状为 (B, num_heads, C//num_heads, kernel_size*kernel_size, h*w)
        v = self.unfold(v).reshape(B, self.num_heads, C // self.num_heads,
                                   self.kernel_size * self.kernel_size,
                                   h * w).permute(0, 1, 4, 3, 2)

        # 对输入 x 进行池化后计算注意力权重，调整形状为 (B, num_heads, h*w, kernel_size*kernel_size, kernel_size*kernel_size)
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
            self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        # 缩放注意力权重
        attn = attn * self.scale
        # 进行 softmax 激活，使权重在最后一个维度上归一化
        attn = attn.softmax(dim=-1)
        # 应用dropout
        attn = self.attn_drop(attn)

        # 对展开后的V特征加权，恢复原始维度
        x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, C * self.kernel_size * self.kernel_size, h * w)
        # 使用 fold 恢复特征图的空间大小
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride)

        # 通过全连接层投影输出并应用 dropout
        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)

        return x

# 输入 B, H, W, C,  输出 B, H, W, C
if __name__ == '__main__':
    block = OutlookAttention(dim=32, num_heads=8).cuda()
    input = torch.rand(3, 64, 64, 32).cuda()
    output = block(input)
    print(input.size())  # 打印输入的尺寸
    print(output.size())  # 打印输出的尺寸
