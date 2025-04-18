import torch
import torch.nn as nn
from einops import rearrange
import math

# 论文：Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration, CVPR 2024.
# 论文地址：https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf
# 微信公众号：AI缝合术
"""
2024年全网最全即插即用模块,全部免费!包含各种卷积变种、最新注意力机制、特征融合模块、上下采样模块，
适用于人工智能(AI)、深度学习、计算机视觉(CV)领域，适用于图像分类、目标检测、实例分割、语义分割、
单目标跟踪(SOT)、多目标跟踪(MOT)、红外与可见光图像融合跟踪(RGBT)、图像去噪、去雨、去雾、去模糊、超分等任务，
模块库持续更新中......
https://github.com/AIFengheshu/Plug-play-modules
"""
class FRFN(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        x_init = x
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)

        x1, x2, = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear1(x)
        # gate mechanism
        x_1, x_2 = x.chunk(2, dim=-1)

        x_1 = rearrange(x_1, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        x_1 = self.dwconv(x_1)
        x_1 = rearrange(x_1, ' b c h w -> b (h w) c', h=hh, w=hh)
        x = x_1 * x_2

        x = self.linear2(x)

        return x + x_init


if __name__ == '__main__':
    # Instantiate the FRFN class
    dim = 64  # Dimension of input features


    # Create an instance of the FRFN module
    frfn = FRFN(dim)

    # Generate a random input tensor
    B = 1  # Batch size
    H = 64  # Height of the feature map
    W = 64  # Width of the feature map
    C = dim  # Number of channels

    input = torch.randn(B, H * W, C)

    # Forward pass
    output = frfn(input)

    # Print input and output shapes
    print(input.size())
    print(output.size())