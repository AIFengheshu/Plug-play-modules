from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath


class GatedCNNBlock(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args: 
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve practical efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and 
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm,eps=1e-6), 
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        return x + shortcut
    
if __name__ == "__main__":
    batch_size = 1  # Batch size
    channels = 32   # 输入通道数
    height = 256    # 输入图像高度
    width = 256     # 输入图像宽度

    # 创建一个模拟输入张量，形状为 (batch_size, height, width, channels)
    x = torch.randn(batch_size, height, width, channels)

    # 初始化 GatedCNNBlock 模块
    model = GatedCNNBlock(dim=channels, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0, drop_path=0.1)
    print(model)
    print("微信公众号: AI缝合术!")
    # 前向传播
    output = model(x)

    # 打印输入和输出张量的形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


