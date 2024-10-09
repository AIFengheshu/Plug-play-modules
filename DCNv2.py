import torch
import torch.nn as nn
import math
from torchvision.ops import deform_conv2d

# 自动填充padding的函数
def autopad(kernel_size, padding):
    # 默认返回的padding让卷积层输入输出大小相同（保持原大小）
    return padding if padding is not None else kernel_size // 2

class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, groups=1, act=True, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (autopad(kernel_size, padding), autopad(kernel_size, padding))
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()



# # 测试代码
# def main():
#     # 随机生成输入张量, 假设batch_size=4, 通道数=3, 高度和宽度=64
#     input_tensor = torch.randn(4, 3, 64, 64)
    
#     # 初始化DCNv2卷积层, 输入通道为3，输出通道为16，卷积核大小为3
#     dcn = DCNv2(in_channels=3, out_channels=16, kernel_size=3)
    
#     # 使用DCNv2卷积层处理输入张量
#     output_tensor = dcn(input_tensor)
    
#     # 打印输出张量的形状
#     print("Output tensor shape:", output_tensor.shape)

# if __name__ == "__main__":
#     main()