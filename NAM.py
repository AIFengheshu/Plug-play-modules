import torch.nn as nn
import torch
# from torchinfo import summary  # 计算参数量，可注释
# from fvcore.nn import FlopCountAnalysis, flop_count_table  # 计算计算量，可注释

# 论文题目：NAM: Normalization-based Attention Module
# 中文题目:  NAM： 基于归一化的注意力模块
# 论文链接：https://arxiv.org/pdf/2111.12419
# 官方github：https://github.com/Christian-lyc/NAM
# 所属机构：东北大学医学院和生物信息工程学院等
# 关键词：归一化注意力、空间注意力、通道注意力、图像分类
# 微信公众号：AI缝合术

class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual
        return x

# 注：作者未开源空间注意力代码，以下代码由《微信公众号：AI缝合术》提供.
class Spatial_Att(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Att, self).__init__()
        self.kernel_size = kernel_size
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 输入特征：x，形状为 (B, C, H, W)
        residual = x
        # 计算每个像素的权重 (Pixel Normalization)
        pixel_weight = x.mean(dim=1, keepdim=True)  # 按通道求均值，形状为 (B, 1, H, W)
        normalized_weight = pixel_weight / pixel_weight.sum(dim=(2, 3), keepdim=True)  # 像素归一化
        # 加权输入特征
        x = x * normalized_weight  # 按像素位置加权
        x = torch.sigmoid(x) * residual  # 与原输入相乘
        return x

class NAM(nn.Module):
    def __init__(self, channels):
        super(NAM, self).__init__()
        self.Channel_Att = Channel_Att(channels)
        self.Spatial_Att = Spatial_Att()

    def forward(self, x):
        x_out1 = self.Channel_Att(x)
        x_out2 = self.Spatial_Att(x_out1)
        return x_out2

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    nam = NAM(channels=32).to(device)  # 将模型移动到设备
    input = torch.rand(1, 32, 256, 256).to(device)  # 将输入数据移动到设备

    # # 参数量计算，可注释
    # print("Model Summary:")
    # print(summary(nam, input_size=(1, 32, 256, 256), device=device.type))

    # # 计算量计算，可注释
    # flops = FlopCountAnalysis(nam, input)
    # print("\nFlop Count Table:")
    # print(flop_count_table(flops))

    output = nam(input)
    print(f"\nInput shape: {input.shape}")
    print(f"Output shape: {output.shape}")