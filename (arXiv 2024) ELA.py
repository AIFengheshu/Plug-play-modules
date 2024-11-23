import torch
import torch.nn as nn

# 论文题目：ELA: Efficient Local Attention for Deep Convolutional Neural Networks
# 中文题目:  ELA: 深度卷积神经网络的高效局部注意力
# 论文链接：https://arxiv.org/pdf/2403.01123
# 官方github：无
# 所属机构：兰州大学信息科学与工程学院，青海省物联网重点实验室，青海师范大学
# 关键词：注意力机制，深度卷积神经网络，图像分类，目标检测，语义分割

# 微信公众号：AI缝合术 https://github.com/AIFengheshu/Plug-play-modules

class EfficientLocalizationAttention(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(EfficientLocalizationAttention, self).__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        b, c, h, w = x.size()
 
        # 处理高度维度
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)
 
        # 处理宽度维度
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)

        # 在两个维度上应用注意力
        return x * x_h * x_w
 
 
# 示例用法 ELABase(ELA-B)
if __name__ == "__main__":
    # 创建一个形状为 [batch_size, channels, height, width]输入张量
    input = torch.randn(1, 32, 256, 256)
    print(f"输入形状: {input.shape}")
    # 初始化模块
    ela = EfficientLocalizationAttention(channel=32, kernel_size=7)
    # 前向传播
    output = ela(input)
    # 打印出输出张量的形状，它将与输入形状相匹配
    print(f"输出形状: {output.shape}")