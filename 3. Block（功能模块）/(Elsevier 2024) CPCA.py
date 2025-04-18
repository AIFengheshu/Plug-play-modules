import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文题目：Channel prior convolutional attention for medical image segmentation
# 中文题目:  用于医疗图像分割的通道先验卷积注意力
# 论文链接：https://arxiv.org/pdf/2306.05196
# 官方github：https://github.com/Cuthbert-Huang/CPCANet
# 代码整理与注释：公众号：AI缝合术
# AI缝合术github：https://github.com/AIFengheshu/Plug-play-modules

# CPCA通道注意力模块
class CPCA_ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(CPCA_ChannelAttention, self).__init__()
        # 使用 1x1 卷积来减少通道维度 (input_channels -> internal_neurons)
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        # 使用 1x1 卷积恢复通道维度 (internal_neurons -> input_channels)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels  # 保存输入通道数

    def forward(self, inputs):
        # 使用自适应平均池化获取每个通道的全局信息
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)  # 通道维度压缩
        x1 = F.relu(x1, inplace=True)  # 激活函数
        x1 = self.fc2(x1)  # 恢复通道维度
        x1 = torch.sigmoid(x1)  # 使用 Sigmoid 激活函数

        # 使用自适应最大池化获取每个通道的全局信息
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)  # 通道维度压缩
        x2 = F.relu(x2, inplace=True)  # 激活函数
        x2 = self.fc2(x2)  # 恢复通道维度
        x2 = torch.sigmoid(x2)  # 使用 Sigmoid 激活函数

        # 将平均池化和最大池化的结果加权求和
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)  # 重新调整形状
        return inputs * x  # 将输入与通道注意力加权后相乘


# CPCA模块
class CPCA(nn.Module):
    def __init__(self, channels, channelAttention_reduce=4):
        super().__init__()

        # 初始化通道注意力模块
        self.ca = CPCA_ChannelAttention(input_channels=channels, internal_neurons=channels // channelAttention_reduce)
        
        # 初始化深度可分离卷积层（分别处理通道和空间信息）
        self.dconv5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)  # 5x5 深度可分离卷积
        self.dconv1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)  # 1x7 深度可分离卷积
        self.dconv7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)  # 7x1 深度可分离卷积
        self.dconv1_11 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)  # 1x11 深度可分离卷积
        self.dconv11_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)  # 11x1 深度可分离卷积
        self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)  # 1x21 深度可分离卷积
        self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)  # 21x1 深度可分离卷积
        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)  # 1x1 标准卷积
        self.act = nn.GELU()  # GELU 激活函数

    def forward(self, inputs):
        # Global Perceptron：通过 1x1 卷积和激活函数生成初始的全局表示
        inputs = self.conv(inputs)
        inputs = self.act(inputs)
        
        # 通过通道注意力模块调整通道权重
        inputs = self.ca(inputs)

        # 使用不同的卷积核处理空间信息，分别获得不同尺度的特征
        x_init = self.dconv5_5(inputs)  # 5x5 卷积
        x_1 = self.dconv1_7(x_init)  # 1x7 卷积
        x_1 = self.dconv7_1(x_1)  # 7x1 卷积
        x_2 = self.dconv1_11(x_init)  # 1x11 卷积
        x_2 = self.dconv11_1(x_2)  # 11x1 卷积
        x_3 = self.dconv1_21(x_init)  # 1x21 卷积
        x_3 = self.dconv21_1(x_3)  # 21x1 卷积
        
        # 合并不同尺度的信息
        x = x_1 + x_2 + x_3 + x_init
        
        # 使用 1x1 卷积进行最终的空间注意力特征生成
        spatial_att = self.conv(x)
        
        # 将空间注意力与输入特征相乘
        out = spatial_att * inputs
        
        # 最后进行一次卷积
        out = self.conv(out)
        
        return out  # 返回最终的输出

# 测试代码
if __name__ == '__main__':
    cpca = CPCA(channels=256)  # 创建 CPCA 模型，输入通道数为 256
    input = torch.randn(1, 256, 32, 32)  # 生成一个随机输入，大小为 (1, 256, 32, 32)
    output = cpca(input)  # 通过 CPCA 模型进行前向传播
    print(input.shape)  # 打印输入张量的形状
    print(output.shape)  # 打印输出张量的形状