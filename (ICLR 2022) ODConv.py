import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

# 论文题目：Omni-Dimensional Dynamic Convolution
# 中文题目:  全维度动态卷积
# 论文链接：https://openreview.net/pdf?id=DmpCfq6Mg39
# 官方github：https://github.com/OSVAI/ODConv
# 代码解读：微信公众号：AI缝合术
# github：https://github.com/AIFengheshu/Plug-play-modules

# ODConv类，继承自nn.Sequential
class ODConv(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=nn.BatchNorm2d,
                 reduction=0.0625, kernel_num=1):
        padding = (kernel_size - 1) // 2  # 根据kernel_size计算填充大小
        super(ODConv, self).__init__(
            # 使用 ODConv2d 卷积，应用于输入和输出通道的变化
            ODConv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups,
                     reduction=reduction, kernel_num=kernel_num),
            norm_layer(out_planes),  # 添加标准化层
            nn.SiLU()  # 使用SiLU激活函数
        )

# 注意力机制类，继承自nn.Module
class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,
                 groups=1,
                 reduction=0.0625,
                 kernel_num=4,
                 min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)  # 确定注意力通道数
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0  # 温度参数，用于控制softmax的平滑度

        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 自适应全局平均池化
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)  # 线性层
        self.bn = nn.BatchNorm2d(attention_channel)  # 批归一化
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数

        # 通道注意力分支
        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention  # 获取通道注意力

        # 判断是否为深度卷积
        if in_planes == groups and in_planes == out_planes:  # 深度卷积的情况
            self.func_filter = self.skip
        else:
            # 设置滤波注意力分支
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention  # 获取滤波注意力

        # 判断是否为点卷积
        if kernel_size == 1:  # 点卷积的情况
            self.func_spatial = self.skip
        else:
            # 设置空间注意力分支
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention  # 获取空间注意力

        # 设置内核注意力分支
        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention  # 获取内核注意力

        self.bn_1 = nn.LayerNorm([attention_channel, 1, 1])  # 层归一化
        self._initialize_weights()  # 初始化权重

    # 初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 更新温度参数
    def update_temperature(self, temperature):
        self.temperature = temperature

    # 静态方法：跳过操作，返回1.0
    @staticmethod
    def skip(_):
        return 1.0

    # 获取通道注意力
    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    # 获取滤波注意力
    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    # 获取空间注意力
    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    # 获取内核注意力
    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention
    
    # 前向传播
    def forward(self, x):
        x = self.avgpool(x)  # 全局平均池化
        x = self.fc(x)  # 通道缩减
        x = self.bn_1(x)  # 层归一化
        x = self.relu(x)  # ReLU激活
        # 返回通道、滤波、空间和内核注意力
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


# 定义 ODConv2d 类，继承自 nn.Module
class ODConv2d(nn.Module):
    def __init__(self,
                 in_planes,  # 输入通道数
                 out_planes,  # 输出通道数
                 kernel_size=3,  # 卷积核大小，默认为3
                 stride=1,  # 步幅，默认为1
                 padding=0,  # 填充大小，默认为0
                 dilation=1,  # 膨胀系数，默认为1
                 groups=1,  # 分组卷积数，默认为1
                 reduction=0.0625,  # 通道缩减比例
                 kernel_num=1):  # 核心数量
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        # 定义Attention对象，实现多种注意力机制
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        # 定义可学习的权重参数，shape为（kernel_num，输出通道数，输入通道数/组数，卷积核大小，卷积核大小）
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()  # 初始化权重

        # 如果为1x1卷积且核心数量为1，则选择特殊的实现方式
        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common  # 否则选择通用实现方式

    # 初始化权重方法
    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    # 更新Attention温度参数
    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    # 通用前向传播实现方式
    def _forward_impl_common(self, x):
        # 获取通道注意力、滤波注意力、空间注意力和核心注意力
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention  # 应用通道注意力
        x = x.reshape(1, -1, height, width)  # 重塑为1维批次数据

        # 计算聚合权重，将空间、核心和原始权重结合
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        # 聚合后的权重按要求形状进行重塑
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])

        # 使用聚合后的权重进行卷积操作
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        # 将输出结果恢复成批次形状，并应用滤波注意力
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention  # 应用滤波注意力
        return output

    # 特定1x1卷积的前向传播实现
    def _forward_impl_pw1x(self, x):
        # 获取通道注意力、滤波注意力、空间注意力和核心注意力
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention  # 应用通道注意力

        # 直接使用权重进行卷积操作
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention  # 应用滤波注意力
        return output

    # 前向传播方法
    def forward(self, x):
        return self._forward_impl(x)  # 调用选定的实现方式

if __name__ == '__main__':
    # 确定设备，如果有GPU则使用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建一个输入张量，形状为 [1, 32, 256, 256] 并放置到设备上
    input = torch.randn(1,32,256,256).to(device)
    # 实例化 odconv 卷积层
    odconv = ODConv(32, 64).to(device)
    # 通过卷积层计算输出
    output = odconv(input)
    # 打印输入和输出的形状
    print(input.shape)
    print(output.shape)
