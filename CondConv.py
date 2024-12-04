import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

# 论文题目：SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution
# 中文题目:  CondConv:用于高效推理的条件参数化卷积
# 论文链接：https://arxiv.org/pdf/1904.04971
# 官方github：https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/condconv
# 所属机构：Google Brain
# 关键词：条件参数化卷积（CondConv）、深度神经网络、卷积层、计算效率、图像分类、目标检测

# 代码整理：微信公众号：AI缝合术

class _routing(nn.Module):
    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)  # 定义Dropout层
        self.fc = nn.Linear(in_channels, num_experts)  # 定义全连接层

    def forward(self, x):
        x = torch.flatten(x)  # 将输入展平
        x = self.dropout(x)  # 应用Dropout
        x = self.fc(x)  # 通过全连接层
        return torch.sigmoid(x)  # 使用Sigmoid激活函数

class CondConv2D(_ConvNd):
    r"""为每个样本学习特定的卷积核。

    根据论文《CondConv: Conditionally Parameterized Convolutions for Efficient Inference》描述，
    条件卷积（CondConv）根据输入动态生成卷积核，
    打破了传统静态卷积核的模式。

    参数：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int或tuple): 卷积核大小
        stride (int或tuple, 可选): 卷积步幅，默认值为1
        padding (int或tuple, 可选): 输入两侧的零填充，默认值为0
        padding_mode (str, 可选): 填充模式，如'zeros'、'reflect'等，默认值为'zeros'
        dilation (int或tuple, 可选): 卷积核元素间距，默认值为1
        groups (int, 可选): 输入输出通道分组数量，默认值为1
        bias (bool, 可选): 是否添加偏置项，默认值为True
        num_experts (int): 每层的专家数量
        dropout_rate (float): Dropout的概率

    输入输出形状：
        输入：形状为(N, C_in, H_in, W_in)
        输出：形状为(N, C_out, H_out, W_out)

    属性：
        weight (Tensor): 学习的卷积核权重
        bias (Tensor): 可选的偏置项
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', num_experts=3, dropout_rate=0.2):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CondConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))  # 自适应平均池化
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)  # 路由函数

        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))  # 初始化卷积核权重

        self.reset_parameters()  # 重置参数

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':  # 如果不是零填充，执行自定义填充模式
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, inputs):
        b, _, _, _ = inputs.size()  # 获取输入的批大小
        res = []
        for input in inputs:
            input = input.unsqueeze(0)  # 增加一个维度
            pooled_inputs = self._avg_pooling(input)  # 自适应平均池化
            routing_weights = self._routing_fn(pooled_inputs)  # 生成路由权重
            kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)  # 加权卷积核
            out = self._conv_forward(input, kernels)  # 执行卷积操作
            res.append(out)  # 存储结果
        return torch.cat(res, dim=0)  # 拼接结果返回

if __name__ == '__main__':

    cond = CondConv2D(32, 64, kernel_size=1, num_experts=3, dropout_rate=0)
    input = torch.randn(1, 32, 256, 256)
    print(input.size())
    output = cond(input)
    print(output.size())