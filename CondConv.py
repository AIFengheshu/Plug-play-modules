import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

# ������Ŀ��SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution
# ������Ŀ:  CondConv:���ڸ�Ч������������������
# �������ӣ�https://arxiv.org/pdf/1904.04971
# �ٷ�github��https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/condconv
# ����������Google Brain
# �ؼ��ʣ����������������CondConv������������硢����㡢����Ч�ʡ�ͼ����ࡢĿ����

# ��������΢�Ź��ںţ�AI�����

class _routing(nn.Module):
    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)  # ����Dropout��
        self.fc = nn.Linear(in_channels, num_experts)  # ����ȫ���Ӳ�

    def forward(self, x):
        x = torch.flatten(x)  # ������չƽ
        x = self.dropout(x)  # Ӧ��Dropout
        x = self.fc(x)  # ͨ��ȫ���Ӳ�
        return torch.sigmoid(x)  # ʹ��Sigmoid�����

class CondConv2D(_ConvNd):
    r"""Ϊÿ������ѧϰ�ض��ľ���ˡ�

    �������ġ�CondConv: Conditionally Parameterized Convolutions for Efficient Inference��������
    ���������CondConv���������붯̬���ɾ���ˣ�
    �����˴�ͳ��̬����˵�ģʽ��

    ������
        in_channels (int): ����ͨ����
        out_channels (int): ���ͨ����
        kernel_size (int��tuple): ����˴�С
        stride (int��tuple, ��ѡ): ���������Ĭ��ֵΪ1
        padding (int��tuple, ��ѡ): �������������䣬Ĭ��ֵΪ0
        padding_mode (str, ��ѡ): ���ģʽ����'zeros'��'reflect'�ȣ�Ĭ��ֵΪ'zeros'
        dilation (int��tuple, ��ѡ): �����Ԫ�ؼ�࣬Ĭ��ֵΪ1
        groups (int, ��ѡ): �������ͨ������������Ĭ��ֵΪ1
        bias (bool, ��ѡ): �Ƿ����ƫ���Ĭ��ֵΪTrue
        num_experts (int): ÿ���ר������
        dropout_rate (float): Dropout�ĸ���

    ���������״��
        ���룺��״Ϊ(N, C_in, H_in, W_in)
        �������״Ϊ(N, C_out, H_out, W_out)

    ���ԣ�
        weight (Tensor): ѧϰ�ľ����Ȩ��
        bias (Tensor): ��ѡ��ƫ����
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

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))  # ����Ӧƽ���ػ�
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)  # ·�ɺ���

        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))  # ��ʼ�������Ȩ��

        self.reset_parameters()  # ���ò���

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':  # �����������䣬ִ���Զ������ģʽ
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, inputs):
        b, _, _, _ = inputs.size()  # ��ȡ���������С
        res = []
        for input in inputs:
            input = input.unsqueeze(0)  # ����һ��ά��
            pooled_inputs = self._avg_pooling(input)  # ����Ӧƽ���ػ�
            routing_weights = self._routing_fn(pooled_inputs)  # ����·��Ȩ��
            kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)  # ��Ȩ�����
            out = self._conv_forward(input, kernels)  # ִ�о������
            res.append(out)  # �洢���
        return torch.cat(res, dim=0)  # ƴ�ӽ������

if __name__ == '__main__':

    cond = CondConv2D(32, 64, kernel_size=1, num_experts=3, dropout_rate=0)
    input = torch.randn(1, 32, 256, 256)
    print(input.size())
    output = cond(input)
    print(output.size())