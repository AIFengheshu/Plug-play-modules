import torch
import torch.nn as nn
from thop import profile  # 用于计算模型的FLOPS和参数量


class Simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(Simam_module, self).__init__()
        self.act = nn.Sigmoid()  # 使用Sigmoid激活函数
        self.e_lambda = e_lambda  # 防止除零的平滑因子

    def forward(self, x):
        # 获取输入的批次大小、通道数、高度和宽度
        b, c, h, w = x.size()
        n = w * h - 1  # 计算有效像素数（去掉一个像素）

        # 计算每个特征点与其局部均值的平方差
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        # 计算能量值y
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        # 通过Sigmoid激活函数生成注意力权重，并与输入特征图相乘
        return x * self.act(y)


# 无参注意力机制，输入 N C H W，输出 N C H W
if __name__ == '__main__':
    model = Simam_module().cuda()  # 实例化SimAM模块并转移到GPU
    x = torch.randn(1, 3, 64, 64).cuda()  # 创建一个随机输入张量
    y = model(x)  # 通过模型前向传播获取输出
    print(y.size())  # 打印输出的尺寸
    flops, params = profile(model, (x,))  # 计算模型的FLOPS和参数量
    print(flops / 1e9)  # 打印FLOPS，单位为十亿
    print(params)  # 打印模型的参数数量
