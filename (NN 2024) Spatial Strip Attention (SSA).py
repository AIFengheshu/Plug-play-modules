import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文题目：Dual-domain strip attention for image restoration
# 中文题目：双域条带注意力用于图像恢复
# 论文链接：https://doi.org/10.1016/j.neunet.2023.12.003
# 官方github：https://github.com/c-yn/DSANet
# 所属机构：
# 德国慕尼黑工业大学计算、信息与技术学院
# 代码整理：微信公众号：AI缝合术

# Spatial Strip Attention (SSA)
class SSA(nn.Module):
    def __init__(self, dim, group, kernel) -> None:
        super().__init__()

        self.H_spatial_att = spatial_strip_att_unit(dim, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att_unit(dim, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))
        self.beta = nn.Parameter(torch.ones(dim,1,1))
    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta

class spatial_strip_att_unit(nn.Module):
    def __init__(self, dim, kernel=5, group=2, H=True) -> None:
        super().__init__()
        self.k = kernel
        pad = kernel // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel//2, 1) if H else (1, kernel//2)

        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Sigmoid()

    def forward(self, x):
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c//self.group, self.k, h*w)
        
        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1//self.k, self.k, p*q).unsqueeze(2)
        filter = self.filter_act(filter)

        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return out

    
if __name__ == "__main__":

    # 模块参数
    batch_size = 1    # 批大小
    channels = 32     # 输入特征通道数
    height = 256      # 图像高度
    width = 256        # 图像宽度
    
    # 创建 SSA 模块
    ssa = SSA(dim=32, group=1, kernel=7)
    print(ssa)
    print("微信公众号:AI缝合术, nb!")
    
    # 生成随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)
    
    # 打印输入张量的形状
    print("Input shape:", x.shape)
    
    # 前向传播计算输出
    output = ssa(x)
    
    # 打印输出张量的形状
    print("Output shape:", output.shape)
