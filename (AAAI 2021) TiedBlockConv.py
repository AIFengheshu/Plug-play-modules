import torch
from torch import nn

# 论文题目： Tied Block Convolution: Leaner and Better CNNs with Shared Thinner Filters
# 中文题目：绑定块卷积：使用共享更薄滤波器的更精简且更优的卷积神经网络
# 论文链接：http://people.eecs.berkeley.edu/~xdwang/papers/AAAI2021_TBC.pdf
# 官方github：https://github.com/frank-xwang/TBC-TiedBlockConvolution
# 所属机构：加州大学伯克利分校，国际计算机科学研究所
# 代码整理：微信公众号《AI缝合术》

class TiedBlockConv2d(nn.Module):
    '''Tied Block Conv2d'''
    def __init__(self, in_planes, planes, kernel_size, stride=1, padding=0, bias=True, \
                B=1, args=None, dropout_tbc=0.0, groups=1):
        super(TiedBlockConv2d, self).__init__()
        assert planes % B == 0
        assert in_planes % B == 0
        self.B = B
        self.stride = stride
        self.padding = padding
        self.out_planes = planes
        self.kernel_size = kernel_size
        self.dropout_tbc = dropout_tbc

        self.conv = nn.Conv2d(in_planes//self.B, planes//self.B, kernel_size=kernel_size, stride=stride, \
                    padding=padding, bias=bias, groups=groups)
        if self.dropout_tbc > 0.0:
            self.drop_out = nn.Dropout(self.dropout_tbc)

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.contiguous().view(n*self.B, c//self.B, h, w)
        h_o = (h - self.kernel_size + 2*self.padding) // self.stride + 1
        w_o = (w - self.kernel_size + 2*self.padding) // self.stride + 1
        x = self.conv(x)
        x = x.view(n, self.out_planes, h_o, w_o)
        if self.dropout_tbc > 0:
            x = self.drop_out(x)
        return x

if __name__ == '__main__':
    
    # 创建实例
    TBC = TiedBlockConv2d(in_planes=32, planes=64, kernel_size=3, stride=1, padding=1)

    # 生成随机输入数据
    input_data = torch.rand(1,32,256,256)
    output = TBC(input_data)
 
    # 打印输入和输出形状
    print("Input size:", input_data.size())
    print("Output size:", output.size())