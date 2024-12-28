import torch
from torch import nn
# 论文题目： Tied Block Convolution: Leaner and Better CNNs with Shared Thinner Filters
# 中文题目：绑定块卷积：使用共享更薄滤波器的更精简且更优的卷积神经网络
# 论文链接：http://people.eecs.berkeley.edu/~xdwang/papers/AAAI2021_TBC.pdf
# 官方github：https://github.com/frank-xwang/TBC-TiedBlockConvolution
# 所属机构：加州大学伯克利分校，国际计算机科学研究所
# 代码整理：微信公众号《AI缝合术》
class TiedSELayer(nn.Module):
    '''Tied Block Squeeze and Excitation Layer'''
    def __init__(self, channel, B=1, reduction=16):
        super(TiedSELayer, self).__init__()
        assert channel % B == 0
        self.B = B
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        channel = channel//B
        self.fc = nn.Sequential(
                nn.Linear(channel, max(2, channel//reduction)),
                nn.ReLU(inplace=True),
                nn.Linear(max(2, channel//reduction), channel),
                nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b*self.B, c//self.B)
        y = self.fc(y).view(b, c, 1, 1)
        return x*y
if __name__ == '__main__':
    
    # 创建实例
    attention = TiedSELayer(32)
    # 生成随机输入数据
    input_data = torch.rand(1,32,256,256)
    output = attention(input_data)
 
    # 打印输入和输出形状
    print("Input size:", input_data.size())
    print("Output size:", output.size())