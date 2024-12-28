import torch
from torch import nn
# ������Ŀ�� Tied Block Convolution: Leaner and Better CNNs with Shared Thinner Filters
# ������Ŀ���󶨿�����ʹ�ù�������˲����ĸ������Ҹ��ŵľ��������
# �������ӣ�http://people.eecs.berkeley.edu/~xdwang/papers/AAAI2021_TBC.pdf
# �ٷ�github��https://github.com/frank-xwang/TBC-TiedBlockConvolution
# �������������ݴ�ѧ��������У�����ʼ������ѧ�о���
# ��������΢�Ź��ںš�AI�������
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
    
    # ����ʵ��
    attention = TiedSELayer(32)
    # ���������������
    input_data = torch.rand(1,32,256,256)
    output = attention(input_data)
 
    # ��ӡ����������״
    print("Input size:", input_data.size())
    print("Output size:", output.size())