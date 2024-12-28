import torch
from torch import nn

# ������Ŀ�� Tied Block Convolution: Leaner and Better CNNs with Shared Thinner Filters
# ������Ŀ���󶨿�����ʹ�ù�������˲����ĸ������Ҹ��ŵľ��������
# �������ӣ�http://people.eecs.berkeley.edu/~xdwang/papers/AAAI2021_TBC.pdf
# �ٷ�github��https://github.com/frank-xwang/TBC-TiedBlockConvolution
# �������������ݴ�ѧ��������У�����ʼ������ѧ�о���
# ��������΢�Ź��ںš�AI�������

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
    
    # ����ʵ��
    TBC = TiedBlockConv2d(in_planes=32, planes=64, kernel_size=3, stride=1, padding=1)

    # ���������������
    input_data = torch.rand(1,32,256,256)
    output = TBC(input_data)
 
    # ��ӡ����������״
    print("Input size:", input_data.size())
    print("Output size:", output.size())