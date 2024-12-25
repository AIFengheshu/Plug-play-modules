import torch
import torch.nn as nn

# ������Ŀ��Dual Residual Attention Network for Image Denoising
# ������Ŀ��˫�زв�ע������������ͼ��ȥ��
# �������ӣ�https://arxiv.org/pdf/2305.04269
# �ٷ�github��https://github.com/WenCongWu/DRANet
# ��������������ʦ����ѧ��Ϣ��ѧ����ѧԺ
# �ؼ��ʣ�ͼ��ȥ�룬˫����Ⱦ�����磬�в�ע����ѧϰ����ϲв�ע����ѧϰ
# ��������΢�Ź��ںš�AI�������

## Spatial Attention
class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding=0, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes
        groups = 1
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
    
class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SAB(nn.Module):
    def __init__(self):
        super(SAB, self).__init__()
        kernel_size = 5
        self.compress = ChannelPool()
        self.spatial = Basic(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale
class Residual_Attention_Block(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(Residual_Attention_Block, self).__init__()
        kernel_size = 3
        stride = 1
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        self.res = nn.Sequential(*layers)
        self.sab = SAB()

    def forward(self, x):
        x1 = x + self.res(x)
        x2 = x1 + self.res(x1)
        x3 = x2 + self.res(x2)

        x3_1 = x1 + x3
        x4 = x3_1 + self.res(x3_1)
        x4_1 = x + x4

        x5 = self.sab(x4_1)
        x5_1 = x + x5

        return x5_1

if __name__ == '__main__':
    print("΢�Ź��ں�:AI�����\n")

    rsb = Residual_Attention_Block(in_channels=32, out_channels=32, bias=True)# ����ͨ���� \ ���ͨ���� \ �Ƿ��ھ����ʹ��ƫ��
    # �����������
    input_tensor = torch.randn(1, 32, 256, 256)
    output_tensor = rsb(input_tensor)
    # �����״���
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)