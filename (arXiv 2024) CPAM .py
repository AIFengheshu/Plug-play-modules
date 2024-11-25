import torch
import torch.nn as nn
import math

# ������Ŀ��ASF-YOLO: A novel YOLO model with attentional scale sequence fusion for cell instance segmentation
# ������Ŀ:  ASF-YOLO������ע�������Ƶĳ߶������ں�YOLO�������ϸ��ͼ��ʵ���ָ�
# �������ӣ�https://arxiv.org/pdf/2312.06458
# �ٷ�github��https://github.com/mkang315/ASF-YOLO
# ������������������Ī��ʲ��ѧ��Ϣ����ѧԺ
# �ؼ��ʣ�ҽѧͼ�������С����ָYOLO�����������ںϣ�ע��������

class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out

#Channel and Position Attention Mechanism (CPAM)
class CPAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.channel_att = channel_att(ch)
        self.local_att = local_att(ch)
    def forward(self, x):
        input1,input2 = x[0],x[1]
        input1 = self.channel_att(input1)
        x = input1 + input2
        x = self.local_att(x)
        return x


if __name__ == '__main__':

    cpam = CPAM(32)

    input1 = torch.randn(1, 32, 256, 256)
    input2 = torch.randn(1, 32, 256, 256)
    print(input1.size())
    print(input2.size())

    inputs = [input1, input2]

    output = cpam(inputs)

    print(output.size())