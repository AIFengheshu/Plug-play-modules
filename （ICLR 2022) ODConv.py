import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

# ������Ŀ��Omni-Dimensional Dynamic Convolution
# ������Ŀ:  ȫά�ȶ�̬���
# �������ӣ�https://openreview.net/pdf?id=DmpCfq6Mg39
# �ٷ�github��https://github.com/OSVAI/ODConv
# ��������΢�Ź��ںţ�AI�����
# github��https://github.com/AIFengheshu/Plug-play-modules

# ODConv�࣬�̳���nn.Sequential
class ODConv(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=nn.BatchNorm2d,
                 reduction=0.0625, kernel_num=1):
        padding = (kernel_size - 1) // 2  # ����kernel_size��������С
        super(ODConv, self).__init__(
            # ʹ�� ODConv2d �����Ӧ������������ͨ���ı仯
            ODConv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups,
                     reduction=reduction, kernel_num=kernel_num),
            norm_layer(out_planes),  # ��ӱ�׼����
            nn.SiLU()  # ʹ��SiLU�����
        )

# ע���������࣬�̳���nn.Module
class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,
                 groups=1,
                 reduction=0.0625,
                 kernel_num=4,
                 min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)  # ȷ��ע����ͨ����
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0  # �¶Ȳ��������ڿ���softmax��ƽ����

        self.avgpool = nn.AdaptiveAvgPool2d(1)  # ����Ӧȫ��ƽ���ػ�
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)  # ���Բ�
        self.bn = nn.BatchNorm2d(attention_channel)  # ����һ��
        self.relu = nn.ReLU(inplace=True)  # ReLU�����

        # ͨ��ע������֧
        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention  # ��ȡͨ��ע����

        # �ж��Ƿ�Ϊ��Ⱦ��
        if in_planes == groups and in_planes == out_planes:  # ��Ⱦ�������
            self.func_filter = self.skip
        else:
            # �����˲�ע������֧
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention  # ��ȡ�˲�ע����

        # �ж��Ƿ�Ϊ����
        if kernel_size == 1:  # ���������
            self.func_spatial = self.skip
        else:
            # ���ÿռ�ע������֧
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention  # ��ȡ�ռ�ע����

        # �����ں�ע������֧
        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention  # ��ȡ�ں�ע����

        self.bn_1 = nn.LayerNorm([attention_channel, 1, 1])  # ���һ��
        self._initialize_weights()  # ��ʼ��Ȩ��

    # ��ʼ��Ȩ��
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # �����¶Ȳ���
    def update_temperature(self, temperature):
        self.temperature = temperature

    # ��̬��������������������1.0
    @staticmethod
    def skip(_):
        return 1.0

    # ��ȡͨ��ע����
    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    # ��ȡ�˲�ע����
    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    # ��ȡ�ռ�ע����
    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    # ��ȡ�ں�ע����
    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention
    
    # ǰ�򴫲�
    def forward(self, x):
        x = self.avgpool(x)  # ȫ��ƽ���ػ�
        x = self.fc(x)  # ͨ������
        x = self.bn_1(x)  # ���һ��
        x = self.relu(x)  # ReLU����
        # ����ͨ�����˲����ռ���ں�ע����
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


# ���� ODConv2d �࣬�̳��� nn.Module
class ODConv2d(nn.Module):
    def __init__(self,
                 in_planes,  # ����ͨ����
                 out_planes,  # ���ͨ����
                 kernel_size=3,  # ����˴�С��Ĭ��Ϊ3
                 stride=1,  # ������Ĭ��Ϊ1
                 padding=0,  # ����С��Ĭ��Ϊ0
                 dilation=1,  # ����ϵ����Ĭ��Ϊ1
                 groups=1,  # ����������Ĭ��Ϊ1
                 reduction=0.0625,  # ͨ����������
                 kernel_num=1):  # ��������
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        # ����Attention����ʵ�ֶ���ע��������
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        # �����ѧϰ��Ȩ�ز�����shapeΪ��kernel_num�����ͨ����������ͨ����/����������˴�С������˴�С��
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()  # ��ʼ��Ȩ��

        # ���Ϊ1x1����Һ�������Ϊ1����ѡ�������ʵ�ַ�ʽ
        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common  # ����ѡ��ͨ��ʵ�ַ�ʽ

    # ��ʼ��Ȩ�ط���
    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    # ����Attention�¶Ȳ���
    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    # ͨ��ǰ�򴫲�ʵ�ַ�ʽ
    def _forward_impl_common(self, x):
        # ��ȡͨ��ע�������˲�ע�������ռ�ע�����ͺ���ע����
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention  # Ӧ��ͨ��ע����
        x = x.reshape(1, -1, height, width)  # ����Ϊ1ά��������

        # ����ۺ�Ȩ�أ����ռ䡢���ĺ�ԭʼȨ�ؽ��
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        # �ۺϺ��Ȩ�ذ�Ҫ����״��������
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])

        # ʹ�þۺϺ��Ȩ�ؽ��о������
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        # ���������ָ���������״����Ӧ���˲�ע����
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention  # Ӧ���˲�ע����
        return output

    # �ض�1x1�����ǰ�򴫲�ʵ��
    def _forward_impl_pw1x(self, x):
        # ��ȡͨ��ע�������˲�ע�������ռ�ע�����ͺ���ע����
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention  # Ӧ��ͨ��ע����

        # ֱ��ʹ��Ȩ�ؽ��о������
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention  # Ӧ���˲�ע����
        return output

    # ǰ�򴫲�����
    def forward(self, x):
        return self._forward_impl(x)  # ����ѡ����ʵ�ַ�ʽ

if __name__ == '__main__':
    # ȷ���豸�������GPU��ʹ��
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ����һ��������������״Ϊ [1, 32, 256, 256] �����õ��豸��
    input = torch.randn(1,32,256,256).to(device)
    # ʵ���� odconv �����
    odconv = ODConv(32, 64).to(device)
    # ͨ�������������
    output = odconv(input)
    # ��ӡ������������״
    print(input.shape)
    print(output.shape)
