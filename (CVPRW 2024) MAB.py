import torch
import torch.nn as nn
import torch.nn.functional as F

# ������Ŀ��Multi-scale Attention Network for Single Image Super-Resolution
# ������Ŀ:  ��ͼ�񳬷ֱ��ʵĶ�߶�ע��������
# �������ӣ�https://arxiv.org/abs/2209.14145
# �ٷ�github��https://github.com/icandle/MAN
# ����������ע�ͣ����ںţ�AI�����
# AI�����github��https://github.com/AIFengheshu/Plug-play-modules

# �Զ����LayerNorm�㣬���ڶ��������ݽ��б�׼��
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        # ����Ȩ�ز�������ʼ��Ϊ1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # ����ƫ�ò�������ʼ��Ϊ0
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        # �ж�data_format�Ƿ����Ԥ��
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        # ���ñ�׼����״
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # ������ݸ�ʽ��"channels_last"��ֱ��ʹ��layer_norm
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # �����"channels_first"���ֶ������ֵ�ͷ���
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# Gated Spatial Attention Unit (GSAU)�����ڿռ�ע��������
class GSAU(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2

        # ��һ��1x1����㣬����������չ
        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        # ��ȿɷ������㣬������ȡ�ռ���Ϣ
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        # �ڶ���1x1�����
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        # ��׼����
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        # ��ѧϰ�ĳ߶Ȳ���
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        # ��������ĸ�������Ϊ�в�����
        shortcut = x.clone()

        # ������׼���͵�һ�������
        x = self.Conv1(self.norm(x))
        # ��ͨ����Ϊ�����֣�a�����ſز�����x��������
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)  # �ռ�ע��������
        x = self.Conv2(x)  # �ٴξ��

        # ���ϲв����Ӳ�����
        return x * self.scale + shortcut

# Multi-scale Large Kernel Attention (MLKA)����߶ȴ�˾��ע��������
class MLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        if n_feats % 3 != 0:
            raise ValueError("n_feats�����ܱ�3��������������MLKA")

        i_feats = 2 * n_feats

        # ��׼����
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        # ��ѧϰ�ĳ߶Ȳ���
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        # ������ͬ�߶ȵľ���˷ֱ𹹽�7x7��5x5��3x3��LKAģ��
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats // 3, dilation=4),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // 3, dilation=3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, stride=1, padding=(5 // 2) * 2, groups=n_feats // 3, dilation=2),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))

        # ������ͬ�߶ȵĶ�������
        self.X3 = nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3)
        self.X5 = nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3)
        self.X7 = nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3)

        # ��һ��1x1���������ͨ����չ
        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        # ����1x1��������ڻָ�ͨ������
        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x):
        # ����������Ϊ�в�����
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        # ��������Ϊa��x������
        a, x = torch.chunk(x, 2, dim=1)
        # a��������ͨ����ͬ�߶ȵľ������
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)
        a = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3)],
                      dim=1)
        # �ϲ���Ľ���پ������ľ����
        x = self.proj_last(x * a) * self.scale + shortcut
        return x

# Multi-scale Attention Blocks (MAB)����߶�ע������
class MAB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        # ��˾��ע����ģ��
        self.LKA = MLKA(n_feats)
        # �ſؿռ�ע������Ԫ
        self.LFE = GSAU(n_feats)

    def forward(self, x):
        # ��ͨ����˾��ע����
        x = self.LKA(x)
        # Ȼ��ͨ���ſؿռ�ע����
        x = self.LFE(x)
        return x


if __name__ == '__main__':
    #ͨ������Ҫ��3����
    n_feats = 3 
    mab = MAB(3)
    input = torch.randn(1, 3, 256, 256)
    output = mab(input)
    print(input.size())
    print(output.size())