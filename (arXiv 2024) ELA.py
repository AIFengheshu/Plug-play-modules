import torch
import torch.nn as nn

# ������Ŀ��ELA: Efficient Local Attention for Deep Convolutional Neural Networks
# ������Ŀ:  ELA: ��Ⱦ��������ĸ�Ч�ֲ�ע����
# �������ӣ�https://arxiv.org/pdf/2403.01123
# �ٷ�github����
# �������������ݴ�ѧ��Ϣ��ѧ�빤��ѧԺ���ຣʡ�������ص�ʵ���ң��ຣʦ����ѧ
# �ؼ��ʣ�ע�������ƣ���Ⱦ�������磬ͼ����࣬Ŀ���⣬����ָ�

# ΢�Ź��ںţ�AI����� https://github.com/AIFengheshu/Plug-play-modules

class EfficientLocalizationAttention(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(EfficientLocalizationAttention, self).__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        b, c, h, w = x.size()
 
        # ����߶�ά��
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)
 
        # ������ά��
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)

        # ������ά����Ӧ��ע����
        return x * x_h * x_w
 
 
# ʾ���÷� ELABase(ELA-B)
if __name__ == "__main__":
    # ����һ����״Ϊ [batch_size, channels, height, width]��������
    input = torch.randn(1, 32, 256, 256)
    print(f"������״: {input.shape}")
    # ��ʼ��ģ��
    ela = EfficientLocalizationAttention(channel=32, kernel_size=7)
    # ǰ�򴫲�
    output = ela(input)
    # ��ӡ�������������״��������������״��ƥ��
    print(f"�����״: {output.shape}")