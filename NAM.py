import torch.nn as nn
import torch
# from torchinfo import summary  # �������������ע��
# from fvcore.nn import FlopCountAnalysis, flop_count_table  # �������������ע��

# ������Ŀ��NAM: Normalization-based Attention Module
# ������Ŀ:  NAM�� ���ڹ�һ����ע����ģ��
# �������ӣ�https://arxiv.org/pdf/2111.12419
# �ٷ�github��https://github.com/Christian-lyc/NAM
# ����������������ѧҽѧԺ��������Ϣ����ѧԺ��
# �ؼ��ʣ���һ��ע�������ռ�ע������ͨ��ע������ͼ�����
# ΢�Ź��ںţ�AI�����

class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual
        return x

# ע������δ��Դ�ռ�ע�������룬���´����ɡ�΢�Ź��ںţ�AI��������ṩ.
class Spatial_Att(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Att, self).__init__()
        self.kernel_size = kernel_size
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # ����������x����״Ϊ (B, C, H, W)
        residual = x
        # ����ÿ�����ص�Ȩ�� (Pixel Normalization)
        pixel_weight = x.mean(dim=1, keepdim=True)  # ��ͨ�����ֵ����״Ϊ (B, 1, H, W)
        normalized_weight = pixel_weight / pixel_weight.sum(dim=(2, 3), keepdim=True)  # ���ع�һ��
        # ��Ȩ��������
        x = x * normalized_weight  # ������λ�ü�Ȩ
        x = torch.sigmoid(x) * residual  # ��ԭ�������
        return x

class NAM(nn.Module):
    def __init__(self, channels):
        super(NAM, self).__init__()
        self.Channel_Att = Channel_Att(channels)
        self.Spatial_Att = Spatial_Att()

    def forward(self, x):
        x_out1 = self.Channel_Att(x)
        x_out2 = self.Spatial_Att(x_out1)
        return x_out2

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    nam = NAM(channels=32).to(device)  # ��ģ���ƶ����豸
    input = torch.rand(1, 32, 256, 256).to(device)  # �����������ƶ����豸

    # # ���������㣬��ע��
    # print("Model Summary:")
    # print(summary(nam, input_size=(1, 32, 256, 256), device=device.type))

    # # ���������㣬��ע��
    # flops = FlopCountAnalysis(nam, input)
    # print("\nFlop Count Table:")
    # print(flop_count_table(flops))

    output = nam(input)
    print(f"\nInput shape: {input.shape}")
    print(f"Output shape: {output.shape}")