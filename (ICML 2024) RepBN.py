import torch
from torch import nn

# ������Ŀ��SLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized Batch Normalization
# ������Ŀ��SLAB�����м�����ע�����ͽ���ʽ�ز�����������һ���ĸ�Ч�任��
# �������ӣ�https://arxiv.org/pdf/2405.11582
# �ٷ�github��https://github.com/xinghaochen/SLAB
# ������������Ϊŵ�Ƿ���ʵ����

# ��������:΢�Ź��ں�:AI�����

# Դ����, ������ά����
class RepBN(nn.Module):
    def __init__(self, channels):
        super(RepBN, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn(x) + self.alpha * x
        x = x.transpose(1, 2)
        return x

# ����չ����ά, ��������:΢�Ź��ں�:AI����� 
class RepBN2d(nn.Module):
    def __init__(self, channels):
        super(RepBN2d, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm2d(channels)  # ʹ��BatchNorm2d

    def forward(self, x):
        # BatchNorm2d������ά��������
        x = self.bn(x) + self.alpha * x  # BatchNorm + alpha * input
        return x
        
# if __name__ == "__main__":
#     # ģ�����
#     batch_size = 1    # ����С
#     channels = 32     # ��������ͨ����
#     N = 16 * 16      # ͼ��߶�*��� height * width

#     model = RepBN(channels = channels)
#     print(model)
#     print("΢�Ź��ں�:AI�����, nb!")

#     # ��������������� (batch_size, channels, height * width (N))
#     x = torch.randn(batch_size, N, channels)
#     # ��ӡ������������״
#     print("Input shape:", x.shape)
#     # ǰ�򴫲��������
#     output = model(x)
#     # ��ӡ�����������״
#     print("Output shape:", output.shape)

if __name__ == "__main__":
    # ģ�����
    batch_size = 1    # ����С
    channels = 32     # ��������ͨ����
    height = 256      # ͼ��߶�
    width = 256        # ͼ����

    model = RepBN2d(channels = channels)
    print(model)
    print("΢�Ź��ں�:AI�����, nb!")

    # ��������������� (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)
    # ��ӡ������������״
    print("Input shape:", x.shape)
    # ǰ�򴫲��������
    output = model(x)
    # ��ӡ�����������״
    print("Output shape:", output.shape)