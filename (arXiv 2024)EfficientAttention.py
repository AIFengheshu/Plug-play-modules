import torch
from torch import nn
from torch.nn import functional as f

# ������Ŀ��Efficient Attention: Attention with Linear Complexities
# ������Ŀ:  ��Чע�������������Ը��Ӷȵ�ע��������
# �������ӣ�https://arxiv.org/pdf/1812.01243
# �ٷ�github��https://github.com/cmsflash/efficient-attention

# ����������ע�ͣ����ںţ�AI�����
# AI�����github��https://github.com/AIFengheshu/Plug-play-modules

# EfficientAttention ģ�飺һ����Ч�Ķ�ͷע��������
class EfficientAttention(nn.Module):
    
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        """
        ��ʼ�������������� EfficientAttention ģ��ĸ�����
        :param in_channels: ����������ͨ����
        :param key_channels: ����ͨ����
        :param head_count: ע����ͷ������
        :param value_channels: ֵ��ͨ����
        """
        super().__init__()
        self.in_channels = in_channels  # ����ͨ����
        self.key_channels = key_channels  # ����ͨ����
        self.head_count = head_count  # ע����ͷ��
        self.value_channels = value_channels  # ֵ��ͨ����

        # 1x1 ����㣬�������ɼ���keys������ѯ��queries����ֵ��values��
        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)

        # ���� 1x1 ������ڽ�ע�������ӳ�������ͨ����
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        """
        ǰ�򴫲����������������ע�������
        :param input_: ��������ͼ (batch_size, in_channels, height, width)
        :return: �������ע������Ȩ�������ͼ
        """
        n, _, h, w = input_.size()  # ��ȡ��������ͼ����״ (batch_size, channels, height, width)
        
        # ���������ѯ��ֵ
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))  # reshape Ϊ (batch_size, key_channels, height * width)
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)  # reshape Ϊ (batch_size, key_channels, height * width)
        values = self.values(input_).reshape((n, self.value_channels, h * w))  # reshape Ϊ (batch_size, value_channels, height * width)
        
        # ����ÿ��ͷ�ļ�ͨ������ֵͨ����
        head_key_channels = self.key_channels // self.head_count  # ÿ��ͷ�ļ���ͨ����
        head_value_channels = self.value_channels // self.head_count  # ÿ��ͷ��ֵ��ͨ����
        
        attended_values = []  # ���ڴ洢ÿ��ͷ�����

        # ��ÿ��ͷ����ע��������
        for i in range(self.head_count):
            # �Ӽ���ֵ����ȡ��ǰͷ�Ĳ���
            key = f.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = f.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            
            # ����ע����
            context = key @ value.transpose(1, 2)  # ����ֵ�ĳ˻����õ���������Ϣ
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)  # ��ѯ�������ĵĳ˻�
            attended_values.append(attended_value)  # ��ÿ��ͷ�������ӵ��б���

        # �ϲ�����ͷ�����
        aggregated_values = torch.cat(attended_values, dim=1)  # ������ͷ�������ͨ��ƴ��
        reprojected_value = self.reprojection(aggregated_values)  # ʹ�� 1x1 �����ƴ�Ӻ�Ľ������ӳ��
        
        # ��������Ǽ�������Ĳв�����
        attention = reprojected_value + input_

        return attention  # ���ؼ�Ȩ�������ͼ

# ���� EfficientAttention ģ��
if __name__ == '__main__':
    # ����һ�� EfficientAttention ʵ��
    attention = EfficientAttention(in_channels=64, key_channels=128, head_count=4, value_channels=128)
    
    # ����һ���������������ģ����������ͼ (batch_size=1, in_channels=64, height=32, width=32)
    input_tensor = torch.randn(1, 64, 32, 32)

    # ͨ��ע����ģ�����ǰ�򴫲�
    output = attention(input_tensor)
    
    # ��ӡ����������������״
    print(f'������״: {input_tensor.shape}')
    print(f'�����״: {output.shape}')
