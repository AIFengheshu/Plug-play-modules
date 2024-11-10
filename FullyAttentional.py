# ������Ŀ��Fully Attentional Network for Semantic Segmentation
# �������ӣ�https://arxiv.org/pdf/2112.04108
# �ٷ�github��https://github.com/maggiesong7/FullyAttentional?tab=readme-ov-file

# ����������ע�ͣ����ںţ�AI�����
# AI�����github��https://github.com/AIFengheshu/Plug-play-modules
# https://github.com/Ilareina/FullyAttentional/blob/main/model.py
 
import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyAttentionalBlock(nn.Module):
    def __init__(self, plane, norm_layer=nn.BatchNorm2d):
        # ��ʼ��������plane��������������ͼ��ͨ������norm_layer�ǹ�һ���㣨Ĭ��ΪBatchNorm2d��
        super(FullyAttentionalBlock, self).__init__()
        # ��������ȫ���Ӳ㣬conv1��conv2
        self.conv1 = nn.Linear(plane, plane)
        self.conv2 = nn.Linear(plane, plane)
        
        # �������� + ��һ���� + �������ReLU��
        self.conv = nn.Sequential(
            nn.Conv2d(plane, plane, 3, stride=1, padding=1, bias=False),  # �������
            norm_layer(plane),  # ��һ����
            nn.ReLU()  # ReLU�����
        )
        
        # ����softmax���������ڼ����ϵ����
        self.softmax = nn.Softmax(dim=-1)
        
        # ��ʼ����ѧϰ�Ĳ���gamma�����ڵ������յ����
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # ǰ�򴫲����̣�xΪ���������ͼ����״Ϊ (batch_size, channels, height, width)
        batch_size, _, height, width = x.size()
        
        # �����������������кͱ��Σ���ȡˮƽ�ʹ�ֱ���������
        feat_h = x.permute(0, 3, 1, 2).contiguous().view(batch_size * width, -1, height)  # ˮƽ��������
        feat_w = x.permute(0, 2, 1, 3).contiguous().view(batch_size * height, -1, width)  # ��ֱ��������
        
        # �����������ֱ���ˮƽ����ʹ�ֱ������гػ�����ͨ��ȫ���Ӳ���б���
        encode_h = self.conv1(F.avg_pool2d(x, [1, width]).view(batch_size, -1, height).permute(0, 2, 1).contiguous())  # ˮƽ�������
        encode_w = self.conv2(F.avg_pool2d(x, [height, 1]).view(batch_size, -1, width).permute(0, 2, 1).contiguous())  # ��ֱ�������
        
        # ����ˮƽ����ʹ�ֱ����Ĺ�ϵ����
        energy_h = torch.matmul(feat_h, encode_h.repeat(width, 1, 1))  # ����ˮƽ����Ĺ�ϵ
        energy_w = torch.matmul(feat_w, encode_w.repeat(height, 1, 1))  # ���㴹ֱ����Ĺ�ϵ
        
        # ���㾭��softmax��Ĺ�ϵ����
        full_relation_h = self.softmax(energy_h)  # ˮƽ����Ĺ�ϵ
        full_relation_w = self.softmax(energy_w)  # ��ֱ����Ĺ�ϵ
        
        # ͨ������˷��͹�ϵ���󣬶��������м�Ȩ����ǿ
        full_aug_h = torch.bmm(full_relation_h, feat_h).view(batch_size, width, -1, height).permute(0, 2, 3, 1)  # ˮƽ�������ǿ
        full_aug_w = torch.bmm(full_relation_w, feat_w).view(batch_size, height, -1, width).permute(0, 2, 1, 3)  # ��ֱ�������ǿ
        
        # ��ˮƽ�ʹ�ֱ�������ǿ���������ںϣ�������ԭʼ��������
        out = self.gamma * (full_aug_h + full_aug_w) + x
        
        # ͨ���������н�һ������������
        out = self.conv(out)
        
        return out  # ���ش���������ͼ

if __name__ == "__main__":
    fab = FullyAttentionalBlock(plane=32).cuda()
    # ��������������� (B, C, H, W)
    input_tensor = torch.rand(1, 32, 256, 256).cuda()
    # ��ӡ������������״
    print(f"������������״: {input_tensor.shape}")
    # ǰ�򴫲�
    output_tensor = fab(input_tensor)
    # ��ӡ�����������״
    print(f"�����������״: {output_tensor.shape}")