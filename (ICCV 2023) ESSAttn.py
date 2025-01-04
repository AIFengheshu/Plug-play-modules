import torch
import torch.nn as nn
import math

# ������Ŀ��ESSAformer: Efficient Transformer for Hyperspectral Image Super-resolution
# ������Ŀ��ESSAformer�����ڸ߹���ͼ�񳬷ֱ��ʵĸ�Ч�任��
# �������ӣ�https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_ESSAformer_Efficient_Transformer_for_Hyperspectral_Image_Super-resolution_ICCV_2023_paper.pdf
# �ٷ�github��https://github.com/Rexzhan/ESSAformer
# �����������������ӿƼ���ѧ��Ϥ���ѧ�������ʵ��ѧ
# ��������΢�Ź��ںš�AI�������

class ESSAttn(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.lnqkv = nn.Linear(dim, dim * 3)
        self.ln = nn.Linear(dim, dim)

    def forward(self, x):
        b,c,h,w = x.shape
        x = x.reshape(b,c,h*w).permute(0,2,1)
        b, N, C = x.shape
        qkv = self.lnqkv(x)
        qkv = torch.split(qkv, C, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.mean(q, dim=2, keepdim=True)
        q = q - a
        a = torch.mean(k, dim=2, keepdim=True)
        k = k - a
        q2 = torch.pow(q, 2)
        q2s = torch.sum(q2, dim=2, keepdim=True)
        k2 = torch.pow(k, 2)
        k2s = torch.sum(k2, dim=2, keepdim=True)
        t1 = v
        k2 = torch.nn.functional.normalize((k2 / (k2s + 1e-7)), dim=-2)
        q2 = torch.nn.functional.normalize((q2 / (q2s + 1e-7)), dim=-1)
        t2 = q2 @ (k2.transpose(-2, -1) @ v) / math.sqrt(N)
        attn = t1 + t2
        attn = self.ln(attn)
        x = attn.reshape(b,h,w,c).permute(0,3,1,2)
        return x
    
if __name__ == '__main__':
    input = torch.randn(1,32,256,256)
    essa = ESSAttn(32)
    output = essa(input)
    print(essa)
    print(input.size())
    print(output.size())