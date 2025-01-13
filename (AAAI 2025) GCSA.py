import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum

# ������Ŀ��Rethinking Transformer-Based Blind-Spot Network for Self-Supervised Image Denoising
# ������Ŀ������Transformer��ä�������Լලͼ��ȥ�����˼��
# �������ӣ�https://arxiv.org/pdf/2404.07846
# �ٷ�github��https://github.com/nagejacob/TBSN
# ������������������ҵ��ѧ
# ��������΢�Ź��ںš�AI�������
   
class GCSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(GCSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out
    
if __name__ == "__main__":

    # ��ģ���ƶ��� GPU��������ã�
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ���������������� (batch_size, channels, height, width)
    x = torch.randn(1, 32, 256, 256).to(device)

    # ��ʼ�� GCSA ģ��
    gcsa = GCSA(dim=32, num_heads=4, bias=True)
    print(gcsa)
    gcsa = gcsa.to(device)

    # ǰ�򴫲�
    output = gcsa(x)

    # ��ӡ����������������״
    print("����������״:", x.shape)
    print("���������״:", output.shape)
