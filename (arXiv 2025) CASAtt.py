import torch
import torch.nn as nn

# ������Ŀ��CAS-ViT: Convolutional Additive Self-attention Vision Transformers for Efficient Mobile Applications
# ������Ŀ��CAS-ViT�����ڸ�Ч�ƶ�Ӧ�õľ��������ע�����Ӿ�Transformer
# �������ӣ�https://arxiv.org/pdf/2408.03703?
# �ٷ�github��https://github.com/Tianfang-Zhang/CAS-ViT
# ���������������Ƽ��о�Ժ���廪��ѧ�Զ���ϵ����ʢ�ٴ�ѧ��������������ϵ���籾������ѧ�������ѧϵ
# ��������΢�Ź��ںš�AI�������
# ȫ�����弴��ģ����룺https://github.com/AIFengheshu/Plug-play-modules

class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)
    
class CASAtt(nn.Module):

    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out


if __name__ == "__main__":

    # ��ģ���ƶ��� GPU��������ã�
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ���������������� (batch_size, channels, height, width)
    x = torch.randn(1, 32, 256, 256).to(device)

    # ��ʼ�� casatt ģ��
    casatt=CASAtt(dim=32)
    print(casatt)
    print("΢�Ź��ں�:AI�����")
    casatt = casatt.to(device)

    # ǰ�򴫲�
    output = casatt(x)
    
    # ��ӡ����������������״
    print("����������״:", x.shape)
    print("���������״:", output.shape)
