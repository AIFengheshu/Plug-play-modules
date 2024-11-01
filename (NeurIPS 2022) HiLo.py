import math
import torch
import torch.nn as nn

# ������Ŀ��Fast Vision Transformers with HiLo Attention
# ������Ŀ:  ����HiLoע�������ƵĿ����Ӿ�Transformers
# �������ӣ�https://arxiv.org/pdf/2205.13213
# �ٷ�github��https://github.com/ziplab/LITv2

class HiLo(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim/num_heads)  # ÿ��ע����ͷ��ͨ��ά��
        self.dim = dim

        # Lo-Fi ע����ͷ������Ƶע������
        self.l_heads = int(num_heads * alpha)
        # Lo-Fi ��ͨ��ά��
        self.l_dim = self.l_heads * head_dim

        # Hi-Fi ע����ͷ������Ƶע������
        self.h_heads = num_heads - self.l_heads
        # Hi-Fi ��ͨ��ά��
        self.h_dim = self.h_heads * head_dim

        # ���ش��ڴ�С���������е� `s` ֵ
        self.ws = window_size

        # �����ڴ�СΪ 1���൱�ڱ�׼�Ķ�ͷ��ע����
        if self.ws == 1:
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        # ���ű��������� qk �˻�ʱ������
        self.scale = qk_scale or head_dim ** -0.5

        # ��Ƶע���� Lo-Fi ��ʼ��
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)  # ƽ���ػ�����
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)  # Lo-Fi ��ѯ����
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)  # Lo-Fi ����ֵ
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)  # Lo-Fi ���ӳ��

        # ��Ƶע���� Hi-Fi ��ʼ��
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)  # Hi-Fi ��ѯ������ֵ
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)  # Hi-Fi ���ӳ��

    # Hi-Fi����Ƶע���������㷽��
    def hifi(self, x):
        B, H, W, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws  # ����ˮƽ�ʹ�ֱ�Ĵ�������

        total_groups = h_group * w_group

        # ����������״��������ά��
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        # �����ѯ������ֵ������Ϊ��ͷ
        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim

        # ����ע�������������й�һ��
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        # ע����Ȩ����ֵ��˵õ����
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)

        x = self.h_proj(x)  # Hi-Fi ӳ��
        return x

    # Lo-Fi����Ƶע���������㷽��
    def lofi(self, x):
        B, H, W, C = x.shape

        # �����ѯ����
        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        if self.ws > 1:
            # �����ڴ�С����1����ػ����룬���������ֵ
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            # ����ֱ�Ӽ������ֵ
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # ����ע�������������й�һ��
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # ����ע������Ȩ���
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        x = self.l_proj(x)  # Lo-Fi ӳ��
        return x

    # ǰ�򴫲�
    def forward(self, x, H, W):
        B, N, C = x.shape

        x = x.reshape(B, H, W, C)  # ����������״

        # ��ʹ�� Lo-Fi
        if self.h_heads == 0:
            x = self.lofi(x)
            return x.reshape(B, N, C)

        # ��ʹ�� Hi-Fi
        if self.l_heads == 0:
            x = self.hifi(x)
            return x.reshape(B, N, C)

        # ͬʱʹ�� Hi-Fi �� Lo-Fi
        hifi_out = self.hifi(x)
        lofi_out = self.lofi(x)

        # ����Ƶ�͵�Ƶ���ƴ��
        x = torch.cat((hifi_out, lofi_out), dim=-1)
        x = x.reshape(B, N, C)

        return x

    # ���� FLOPs�������������
    def flops(self, H, W):
        # ������ͼ�ĸ߶ȺͿ�Ȳ��ܱ����ڴ�С����ʱ���������
        Hp = self.ws * math.ceil(H / self.ws)
        Wp = self.ws * math.ceil(W / self.ws)

        Np = Hp * Wp

        # Hi-Fi ���� FLOPs ����
        hifi_flops = Np * self.dim * self.h_dim * 3
        nW = (Hp // self.ws) * (Wp // self.ws)
        window_len = self.ws * self.ws
        # ���� q @ k �� attn @ v �� FLOPs
        window_flops = window_len * window_len * self.h_dim * 2
        hifi_flops += nW * window_flops
        # ͶӰ FLOPs
        hifi_flops += Np * self.h_dim * self.h_dim

        # Lo-Fi ���� FLOPs ����
        lofi_flops = Np * self.dim * self.l_dim  # q
        kv_len = (Hp // self.ws) * (Wp // self.ws)
        lofi_flops += kv_len * self.dim * self.l_dim * 2  # k, v
        lofi_flops += Np * self.l_dim * kv_len * 2  # q @ k �� attn @ v
        lofi_flops += Np * self.l_dim * self.l_dim  # ͶӰ

        return hifi_flops + lofi_flops

# �����������ڲ���
if __name__ == '__main__':
    hilo = HiLo(dim=128)  # ���� HiLo ����
    input = torch.rand(32, 128, 128)  # �������룬shapeΪ (B, N, C)
    output = hilo(input, 16, 8)       # H = 16, W = 8, H * W Ӧ���� N
    print(input.size())
    print(output.size())