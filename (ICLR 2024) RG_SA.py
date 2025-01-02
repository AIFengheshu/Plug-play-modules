import torch
from torch import nn
import math

# ������Ŀ��Recursive Generalization Transformer for Image Super-Resolution
# ������Ŀ���ݹ鷺��Transformer����ͼ�񳬷ֱ���
# �������ӣ�https://arxiv.org/pdf/2303.06373
# �ٷ�github��https://github.com/zhengchen1999/RGT
# �����������Ϻ���ͨ��ѧ���Ϻ��˹�����ʵ���ң�Ϥ���ѧ
# ��������΢�Ź��ںš�AI�������

class RG_SA(nn.Module):
    """
    Recursive-Generalization Self-Attention (RG-SA).
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        c_ratio (float): channel adjustment factor.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., c_ratio=0.5):
        super(RG_SA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.cr = int(dim * c_ratio) # scaled channel dimension

        # self.scale = qk_scale or head_dim ** -0.5
        self.scale = qk_scale or (head_dim * c_ratio) ** -0.5

        # RGM
        self.reduction1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4, groups=dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)
        self.conv = nn.Conv2d(dim, self.cr, kernel_size=1, stride=1)
        self.norm_act = nn.Sequential(
            nn.LayerNorm(self.cr),
            nn.GELU())
        # CA
        self.q = nn.Linear(dim, self.cr, bias=qkv_bias)
        self.k = nn.Linear(self.cr, self.cr, bias=qkv_bias)
        self.v = nn.Linear(self.cr, dim, bias=qkv_bias)

        # CPE
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape

        _scale = 1

        # reduction
        _x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        if self.training:
            _time = max(int(math.log(H//4, 4)), int(math.log(W//4, 4)))
        else:
            _time = max(int(math.log(H//16, 4)), int(math.log(W//16, 4)))
            if _time < 2: _time = 2 # testing _time must equal or larger than training _time (2)

        _scale = 4 ** _time

        # Recursion xT
        for _ in range(_time):
            _x = self.reduction1(_x)

        _x = self.conv(self.dwconv(_x)).reshape(B, self.cr, -1).permute(0, 2, 1).contiguous()  # shape=(B, N', C')
        _x = self.norm_act(_x)

        # q, k, v, where q_shape=(B, N, C'), k_shape=(B, N', C'), v_shape=(B, N', C)
        q = self.q(x).reshape(B, N, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
        k = self.k(_x).reshape(B, -1, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
        v = self.v(_x).reshape(B, -1, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)
        
        # corss-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # CPE
        # v_shape=(B, H, N', C//H)
        v = v + self.cpe(v.transpose(1, 2).reshape(B, -1, C).transpose(1, 2).contiguous().view(B, C, H // _scale, W // _scale)).view(B, C, -1).view(B, self.num_heads, int(C / self.num_heads), -1).transpose(-1, -2)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

if __name__ == "__main__":
    # ģ�����
    dim = 64  # ��������ά��
    num_heads = 8  # ע����ͷ������
    qkv_bias = True  # �Ƿ����ƫ��
    qk_scale = None  # �Զ������������
    attn_drop = 0.1  # ע����Ȩ�ص�dropout����
    proj_drop = 0.1  # �����dropout����
    c_ratio = 0.5  # ͨ����������

    # ������������
    batch_size = 4  # ������С
    seq_length = 256  # ���г���
    height, width = 16, 16  # ����ͼ�ĸߺͿ�

    # ��������������� (B, N, C)
    input_tensor = torch.rand(batch_size, seq_length, dim)

    # ʵ���� RG_SA ģ��
    rg_sa = RG_SA(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  attn_drop=attn_drop, proj_drop=proj_drop, c_ratio=c_ratio)

    # ģ�鴦��ѵ��ģʽ
    rg_sa.train()

    # �������������ݸ� RG_SA ģ��
    output = rg_sa(input_tensor, height, width)

    # ��ӡ�����������״
    print(f"����������״: {input_tensor.shape}")
    print(f"���������״: {output.shape}")