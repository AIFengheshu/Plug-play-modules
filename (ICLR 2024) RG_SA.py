import torch
from torch import nn
import math

# 论文题目：Recursive Generalization Transformer for Image Super-Resolution
# 中文题目：递归泛化Transformer用于图像超分辨率
# 论文链接：https://arxiv.org/pdf/2303.06373
# 官方github：https://github.com/zhengchen1999/RGT
# 所属机构：上海交通大学，上海人工智能实验室，悉尼大学
# 代码整理：微信公众号《AI缝合术》

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
    # 模块参数
    dim = 64  # 输入特征维度
    num_heads = 8  # 注意力头的数量
    qkv_bias = True  # 是否添加偏置
    qk_scale = None  # 自定义的缩放因子
    attn_drop = 0.1  # 注意力权重的dropout比率
    proj_drop = 0.1  # 输出的dropout比率
    c_ratio = 0.5  # 通道调整比例

    # 输入张量参数
    batch_size = 4  # 批量大小
    seq_length = 256  # 序列长度
    height, width = 16, 16  # 特征图的高和宽

    # 构造随机输入张量 (B, N, C)
    input_tensor = torch.rand(batch_size, seq_length, dim)

    # 实例化 RG_SA 模块
    rg_sa = RG_SA(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  attn_drop=attn_drop, proj_drop=proj_drop, c_ratio=c_ratio)

    # 模块处于训练模式
    rg_sa.train()

    # 将输入张量传递给 RG_SA 模块
    output = rg_sa(input_tensor, height, width)

    # 打印输出张量的形状
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output.shape}")