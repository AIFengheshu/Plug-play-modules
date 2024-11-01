import math
import torch
import torch.nn as nn

# 论文题目：Fast Vision Transformers with HiLo Attention
# 中文题目:  带有HiLo注意力机制的快速视觉Transformers
# 论文链接：https://arxiv.org/pdf/2205.13213
# 官方github：https://github.com/ziplab/LITv2

class HiLo(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim/num_heads)  # 每个注意力头的通道维度
        self.dim = dim

        # Lo-Fi 注意力头数（低频注意力）
        self.l_heads = int(num_heads * alpha)
        # Lo-Fi 的通道维度
        self.l_dim = self.l_heads * head_dim

        # Hi-Fi 注意力头数（高频注意力）
        self.h_heads = num_heads - self.l_heads
        # Hi-Fi 的通道维度
        self.h_dim = self.h_heads * head_dim

        # 本地窗口大小，即论文中的 `s` 值
        self.ws = window_size

        # 若窗口大小为 1，相当于标准的多头自注意力
        if self.ws == 1:
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        # 缩放比例，用于 qk 乘积时的缩放
        self.scale = qk_scale or head_dim ** -0.5

        # 低频注意力 Lo-Fi 初始化
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)  # 平均池化操作
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)  # Lo-Fi 查询向量
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)  # Lo-Fi 键和值
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)  # Lo-Fi 输出映射

        # 高频注意力 Hi-Fi 初始化
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)  # Hi-Fi 查询、键和值
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)  # Hi-Fi 输出映射

    # Hi-Fi（高频注意力）计算方法
    def hifi(self, x):
        B, H, W, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws  # 计算水平和垂直的窗口组数

        total_groups = h_group * w_group

        # 重塑输入形状，并调整维度
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        # 计算查询、键和值，并分为多头
        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim

        # 计算注意力分数并进行归一化
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        # 注意力权重与值相乘得到输出
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)

        x = self.h_proj(x)  # Hi-Fi 映射
        return x

    # Lo-Fi（低频注意力）计算方法
    def lofi(self, x):
        B, H, W, C = x.shape

        # 计算查询向量
        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        if self.ws > 1:
            # 若窗口大小大于1，则池化输入，并计算键和值
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            # 否则直接计算键和值
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # 计算注意力分数并进行归一化
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 计算注意力加权输出
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        x = self.l_proj(x)  # Lo-Fi 映射
        return x

    # 前向传播
    def forward(self, x, H, W):
        B, N, C = x.shape

        x = x.reshape(B, H, W, C)  # 重塑输入形状

        # 仅使用 Lo-Fi
        if self.h_heads == 0:
            x = self.lofi(x)
            return x.reshape(B, N, C)

        # 仅使用 Hi-Fi
        if self.l_heads == 0:
            x = self.hifi(x)
            return x.reshape(B, N, C)

        # 同时使用 Hi-Fi 和 Lo-Fi
        hifi_out = self.hifi(x)
        lofi_out = self.lofi(x)

        # 将高频和低频输出拼接
        x = torch.cat((hifi_out, lofi_out), dim=-1)
        x = x.reshape(B, N, C)

        return x

    # 计算 FLOPs（浮点操作数）
    def flops(self, H, W):
        # 当特征图的高度和宽度不能被窗口大小整除时，进行填充
        Hp = self.ws * math.ceil(H / self.ws)
        Wp = self.ws * math.ceil(W / self.ws)

        Np = Hp * Wp

        # Hi-Fi 部分 FLOPs 计算
        hifi_flops = Np * self.dim * self.h_dim * 3
        nW = (Hp // self.ws) * (Wp // self.ws)
        window_len = self.ws * self.ws
        # 计算 q @ k 和 attn @ v 的 FLOPs
        window_flops = window_len * window_len * self.h_dim * 2
        hifi_flops += nW * window_flops
        # 投影 FLOPs
        hifi_flops += Np * self.h_dim * self.h_dim

        # Lo-Fi 部分 FLOPs 计算
        lofi_flops = Np * self.dim * self.l_dim  # q
        kv_len = (Hp // self.ws) * (Wp // self.ws)
        lofi_flops += kv_len * self.dim * self.l_dim * 2  # k, v
        lofi_flops += Np * self.l_dim * kv_len * 2  # q @ k 和 attn @ v
        lofi_flops += Np * self.l_dim * self.l_dim  # 投影

        return hifi_flops + lofi_flops

# 主函数，用于测试
if __name__ == '__main__':
    hilo = HiLo(dim=128)  # 定义 HiLo 对象
    input = torch.rand(32, 128, 128)  # 定义输入，shape为 (B, N, C)
    output = hilo(input, 16, 8)       # H = 16, W = 8, H * W 应等于 N
    print(input.size())
    print(output.size())