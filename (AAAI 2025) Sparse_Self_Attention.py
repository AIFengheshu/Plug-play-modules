import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_, DropPath, to_2tuple

# 论文题目：我们可以摆脱手工特征提取器吗？
# SparseViT：以非语义为中心，参数高效的图像操纵通过稀疏编码变换器进行定位
# 中文题目：Can We Get Rid of Handcrafted Feature Extractors?
# SparseViT: Nonsemantics-Centered, Parameter-Efficient Image Manipulation Localization through Spare-Coding Transformer
# 论文链接：https://arxiv.org/pdf/2412.14598
# 官方github：https://github.com/scu-zjz/SparseViT
# 所属机构：四川大学计算机科学学院，中国教育部机器学习与工业智能工程研究中心，穆罕默德·本·扎耶德人工智能大学，澳门大学计算机与信息科学系
# 代码整理：微信公众号《AI缝合术》

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # print(x.shape)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

  
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def block(x,block_size):
    B,H,W,C = x.shape
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  
    Hp, Wp = H + pad_h, W + pad_w  
    x = x.reshape(B,Hp//block_size,block_size,Wp//block_size,block_size, C)
    x = x.permute(0,1,3,2,4,5).contiguous()
    return x, H, Hp, C

def unblock(x, Ho):
    B,H,W,win_H,win_W,C = x.shape
    x = x.permute(0,1,3,2,4,5).contiguous().reshape(B,H*win_H,W*win_W, C)
    Wp = Hp = H*win_H
    Wo = Ho
    if Hp > Ho or Wp > Wo:
        x = x[:, :Ho, :Wo, :].contiguous()
    return x


def alter_sparse(x, sparse_size=8):
    x = x.permute(0, 2, 3, 1)
    assert x.shape[1]%sparse_size == 0 & x.shape[2]%sparse_size == 0, 'image size should be divisible by block_size'
    grid_size = x.shape[1]//sparse_size
    out, H, Hp, C = block(x, grid_size)
    out = out.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = out.reshape(-1, sparse_size, sparse_size, C)
    out = out.permute(0, 3, 1, 2)
    return out, H, Hp, C   


def alter_unsparse(x, H, Hp, C, sparse_size=8):
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(-1, Hp//sparse_size, Hp//sparse_size, sparse_size, sparse_size, C)
    x = x.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = unblock(x, H)
    out = out.permute(0, 3, 1, 2)
    return out

class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class  Sparse_Self_Attention(nn.Module):
    def __init__(self, dim, num_heads, sparse_size=0, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        global layer_scale
        self.ls = layer_scale
        self.sparse_size = sparse_size
        if self.ls:
            global init_value
            print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x_befor = x.flatten(2).transpose(1, 2)
        B, N, H, W = x.shape
        if self.ls:
            x, Ho, Hp, C = alter_sparse(x, self.sparse_size)
            Bf, Nf, Hf, Wf = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.attn(self.norm1(x))
            x = x.transpose(1, 2).reshape(Bf, Nf, Hf, Wf)
            x = alter_unsparse(x, Ho, Hp, C, self.sparse_size)
            x = x.flatten(2).transpose(1, 2)  
            x = x_befor + self.drop_path(self.gamma_1 * x)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), H, W))
        else:
            x, Ho, Hp, C = alter_sparse(x, self.sparse_size)
            Bf, Nf, Hf, Wf = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.attn(self.norm1(x))
            x = x.transpose(1, 2).reshape(Bf, Nf, Hf, Wf)
            x = alter_unsparse(x, Ho, Hp, C, self.sparse_size)
            x = x.flatten(2).transpose(1, 2)
            x = x_befor + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x        


# 设置全局变量
layer_scale = True
init_value = 1e-6

if __name__ == '__main__':
    # 定义输入张量大小（批量大小、通道数、高度、宽度）
    B, C, H, W = 1, 64, 256, 256  # 可以根据需要调整形状
    input_tensor = torch.randn(B, C, H, W)  # 随机生成输入张量

    # 初始化 SABlock
    dim = C  # 输入和输出通道数
    num_heads = 4  # 注意力头的数量
    sparse_size = 4  # 稀疏处理块大小
    mlp_ratio = 4.0  # MLP 隐藏层的放大比例
    qkv_bias = True  # 是否使用 QKV 偏置
    drop = 0.1  # dropout 概率
    attn_drop = 0.1  # 注意力 dropout 概率
    drop_path = 0.1  # DropPath 概率

    # 创建 Sparse_Self_Attention 实例
    sablock = Sparse_Self_Attention(
        dim=dim,
        num_heads=num_heads,
        sparse_size=sparse_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop=drop,
        attn_drop=attn_drop,
        drop_path=drop_path
    )

    # 将模块移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sablock = sablock.to(device)
    print(sablock)
    input_tensor = input_tensor.to(device)

    # 执行前向传播
    output = sablock(input_tensor)

    # 打印输入和输出的形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")