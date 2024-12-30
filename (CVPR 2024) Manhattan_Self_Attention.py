import torch
from torch import nn
from typing import Tuple

# 论文题目：RMT: Retentive Networks Meet Vision Transformers
# 中文题目：RMT：保留网络与视觉变压器相遇
# 论文链接：https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_RMT_Retentive_Networks_Meet_Vision_Transformers_CVPR_2024_paper.pdf
# 官方github：https://github.com/qhfan/RMT
# 所属机构：中国科学院自动化研究所 & CRIPAC，中国科学院大学人工智能学院，北京科技大学
# 代码整理：微信公众号《AI缝合术》

def rotate_every_two(x):
    # 对输入张量的最后一个维度，每隔两个元素分组，将其中一个分量进行反转
    x1 = x[:, :, :, :, ::2]  # 选取每隔两个元素的第一个
    x2 = x[:, :, :, :, 1::2]  # 选取每隔两个元素的第二个
    x = torch.stack([-x2, x1], dim=-1)  # 交换顺序并按维度堆叠
    return x.flatten(-2)  # 将最后两个维度合并成一个维度

def theta_shift(x, sin, cos):
    # 应用旋转变换，基于正弦和余弦调整输入张量
    return (x * cos) + (rotate_every_two(x) * sin)

class DWConv2d(nn.Module):
    # 深度可分离卷积类
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        # 定义深度卷积：每个通道独立处理
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c) 输入张量的形状为 (批次, 高度, 宽度, 通道数)
        '''
        x = x.permute(0, 3, 1, 2)  # 转置为 (b c h w) 以适配卷积操作
        x = self.conv(x)  # 应用深度卷积
        x = x.permute(0, 2, 3, 1)  # 转置回原始形状 (b h w c)
        return x

class RetNetRelPos2d(nn.Module):
    # RetNet 的二维相对位置编码模块
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        initial_value: 初始衰减值
        heads_range: 衰减范围
        '''
        super().__init__()
        # 初始化角度参数，用于相对位置的正弦和余弦编码
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()  # 重复以获得正弦和余弦部分
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads

        # 初始化衰减参数
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_2d_decay(self, H: int, W: int):
        '''
        生成二维的衰减掩码
        H, W: 张量的高度和宽度
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])  # 创建网格
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  # 重塑为二维坐标
        mask = grid[:, None, :] - grid[None, :, :]  # 计算相对位置差
        mask = (mask.abs()).sum(dim=-1)  # 计算曼哈顿距离
        mask = mask * self.decay[:, None, None]  # 加权衰减
        return mask

    def generate_1d_decay(self, l: int):
        '''
        生成一维的衰减掩码
        l: 序列长度
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]  # 计算相对位置差
        mask = mask.abs()  # 取绝对值
        mask = mask * self.decay[:, None, None]  # 加权衰减
        return mask

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w) 输入张量的高度和宽度
        activate_recurrent: 是否启用循环计算
        chunkwise_recurrent: 是否启用分块循环计算
        '''
        if activate_recurrent:
            # 循环模式：基于总长度生成正弦和余弦
            sin = torch.sin(self.angle * (slen[0] * slen[1] - 1))
            cos = torch.cos(self.angle * (slen[0] * slen[1] - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())

        elif chunkwise_recurrent:
            # 分块模式：对每块单独生成正弦和余弦，以及一维掩码
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(slen[0], slen[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(slen[0], slen[1], -1)
            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])
            retention_rel_pos = ((sin, cos), (mask_h, mask_w))

        else:
            # 普通模式：生成正弦、余弦和二维掩码
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(slen[0], slen[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(slen[0], slen[1], -1)
            mask = self.generate_2d_decay(slen[0], slen[1])
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos

class VisionRetentionChunk(nn.Module):
    # Vision Retention 模块，用于实现视觉特征的注意力机制
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5  # 缩放因子，用于规范化
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)  # 查询投影层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)  # 键值投影层
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)  # 值投影层
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)  # 局部增强模块
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)  # 输出投影层
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化权重参数
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c) 输入张量
        rel_pos: 相对位置编码
        '''
        bsz, h, w, _ = x.size()
        (sin, cos), (mask_h, mask_w) = rel_pos

        # 计算查询、键和值向量
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)  # 计算局部增强特征

        # 对查询和键进行缩放和旋转变换
        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        # 计算宽度方向上的注意力
        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)
        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        qk_mat_w = qk_mat_w + mask_w
        qk_mat_w = torch.softmax(qk_mat_w, -1)
        v = torch.matmul(qk_mat_w, v)

        # 计算高度方向上的注意力
        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)
        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        qk_mat_h = qk_mat_h + mask_h
        qk_mat_h = torch.softmax(qk_mat_h, -1)
        output = torch.matmul(qk_mat_h, v)

        # 恢复输出形状并添加局部增强特征
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

if __name__ == '__main__':

    # 生成随机输入数据，注意输入张量顺序为B H W C
    input_data = torch.randn(1, 256, 256, 32)  # 输入张量
    b, h, w, c = input_data.size()
    pos = RetNetRelPos2d(embed_dim=32, num_heads=4, initial_value=1, heads_range=3)  # 初始化位置编码
    rel_pos = pos((h, w), chunkwise_recurrent=True)  # 计算相对位置编码

    retention = VisionRetentionChunk(embed_dim=32, num_heads=4)  # 初始化模型
    output = retention(input_data, rel_pos)

    # 打印输入和输出形状
    print("Input size:", input_data.size())
    print("Output size:", output.size())
