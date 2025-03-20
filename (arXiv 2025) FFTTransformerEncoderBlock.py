import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # Generate binary tensor mask; shape: (batch_size, 1, 1, ..., 1)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    DropPath module that performs stochastic depth.
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class MultiHeadSpectralAttention(nn.Module):
    def __init__(self, embed_dim, seq_len, num_heads=4, dropout=0.1, adaptive=True):
        """
        频谱注意力模块,在保持 O(n log n) 计算复杂度的同时,引入额外的非线性和自适应能力。
        参数:
          - embed_dim: 总的嵌入维度。
          - seq_len: 序列长度(例如 Transformer 中 token 的数量,包括类 token)。
          - num_heads: 注意力头的数量。
          - dropout: 逆傅里叶变换(iFFT)后的 dropout 率。
          - adaptive: 是否启用自适应 MLP 以生成乘法和加法的自适应调制参数。
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len
        self.adaptive = adaptive

        # 频域的 FFT 频率桶数量: (seq_len//2 + 1)
        self.freq_bins = seq_len // 2 + 1

        # 基础乘法滤波器: 每个注意力头和频率桶一个
        self.base_filter = nn.Parameter(torch.ones(num_heads, self.freq_bins, 1))
        # 基础加性偏置: 作为频率幅度的学习偏移
        self.base_bias = nn.Parameter(torch.full((num_heads, self.freq_bins, 1), -0.1))

        if adaptive:
            # 自适应 MLP: 每个头部和频率桶生成 2 个值(缩放因子和偏置)
            self.adaptive_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_heads * self.freq_bins * 2)
            )
        self.dropout = nn.Dropout(dropout)
        # 预归一化层,提高傅里叶变换的稳定性
        self.pre_norm = nn.LayerNorm(embed_dim)

    def complex_activation(self, z):
        """
        对复数张量应用非线性激活函数。该函数计算 z 的幅度,将其传递到 GELU 进行非线性变换,并按比例缩放 z,以保持相位不变。
        参数:
          z: 形状为 (B, num_heads, freq_bins, head_dim) 的复数张量
        返回:
          经过非线性变换的复数张量,形状相同。
        """
        mag = torch.abs(z)
        # 对幅度进行非线性变换,GELU 提供平滑的非线性
        mag_act = F.gelu(mag)
        # 计算缩放因子,防止除零错误
        scale = mag_act / (mag + 1e-6)
        return z * scale

    def forward(self, x):
        """
        增强型频谱注意力模块的前向传播。
        参数:
        x: 输入张量,形状为 (B, seq_len, embed_dim)
        返回:经过频谱调制和残差连接的张量,形状仍为 (B, seq_len, embed_dim)
        """
        B, N, D = x.shape

        # 预归一化,提高频域变换的稳定性
        x_norm = self.pre_norm(x)

        # 重新排列张量以分离不同的注意力头,形状变为 (B, num_heads, seq_len, head_dim)
        x_heads = x_norm.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 沿着序列维度计算 FFT,结果为复数张量,形状为 (B, num_heads, freq_bins, head_dim)
        F_fft = torch.fft.rfft(x_heads, dim=2, norm='ortho')

        # 计算自适应调制参数(如果启用)
        if self.adaptive:
            # 全局上下文:对 token 维度求均值,形状为 (B, embed_dim)
            context = x_norm.mean(dim=1)
            # 经过 MLP 计算自适应参数,输出形状为 (B, num_heads*freq_bins*2)
            adapt_params = self.adaptive_mlp(context)
            adapt_params = adapt_params.view(B, self.num_heads, self.freq_bins, 2)
            # 划分为乘法缩放因子和加法偏置
            adaptive_scale = adapt_params[..., 0:1]  # 形状: (B, num_heads, freq_bins, 1)
            adaptive_bias  = adapt_params[..., 1:2]  # 形状: (B, num_heads, freq_bins, 1)
        else:
            # 如果不使用自适应机制,则缩放因子和偏置设为 0
            adaptive_scale = torch.zeros(B, self.num_heads, self.freq_bins, 1, device=x.device)
            adaptive_bias  = torch.zeros(B, self.num_heads, self.freq_bins, 1, device=x.device)

        # 结合基础滤波器和自适应调制参数
        # effective_filter: 影响频谱响应的缩放因子
        effective_filter = self.base_filter * (1 + adaptive_scale)
        # effective_bias: 影响频谱响应的偏置
        effective_bias = self.base_bias + adaptive_bias

        # 在频域进行自适应调制
        # 先进行乘法缩放,再添加偏置(在 head_dim 维度上广播)
        F_fft_mod = F_fft * effective_filter + effective_bias

        # 在频域应用非线性激活
        F_fft_nl = self.complex_activation(F_fft_mod)

        # 逆傅里叶变换(iFFT)还原到时序空间
        # 需要指定 n=self.seq_len 以确保输出长度匹配输入
        x_filtered = torch.fft.irfft(F_fft_nl, dim=2, n=self.seq_len, norm='ortho')
        # 重新排列张量,将注意力头合并回嵌入维度
        x_filtered = x_filtered.permute(0, 2, 1, 3).reshape(B, N, D)

        # 残差连接并应用 Dropout
        return x + self.dropout(x_filtered)

class FFTTransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1, attention_module=None, drop_path=0.0):
        """
        一个通用的 Transformer 编码器块，集成了 drop path 随机深度 。
          - embed_dim: 嵌入维度。
          - mlp_ratio: MLP 的扩展因子。
          - dropout: dropout 比率。
          - attention_module: 处理自注意力的模块。
          - drop_path: 随机深度的 drop path 比率。
        """
        super().__init__()
        if attention_module is None:
            raise ValueError("必须提供一个注意力模块! 此处应调用 MultiHeadSpectralAttention")
        self.attention = attention_module
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(embed_dim)
        # 用于随机深度的 drop path 层
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # 在残差连接中应用带有 drop path 的注意力。
        x = x + self.drop_path(self.attention(x))
        # 在残差连接中应用 MLP（经过层归一化）并加入 drop path。
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x
    
if __name__ == '__main__':
    # 参数设置
    batch_size = 1      # 批大小
    seq_len = 224 * 224 # 序列长度
    embed_dim = 32      # 嵌入维度
    num_heads = 4       # 注意力头数

    # 创建随机输入张量 (batch_size, seq_len, embed_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)

    # 初始化 MultiHeadSpectralAttention
    attention_module = MultiHeadSpectralAttention(embed_dim=embed_dim, seq_len=seq_len, num_heads=num_heads)

    # 初始化 TransformerEncoderBlock
    transformer_block = FFTTransformerEncoderBlock(embed_dim=embed_dim, attention_module=attention_module)
    print(transformer_block)
    print("微信公众号: AI缝合术!")

    # 前向传播测试
    output = transformer_block(x)

    # 打印输出形状
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)    
