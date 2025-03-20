import torch
import torch.nn as nn
import torch.nn.functional as F

class ModReLU(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.b = nn.Parameter(torch.Tensor(features))
        self.b.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        return torch.abs(x) * F.relu(torch.cos(torch.angle(x) + self.b))

class FFTNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.filter = nn.Linear(dim, dim)
        self.modrelu = ModReLU(dim)

    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        x_fft = torch.fft.fft(x, dim=1)  # FFT along the sequence dimension
        x_filtered = self.filter(x_fft.real) + 1j * self.filter(x_fft.imag)
        x_filtered = self.modrelu(x_filtered)
        x_out = torch.fft.ifft(x_filtered, dim=1).real
        return x_out

    
if __name__ == '__main__':
    # 参数设置
    batch_size = 1      # 批量大小
    seq_len = 224 * 224 # 序列长度(Transformer 中的 token 数量)
    dim = 32      # 维度


    # 创建随机输入张量,形状为 (batch_size, seq_len, embed_dim)
    x = torch.randn(batch_size, seq_len, dim)

    # 初始化 MultiHeadSpectralAttention 模块
    model = FFTNetBlock(dim = dim)
    print(model)
    print("微信公众号: AI缝合术!")

    output = model(x)
    print(x.shape)
    print(output.shape) 