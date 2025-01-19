import torch
import torch.nn as nn

# 论文题目：Conv2Former: A Simple Transformer-Style ConvNet for Visual Recognition
# 中文题目：Conv2Former: 一种简单的视觉识别用的Transformer风格卷积网络
# 论文链接：https://arxiv.org/pdf/2211.11943
# 官方github：https://github.com/HVision-NKU/Conv2Former
# 所属机构：天津南开大学计算机科学学院，字节跳动（新加坡）
# 代码整理：微信公众号《AI缝合术》
# 全部即插即用模块代码：https://github.com/AIFengheshu/Plug-play-modules

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        

class SpatialAttention(nn.Module):
    def __init__(self, dim, kernel_size, expand_ratio=2):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.att = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)        
        x = self.att(x) * self.v(x)
        x = self.proj(x)
        return x

if __name__ == "__main__":

    # 将模块移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建测试输入张量 (batch_size, channels, height, width)
    x = torch.randn(1, 32, 256, 256).to(device)
    # 初始化 SpatialAtt 模块
    SpatialAtt=SpatialAttention(dim=32, kernel_size=3)
    print(SpatialAtt)
    print("微信公众号:AI缝合术")
    SpatialAtt = SpatialAtt.to(device)
    # 前向传播
    output = SpatialAtt(x)
    
    # 打印输入和输出张量的形状
    print("输入张量形状:", x.shape)
    print("输出张量形状:", output.shape)
