import torch
from torch import Tensor,nn

# 论文题目：OrthoNets : Orthogonal Channel Attention Networks
# 中文题目：OrthoNets : 正交通道注意力网络
# 论文链接：https://arxiv.org/pdf/2311.03071
# 官方github：https://github.com/hady1011/OrthoNets
# 所属机构：阿肯色大学计算机科学与计算机工程系
# 代码整理：微信公众号《AI缝合术》
# 注：以下代码由源码优化整理得到，比源码逻辑更清晰，运行更流畅，且实现了完整功能，源码较为散乱，详细请见源码

def gram_schmidt(input):
    def projection(u, v):
        return (torch.dot(u.view(-1), v.view(-1)) / torch.dot(u.view(-1), u.view(-1))) * u
    output = []
    for x in input:
        for y in output:
            x = x - projection(y, x)
        x = x / x.norm(p=2)
        output.append(x)
    return torch.stack(output)

def initialize_orthogonal_filters(c, h, w):
    if h * w < c:
        n = c // (h * w)
        gram = []
        for i in range(n):
            gram.append(gram_schmidt(torch.rand([h * w, 1, h, w])))
        return torch.cat(gram, dim=0)
    else:
        return gram_schmidt(torch.rand([c, 1, h, w]))

class GramSchmidtTransform(torch.nn.Module):
    instance = {}  # 初始化字典
    constant_filter: Tensor

    @staticmethod
    def build(c: int, h: int):
        if (c, h) not in GramSchmidtTransform.instance:
            GramSchmidtTransform.instance[(c, h)] = GramSchmidtTransform(c, h)
        return GramSchmidtTransform.instance[(c, h)]

    def __init__(self, c: int, h: int):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            rand_ortho_filters = initialize_orthogonal_filters(c, h, h).view(c, h, h)
        self.register_buffer("constant_filter", rand_ortho_filters.to(self.device).detach())

    def forward(self, x):
        _, _, h, w = x.shape
        _, H, W = self.constant_filter.shape
        if h != H or w != W: 
            x = torch.nn.functional.adaptive_avg_pool2d(x, (H, W))
        return (self.constant_filter * x).sum(dim=(-1, -2), keepdim=True)

# class Orthogonal_Channel_Attention(torch.nn.Module):
#     def __init__(self, c: int, h:int):
#         super().__init__()
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.FWT = GramSchmidtTransform.build(c, h)  # 初始化 FWT

#     def forward(self, input: Tensor):
#         x = input
#         while input[0].size(-1) > 1:
#             input = self.FWT(input.to(self.device))
#         b = input.size(0)
#         return input.view(b, -1)
    
class Orthogonal_Channel_Attention(nn.Module):
    def __init__(self, channels: int, height: int):
        """
        初始化Orthogonal_Channel_Attention模块
        :param channels: 输入张量的通道数 (C)
        :param height: Gram-Schmidt 变换所需的高度 (假设 H=W)
        """
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.channels = channels
        self.height = height

        # Gram-Schmidt 变换初始化
        self.F_C_A = GramSchmidtTransform.build(channels, height)
        
        # 通道注意力映射（SE Block 结构）
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        :param x: 输入张量 (B, C, H, W)
        :return: 输出张量 (B, C, H, W)
        """
        B, C, H, W = x.shape
        # 如果输入的 H 和 W 与初始化不匹配，则自适应池化
        if H != self.height or W != self.height:
            x = nn.functional.adaptive_avg_pool2d(x, (self.height, self.height))
        # Gram-Schmidt 变换
        transformed = self.F_C_A(x)  # (B, C, 1, 1)
        # 去除空间维度，进入通道注意力网络
        compressed = transformed.view(B, C)
        # 通道注意力生成
        excitation = self.channel_attention(compressed).view(B, C, 1, 1)
        # 加权原始输入特征
        output = x * excitation
        return output        

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化 Orthogonal_Channel_Attention
    channels = 32
    height = 256
    attention_module = Orthogonal_Channel_Attention(channels, height).to(device)
    # 输入张量 (B, C, H, W)
    input_tensor = torch.rand(1, channels, 256, 256).to(device)
    # 前向传播
    output_tensor = attention_module(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")

