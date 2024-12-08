import torch
from torch import nn
import einops

# 论文题目：SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications
# 中文题目: SwiftFormer：针对基于Transformer的实时移动视觉应用的高效加性注意力
# 论文链接：https://openaccess.thecvf.com/content/ICCV2023/papers/Shaker_SwiftFormer_Efficient_Additive_Attention_for_Transformer-based_Real-time_Mobile_Vision_Applications_ICCV_2023_paper.pdf
# 官方github：https://github.com/Amshaker/SwiftFormer
# 所属机构：穆罕默德·本·扎耶德AI大学、加州大学默塞德分校、延世大学、谷歌研究中心、Link?ping大学
# 关键词：SwiftFormer, Transformer, 自注意力, 移动视觉应用, 实时性能, 混合设计
# 微信公众号：AI缝合术

class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """
    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()
        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)
        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)
        query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
        key = torch.nn.functional.normalize(key, dim=-1) #BxNxD
        query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor # BxNx1
        A = torch.nn.functional.normalize(A, dim=1) # BxNx1
        G = torch.sum(A * query, dim=1) # BxD
        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        ) # BxNxD
        out = self.Proj(G * key) + query #BxNxD
        out = self.final(out) # BxNxD
        return out
   
if __name__ == "__main__":
    # 测试输入张量
    batch_size = 4
    num_tokens = 64
    in_dims = 512
    input_tensor = torch.randn(batch_size, num_tokens, in_dims)
    # Efficient Additive Attention 参数
    token_dim = 256
    num_heads = 2
    # 初始化 Efficient Additive Attention 模块
    attention_module = EfficientAdditiveAttnetion(
        in_dims=in_dims,
        token_dim=token_dim,
        num_heads=num_heads
    )
    # 转换到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention_module.to(device)
    input_tensor = input_tensor.to(device)
    # 前向传播
    output = attention_module(input_tensor)
    # 输出结果
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output.shape}")