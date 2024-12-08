import torch
from torch import nn
import einops

# ������Ŀ��SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications
# ������Ŀ: SwiftFormer����Ի���Transformer��ʵʱ�ƶ��Ӿ�Ӧ�õĸ�Ч����ע����
# �������ӣ�https://openaccess.thecvf.com/content/ICCV2023/papers/Shaker_SwiftFormer_Efficient_Additive_Attention_for_Transformer-based_Real-time_Mobile_Vision_Applications_ICCV_2023_paper.pdf
# �ٷ�github��https://github.com/Amshaker/SwiftFormer
# �����������º�Ĭ�¡�������Ү��AI��ѧ�����ݴ�ѧĬ���·�У��������ѧ���ȸ��о����ġ�Link?ping��ѧ
# �ؼ��ʣ�SwiftFormer, Transformer, ��ע����, �ƶ��Ӿ�Ӧ��, ʵʱ����, ������
# ΢�Ź��ںţ�AI�����

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
    # ������������
    batch_size = 4
    num_tokens = 64
    in_dims = 512
    input_tensor = torch.randn(batch_size, num_tokens, in_dims)
    # Efficient Additive Attention ����
    token_dim = 256
    num_heads = 2
    # ��ʼ�� Efficient Additive Attention ģ��
    attention_module = EfficientAdditiveAttnetion(
        in_dims=in_dims,
        token_dim=token_dim,
        num_heads=num_heads
    )
    # ת���� GPU��������ã�
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention_module.to(device)
    input_tensor = input_tensor.to(device)
    # ǰ�򴫲�
    output = attention_module(input_tensor)
    # ������
    print(f"����������״: {input_tensor.shape}")
    print(f"���������״: {output.shape}")