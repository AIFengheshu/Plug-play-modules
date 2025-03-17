import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def patch_divide(x, step, ps):
    """Crop image into patches.
    Args:
        x (Tensor): Input feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        crop_x (Tensor): Cropped patches.
        nh (int): Number of patches along the horizontal direction.
        nw (int): Number of patches along the vertical direction.
    """
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        nh += 1
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    crop_x = torch.stack(crop_x, dim=0)  # (n, b, c, ps, ps)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()  # (b, n, c, ps, ps)
    return crop_x, nh, nw

def patch_reverse(crop_x, x, step, ps):
    """Reverse patches into image.
    Args:
        crop_x (Tensor): Cropped patches.
        x (Tensor): Feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        ouput (Tensor): Reversed image.
    """
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2
    return output

class PreNorm(nn.Module):
    """Normalization layer.
    Args:
        dim (int): Base channels.
        fn (Module): Module after normalization.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class Attention(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
    """

    def __init__(self, dim, heads, qk_dim):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q,k,v) # scaled_dot_product_attention 需要PyTorch2.0之后版本
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)
    
class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x
    
class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x
     
class LRSA(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        num (int): Number of blocks.
        qk_dim (int): Channels of query and key in Attention.
        mlp_dim (int): Channels of hidden mlp in Mlp.
        heads (int): Head numbers of Attention.
    """

    def __init__(self, dim, qk_dim, mlp_dim, heads=1):
        super().__init__()
        self.layer = nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, qk_dim)),
                PreNorm(dim, ConvFFN(dim, mlp_dim))])

    def forward(self, x, ps):
        step = ps - 2
        crop_x, nh, nw = patch_divide(x, step, ps)  # (b, n, c, ps, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')
        attn, ff = self.layer
        crop_x = attn(crop_x) + crop_x
        crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)
        x = patch_reverse(crop_x, x, step, ps)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = ff(x, x_size=(h, w)) + x
        x = rearrange(x, 'b (h w) c->b c h w', h=h)
        return x

if __name__ == "__main__":
    # 输入参数配置
    batch_size = 1  # Batch size
    channels = 32   # 输入通道数
    height = 256    # 高度
    width = 256     # 宽度
    ps = 16         # Patch size
    qk_dim = 36     # Query-Key维度
    mlp_dim = 96    # MLP维度
    heads = 4       # Attention头数
    
    # 创建一个输入张量，形状为 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)

    # 初始化 LRSA 模块
    model = LRSA(dim=channels, qk_dim=qk_dim, mlp_dim=mlp_dim, heads=heads)
    print(model)
    print("微信公众号: AI缝合术!")

    # 前向传播，传入输入张量和patch大小
    output = model(x, ps)

    # 打印输入和输出张量的形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

