import typing as t
import torch
import torch.nn as nn
from einops import rearrange

# ������Ŀ��SCSA: Exploring the Synergistic Effects Between Spatial and Channel Attention
# ������Ŀ:  SCSA: ̽���ռ�ע������ͨ��ע����֮���ЭͬЧӦ
# �������ӣ�https://arxiv.org/pdf/2407.05128

# ������Դ��https://github.com/HZAI-ZJNU/SCSA
# ����������ע�ͣ����ںţ�AI�����
"""
2024��ȫ����ȫ���弴��ģ��,ȫ�����!�������־�����֡�����ע�������ơ������ں�ģ�顢���²���ģ�飬
�������˹�����(AI)�����ѧϰ��������Ӿ�(CV)����������ͼ����ࡢĿ���⡢ʵ���ָ����ָ
��Ŀ�����(SOT)����Ŀ�����(MOT)��������ɼ���ͼ���ںϸ���(RGBT)��ͼ��ȥ�롢ȥ�ꡢȥ��ȥģ�������ֵ�����
ģ������������......
"""
# AI�����github��https://github.com/AIFengheshu/Plug-play-modules

class SCSA(nn.Module):
    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(SCSA, self).__init__()  # ���� nn.Module �Ĺ��캯��
        self.dim = dim  # ����ά��
        self.head_num = head_num  # ע����ͷ��
        self.head_dim = dim // head_num  # ÿ��ͷ��ά��
        self.scaler = self.head_dim ** -0.5  # ��������
        self.group_kernel_sizes = group_kernel_sizes  # �������˴�С
        self.window_size = window_size  # ���ڴ�С
        self.qkv_bias = qkv_bias  # �Ƿ�ʹ��ƫ��
        self.fuse_bn = fuse_bn  # �Ƿ��ں�����һ��
        self.down_sample_mode = down_sample_mode  # �²���ģʽ

        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'  # ȷ��ά�ȿɱ�4����
        self.group_chans = group_chans = self.dim // 4  # ����ͨ����

        # ����ֲ���ȫ����Ⱦ����
        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)

        # ע�����ſز�
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)  # ˮƽ����Ĺ�һ��
        self.norm_w = nn.GroupNorm(4, dim)  # ��ֱ����Ĺ�һ��

        self.conv_d = nn.Identity()  # ֱ������
        self.norm = nn.GroupNorm(1, dim)  # ͨ����һ��
        # �����ѯ������ֵ�ľ����
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)  # ע����������
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()  # ͨ��ע�����ſ�

        # ���ݴ��ڴ�С���²���ģʽѡ���²�������
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))  # ����Ӧƽ���ػ�
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans  # ������²���
                # ά�Ƚ���
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)  # ƽ���ػ�
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)  # ���ػ�

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        �������� x ��ά��Ϊ (B, C, H, W)
        """
        # ����ռ�ע�������ȼ�
        b, c, h_, w_ = x.size()  # ��ȡ�������״
        # (B, C, H)
        x_h = x.mean(dim=3)  # ���ſ��ά����ƽ��
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)  # ���ͨ��
        # (B, C, W)
        x_w = x.mean(dim=2)  # ���Ÿ߶�ά����ƽ��
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)  # ���ͨ��

        # ����ˮƽע����
        x_h_attn = self.sa_gate(self.norm_h(torch.cat((  
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)  # ������״

        # ���㴹ֱע����
        x_w_attn = self.sa_gate(self.norm_w(torch.cat((  
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)  # ������״

        # �������յ�ע������Ȩ
        x = x * x_h_attn * x_w_attn

        # ������ע������ͨ��ע����
        # ���ټ�����
        y = self.down_func(x)  # �²���
        y = self.conv_d(y)  # ά��ת��
        _, _, h_, w_ = y.size()  # ��ȡ��״

        # �ȹ�һ����Ȼ������ -> (B, H, W, C) -> (B, C, H * W)�������� q, k �� v
        y = self.norm(y)  # ��һ��
        q = self.q(y)  # �����ѯ
        k = self.k(y)  # �����
        v = self.v(y)  # ����ֵ
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # ����ע����
        attn = q @ k.transpose(-2, -1) * self.scaler  # ���ע��������
        attn = self.attn_drop(attn.softmax(dim=-1))  # Ӧ��ע��������
        # (B, head_num, head_dim, N)
        attn = attn @ v  # ��Ȩֵ
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)  # ��ƽ��
        attn = self.ca_gate(attn)  # Ӧ��ͨ��ע�����ſ�
        return attn * x  # ���ؼ�Ȩ�������

if __name__ == "__main__":
    
    #����: dim����ά��; head_numע����ͷ��; window_size = 7 ���ڴ�С
    scsa = SCSA(dim=32, head_num=8, window_size=7)
    # ��������������� (B, C, H, W)
    input_tensor = torch.rand(1, 32, 256, 256)
    # ��ӡ������������״
    print(f"������������״: {input_tensor.shape}")
    # ǰ�򴫲�
    output_tensor = scsa(input_tensor)
    # ��ӡ�����������״
    print(f"�����������״: {output_tensor.shape}")