import torch
from torch import nn
from typing import Tuple

# ������Ŀ��RMT: Retentive Networks Meet Vision Transformers
# ������Ŀ��RMT�������������Ӿ���ѹ������
# �������ӣ�https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_RMT_Retentive_Networks_Meet_Vision_Transformers_CVPR_2024_paper.pdf
# �ٷ�github��https://github.com/qhfan/RMT
# �����������й���ѧԺ�Զ����о��� & CRIPAC���й���ѧԺ��ѧ�˹�����ѧԺ�������Ƽ���ѧ
# ��������΢�Ź��ںš�AI�������

def rotate_every_two(x):
    # ���������������һ��ά�ȣ�ÿ������Ԫ�ط��飬������һ���������з�ת
    x1 = x[:, :, :, :, ::2]  # ѡȡÿ������Ԫ�صĵ�һ��
    x2 = x[:, :, :, :, 1::2]  # ѡȡÿ������Ԫ�صĵڶ���
    x = torch.stack([-x2, x1], dim=-1)  # ����˳�򲢰�ά�ȶѵ�
    return x.flatten(-2)  # ���������ά�Ⱥϲ���һ��ά��

def theta_shift(x, sin, cos):
    # Ӧ����ת�任���������Һ����ҵ�����������
    return (x * cos) + (rotate_every_two(x) * sin)

class DWConv2d(nn.Module):
    # ��ȿɷ�������
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        # ������Ⱦ����ÿ��ͨ����������
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c) ������������״Ϊ (����, �߶�, ���, ͨ����)
        '''
        x = x.permute(0, 3, 1, 2)  # ת��Ϊ (b c h w) ������������
        x = self.conv(x)  # Ӧ����Ⱦ��
        x = x.permute(0, 2, 3, 1)  # ת�û�ԭʼ��״ (b h w c)
        return x

class RetNetRelPos2d(nn.Module):
    # RetNet �Ķ�ά���λ�ñ���ģ��
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        embed_dim: Ƕ��ά��
        num_heads: ע����ͷ��
        initial_value: ��ʼ˥��ֵ
        heads_range: ˥����Χ
        '''
        super().__init__()
        # ��ʼ���ǶȲ������������λ�õ����Һ����ұ���
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()  # �ظ��Ի�����Һ����Ҳ���
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads

        # ��ʼ��˥������
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_2d_decay(self, H: int, W: int):
        '''
        ���ɶ�ά��˥������
        H, W: �����ĸ߶ȺͿ��
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])  # ��������
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  # ����Ϊ��ά����
        mask = grid[:, None, :] - grid[None, :, :]  # �������λ�ò�
        mask = (mask.abs()).sum(dim=-1)  # ���������پ���
        mask = mask * self.decay[:, None, None]  # ��Ȩ˥��
        return mask

    def generate_1d_decay(self, l: int):
        '''
        ����һά��˥������
        l: ���г���
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]  # �������λ�ò�
        mask = mask.abs()  # ȡ����ֵ
        mask = mask * self.decay[:, None, None]  # ��Ȩ˥��
        return mask

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w) ���������ĸ߶ȺͿ��
        activate_recurrent: �Ƿ�����ѭ������
        chunkwise_recurrent: �Ƿ����÷ֿ�ѭ������
        '''
        if activate_recurrent:
            # ѭ��ģʽ�������ܳ����������Һ�����
            sin = torch.sin(self.angle * (slen[0] * slen[1] - 1))
            cos = torch.cos(self.angle * (slen[0] * slen[1] - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())

        elif chunkwise_recurrent:
            # �ֿ�ģʽ����ÿ�鵥���������Һ����ң��Լ�һά����
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(slen[0], slen[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(slen[0], slen[1], -1)
            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])
            retention_rel_pos = ((sin, cos), (mask_h, mask_w))

        else:
            # ��ͨģʽ���������ҡ����ҺͶ�ά����
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(slen[0], slen[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(slen[0], slen[1], -1)
            mask = self.generate_2d_decay(slen[0], slen[1])
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos

class VisionRetentionChunk(nn.Module):
    # Vision Retention ģ�飬����ʵ���Ӿ�������ע��������
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5  # �������ӣ����ڹ淶��
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)  # ��ѯͶӰ��
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)  # ��ֵͶӰ��
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)  # ֵͶӰ��
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)  # �ֲ���ǿģ��
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)  # ���ͶӰ��
        self.reset_parameters()

    def reset_parameters(self):
        # ��ʼ��Ȩ�ز���
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c) ��������
        rel_pos: ���λ�ñ���
        '''
        bsz, h, w, _ = x.size()
        (sin, cos), (mask_h, mask_w) = rel_pos

        # �����ѯ������ֵ����
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)  # ����ֲ���ǿ����

        # �Բ�ѯ�ͼ��������ź���ת�任
        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        # �����ȷ����ϵ�ע����
        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)
        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        qk_mat_w = qk_mat_w + mask_w
        qk_mat_w = torch.softmax(qk_mat_w, -1)
        v = torch.matmul(qk_mat_w, v)

        # ����߶ȷ����ϵ�ע����
        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)
        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        qk_mat_h = qk_mat_h + mask_h
        qk_mat_h = torch.softmax(qk_mat_h, -1)
        output = torch.matmul(qk_mat_h, v)

        # �ָ������״����Ӿֲ���ǿ����
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

if __name__ == '__main__':

    # ��������������ݣ�ע����������˳��ΪB H W C
    input_data = torch.randn(1, 256, 256, 32)  # ��������
    b, h, w, c = input_data.size()
    pos = RetNetRelPos2d(embed_dim=32, num_heads=4, initial_value=1, heads_range=3)  # ��ʼ��λ�ñ���
    rel_pos = pos((h, w), chunkwise_recurrent=True)  # �������λ�ñ���

    retention = VisionRetentionChunk(embed_dim=32, num_heads=4)  # ��ʼ��ģ��
    output = retention(input_data, rel_pos)

    # ��ӡ����������״
    print("Input size:", input_data.size())
    print("Output size:", output.size())
