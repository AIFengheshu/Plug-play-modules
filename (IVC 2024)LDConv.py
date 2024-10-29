import torch
import torch.nn as nn
from einops import rearrange
import math

# ������Ŀ��LDConv: Linear deformable convoluton for improving convolutioanl neural networks
# ������Ŀ: LDConv�����ڸĽ��������������Կɱ��ξ��
# �������ӣ�https://doi.org/10.1016/j.imavis.2024.105190

# ����������ע�ͣ����ںţ�AI�����
# AI�����github��https://github.com/AIFengheshu/Plug-play-modules

class LDConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param  # ��������
        self.stride = stride  # ����
        # �������㣬����BN���SiLU�������������ԭʼYOLOv5�ľ�����бȽ�
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
            nn.BatchNorm2d(outc),
            nn.SiLU()
        )
        # ����ƫ��������㣬��������2 * num_param��ƫ�Ʋ���
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)  # ��ʼ��ƫ���������Ȩ��Ϊ0
        # ע�ᷴ�򴫲�ʱ�Ĺ��Ӻ���������ѧϰ��
        self.p_conv.register_full_backward_hook(self._set_lr)

    # ��̬�����������ڷ��򴫲�ʱ�����ݶȣ���С�ݶȷ���
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N ��ʾ��������
        offset = self.p_conv(x)  # ����ƫ����
        dtype = offset.data.type()
        N = offset.size(1) // 2  # ƫ�����ĸ���
        # ����ƫ������������
        p = self._get_p(offset, dtype)

        # ����������Ϊ (b, h, w, 2N) �ĸ�ʽ
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()  # ���Ͻ�����ȡ��
        q_rb = q_lt + 1  # ���½�����

        # �������귶Χ��ʹ�䲻��������ͼ��ı߽�
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)  # ���½�����
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)  # ���Ͻ�����

        # ������p������ͼ��Χ��
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # ˫���Բ�ֵϵ��
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # ���ݵ�������������²�������
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # ˫���Բ�ֵ
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # ����ƫ����������״
        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)  # ���վ������

        return out

    # ���ɲ�ͬ��С�ĳ�ʼ������״
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))  # ��׼������
        row_number = self.num_param // base_int  # ����
        mod_number = self.num_param % base_int  # ʣ������
        # ������������
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int),
            indexing='ij'  # ��ʽָ������ģʽ
        )
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            # �������������ʣ�ಿ��Ҳ��������
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number+1),
                torch.arange(0, mod_number)
            )
            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)  # �ϲ�����
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)  # ������״
        return p_n

    # �����
    def _get_p_0(self, h, w, N, dtype):
        # ������������
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride),
            indexing='ij'
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    # ����ʵ��ƫ������
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)  # ��ȡ����������
        p_0 = self._get_p_0(h, w, N, dtype)  # ��ȡ��������
        p = p_0 + p_n + offset  # ����ƫ����
        return p

    # ��ȡ���������ϵ�����ֵ
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)  # չƽ����

        # ��������ֵ
        index = q[..., :N] * padded_w + q[..., N:]  # ����ƫ������
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        # ��ȡƫ�ƺ������
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    
    # Stacking resampled features in the row direction.
    # ���з����϶ѵ��ز���������
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()  # ��ȡ x_offset �����ĳߴ���������δ�С(b)��ͨ����(c)���߶�(h)�����(w)���ز�������(n)
        
        # using Conv3d
        # ʹ�� 3D ����������� x_offset ����ά������ (b, c, n, h, w)��Ȼ��ʹ�� Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c, c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        
        # using 1 �� 1 Conv
        # ʹ�� 1 �� 1 ������� x_offset ����ά������ (b, c, n, h, w)��Ȼ�� x_offset ����Ϊ (b, c��num_param, h, w)
        # Ȼ��ʹ�� Conv2d ����˴�СΪ1��1�Ĳ�����������任
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b, c��num_param, h, w)  
        # finally, Conv2d(c��num_param, c_out, kernel_size =1, stride=1, bias= False)

        # using the column conv as follow�� then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)
        # ʹ���о�������Ƚ����з����ϵľ��������ʹ�� kernel_size=(num_param, 1) �� stride=(num_param, 1) �� Conv2d
        
        # Rearrange x_offset dimensions for column stacking.
        # ʹ�� einops ��� rearrange �������� x_offset �� (b, c, h, w, n) ��������Ϊ (b, c, h��n, w)
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        
        return x_offset  # �����������к�� x_offset


def main():
    # ��ʼ��LDConvģ��,������Ҫʹ��3*3�ľ��,����ͨ����1�����ͨ����16
    num_param = 4  # �ɱ��ξ���Ĳ�������
    ld_conv = LDConv(inc=3, outc=16, num_param=num_param, stride=1)

    # ��ӡLDConv�Ľṹ
    # print("LDConv Structure:\n", ld_conv)

    # ����һ������������� (batch_size=1, channels=3, height=256, width=256)
    input_tensor = torch.randn(1, 3, 256, 256)
    print("Input Tensor Shape:", input_tensor.shape)

    # ǰ�򴫲�����
    output = ld_conv(input_tensor)
    print("Output Tensor Shape:", output.shape)

if __name__ == "__main__":
    main()