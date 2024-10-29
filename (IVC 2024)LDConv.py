import torch
import torch.nn as nn
from einops import rearrange
import math

# 论文题目：LDConv: Linear deformable convoluton for improving convolutioanl neural networks
# 中文题目: LDConv：用于改进卷积神经网络的线性可变形卷积
# 论文链接：https://doi.org/10.1016/j.imavis.2024.105190

# 代码整理与注释：公众号：AI缝合术
# AI缝合术github：https://github.com/AIFengheshu/Plug-play-modules

class LDConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param  # 参数个数
        self.stride = stride  # 步长
        # 定义卷积层，包含BN层和SiLU激活函数，用于与原始YOLOv5的卷积进行比较
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
            nn.BatchNorm2d(outc),
            nn.SiLU()
        )
        # 定义偏移量卷积层，用于生成2 * num_param个偏移参数
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)  # 初始化偏移量卷积的权重为0
        # 注册反向传播时的钩子函数，调整学习率
        self.p_conv.register_full_backward_hook(self._set_lr)

    # 静态方法，用于在反向传播时调整梯度，减小梯度幅度
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N 表示参数数量
        offset = self.p_conv(x)  # 生成偏移量
        dtype = offset.data.type()
        N = offset.size(1) // 2  # 偏移量的个数
        # 根据偏移量生成坐标
        p = self._get_p(offset, dtype)

        # 将坐标排列为 (b, h, w, 2N) 的格式
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()  # 左上角坐标取整
        q_rb = q_lt + 1  # 右下角坐标

        # 限制坐标范围，使其不超出输入图像的边界
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)  # 左下角坐标
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)  # 右上角坐标

        # 将坐标p限制在图像范围内
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # 双线性插值系数
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # 根据调整后的坐标重新采样特征
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # 双线性插值
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # 调整偏移特征的形状
        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)  # 最终卷积操作

        return out

    # 生成不同大小的初始采样形状
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))  # 基准整形数
        row_number = self.num_param // base_int  # 行数
        mod_number = self.num_param % base_int  # 剩余数量
        # 创建网格坐标
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int),
            indexing='ij'  # 显式指定索引模式
        )
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            # 如果有余数，将剩余部分也加入网格
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number+1),
                torch.arange(0, mod_number)
            )
            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)  # 合并坐标
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)  # 重塑形状
        return p_n

    # 无填充
    def _get_p_0(self, h, w, N, dtype):
        # 生成坐标网格
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride),
            indexing='ij'
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    # 计算实际偏移坐标
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)  # 获取基础采样点
        p_0 = self._get_p_0(h, w, N, dtype)  # 获取网格坐标
        p = p_0 + p_n + offset  # 加上偏移量
        return p

    # 获取在新坐标上的特征值
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)  # 展平特征

        # 计算索引值
        index = q[..., :N] * padded_w + q[..., N:]  # 计算偏移索引
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        # 获取偏移后的特征
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    
    # Stacking resampled features in the row direction.
    # 在行方向上堆叠重采样的特征
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()  # 获取 x_offset 张量的尺寸参数：批次大小(b)、通道数(c)、高度(h)、宽度(w)和重采样数量(n)
        
        # using Conv3d
        # 使用 3D 卷积操作：将 x_offset 进行维度排列 (b, c, n, h, w)，然后使用 Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c, c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        
        # using 1 × 1 Conv
        # 使用 1 × 1 卷积：将 x_offset 进行维度排列 (b, c, n, h, w)，然后将 x_offset 重塑为 (b, c×num_param, h, w)
        # 然后使用 Conv2d 卷积核大小为1×1的操作完成特征变换
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b, c×num_param, h, w)  
        # finally, Conv2d(c×num_param, c_out, kernel_size =1, stride=1, bias= False)

        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)
        # 使用列卷积：首先进行列方向上的卷积，接着使用 kernel_size=(num_param, 1) 和 stride=(num_param, 1) 的 Conv2d
        
        # Rearrange x_offset dimensions for column stacking.
        # 使用 einops 库的 rearrange 方法，将 x_offset 从 (b, c, h, w, n) 重新排列为 (b, c, h×n, w)
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        
        return x_offset  # 返回重新排列后的 x_offset


def main():
    # 初始化LDConv模块,假设你要使用3*3的卷积,输入通道数1，输出通道数16
    num_param = 4  # 可变形卷积的参数数量
    ld_conv = LDConv(inc=3, outc=16, num_param=num_param, stride=1)

    # 打印LDConv的结构
    # print("LDConv Structure:\n", ld_conv)

    # 创建一个随机输入张量 (batch_size=1, channels=3, height=256, width=256)
    input_tensor = torch.randn(1, 3, 256, 256)
    print("Input Tensor Shape:", input_tensor.shape)

    # 前向传播测试
    output = ld_conv(input_tensor)
    print("Output Tensor Shape:", output.shape)

if __name__ == "__main__":
    main()