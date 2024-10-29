from typing import Optional
import torch.nn as nn
import torch

# 论文题目：Poly Kernel Inception Network for Remote Sensing Detection
# 中文题目:  面向遥感检测的多核Inception网络
# 论文链接：
# https://openaccess.thecvf.com/content/CVPR2024/papers/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.pdf
# 代码整理与注释：公众号：AI缝合术
# AI缝合术github：https://github.com/AIFengheshu/Plug-play-modules

# 定义卷积模块类，来自mmcv.cnn
class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,      # 输入通道数
            out_channels: int,     # 输出通道数
            kernel_size: int,      # 卷积核大小
            stride: int = 1,       # 步长
            padding: int = 0,      # 填充
            groups: int = 1,       # 组卷积数
            norm_cfg: Optional[dict] = None,  # 归一化配置
            act_cfg: Optional[dict] = None):  # 激活函数配置
        super().__init__()
        layers = []
        # 卷积层
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=(norm_cfg is None)))
        # 归一化层
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)
        # 激活层
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        # 将所有层组合为一个序列层
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    # 定义获取归一化层的辅助函数
    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
        # 若需要其他归一化类型可以在此处添加
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    # 定义获取激活层的辅助函数
    def _get_act_layer(self, act_cfg):
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        # 若需要其他激活类型可以在此处添加
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")

# 定义上下文锚点注意力 (Context Anchor Attention) 模块
class CAA(nn.Module):
    """上下文锚点注意力模块"""
    def __init__(
            self,
            channels: int,                     # 输入通道数
            h_kernel_size: int = 11,           # 水平卷积核大小
            v_kernel_size: int = 11,           # 垂直卷积核大小
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),  # 归一化配置
            act_cfg: Optional[dict] = dict(type='SiLU')):                         # 激活函数配置
        super().__init__()
        # 平均池化层
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        # 1x1卷积模块，用于调整通道数
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        # 水平卷积模块，使用1xh_kernel_size的卷积核，仅在水平方向上进行卷积
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        # 垂直卷积模块，使用v_kernel_sizex1的卷积核，仅在垂直方向上进行卷积
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        # 1x1卷积模块，用于进一步调整通道数
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        # 使用Sigmoid激活函数
        self.act = nn.Sigmoid()

    # 前向传播函数
    def forward(self, x):
        # 通过平均池化、卷积和激活函数计算注意力系数
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        # x与生成的注意力系数相乘，生成增强后特征图
        x = x*attn_factor
        return x

# 测试
if __name__ == "__main__":

    caa = CAA(32)
    # 随机生成输入张量 (B, C, H, W)
    input_tensor = torch.rand(1, 32, 256, 256)
    # 打印输入张量的形状
    print(f"输入张量的形状: {input_tensor.shape}")
    # 前向传播
    output_tensor = caa(input_tensor)
    # 打印输出张量的形状
    print(f"输出张量的形状: {output_tensor.shape}")
