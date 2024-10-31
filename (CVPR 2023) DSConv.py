import torch
from torch import nn
import einops
from typing import Union

# 论文题目：Dynamic Snake Convolution based on Topological Geometric Constraints for Tubular Structure Segmentation
# 中文题目:  拓扑几何约束管状结构分割的动态蛇卷积
# 英文论文链接：https://arxiv.org/pdf/2307.08388
# 中文论文链接：
# https://yaoleiqi.github.io/publication/2023_ICCV/DSCNet_Chinese.pdf
# 官方github：https://github.com/YaoleiQi/DSCNet
# 所属机构：东南大学人工智能新一代技术及其跨学科应用教育部重点实验室，江苏省医学信息处理国际联合实验室，中法生物医学信息研究中心
# 关键词：先验知识融合，动态蛇形卷积，多视角特征融合，持续同调，管状结构分割

"""Dynamic Snake Convolution Module"""
class DSConv_pro(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        morph: int = 0,
        if_offset: bool = True,
        # device: str | torch.device = "cuda",
        device: Union[str, torch.device] = "cuda",
    ):
        """
        动态蛇形卷积模块 (Dynamic Snake Convolution) 的实现。
        Args:
            in_channels: 输入通道数，默认为1。
            out_channels: 输出通道数，默认为1。
            kernel_size: 卷积核大小，默认为9。
            extend_scope: 卷积核扩展范围，用于控制卷积操作的影响范围。默认为1。
            morph: 卷积核形态类型，沿x轴（0）或y轴（1）。参见论文了解更多细节。
            if_offset: 是否进行形变操作，若为 False，则为标准卷积。默认为 True。
        """
        super().__init__()
        if morph not in (0, 1):
            raise ValueError("morph 应该是 0 或 1。")
        
        # 保存输入参数
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device(device)
        self.to(device)

        # 偏移量归一化层，使用分组归一化处理偏移特征
        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        # 输出特征归一化层，使用分组归一化处理输出特征
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        # 偏移卷积层，生成特征偏移量，用于动态卷积
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)

        # 动态蛇形卷积沿x轴的卷积核，卷积核大小为 (kernel_size, 1)
        self.dsc_conv_x = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        # 动态蛇形卷积沿y轴的卷积核，卷积核大小为 (1, kernel_size)
        self.dsc_conv_y = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

    def forward(self, input: torch.Tensor):
        # 生成偏移量图，数值范围为 [-1, 1]
        offset = self.offset_conv(input)
        # 对偏移量进行分组归一化处理
        offset = self.gn_offset(offset)
        offset = self.tanh(offset)

        # 获取变形卷积的偏移坐标映射
        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(
            offset=offset,
            morph=self.morph,
            extend_scope=self.extend_scope,
            device=self.device,
        )
        # 根据偏移坐标获取插值后的特征
        deformed_feature = get_interpolated_feature(
            input,
            y_coordinate_map,
            x_coordinate_map,
        )

        # 根据卷积形态选择合适的卷积操作
        if self.morph == 0:
            # 沿 x 轴的动态蛇形卷积
            output = self.dsc_conv_x(deformed_feature)
        elif self.morph == 1:
            # 沿 y 轴的动态蛇形卷积
            output = self.dsc_conv_y(deformed_feature)

        # 使用分组归一化和 ReLU 激活函数处理卷积结果
        output = self.gn(output)
        output = self.relu(output)

        return output

def get_coordinate_map_2D(
    offset: torch.Tensor,
    morph: int,
    extend_scope: float = 1.0,
    device: Union[str, torch.device] = "cuda",
):
    """
    计算动态蛇形卷积的2D坐标映射。
    Args:
        offset: 网络预测的偏移量，形状为 [B, 2*K, W, H]，其中 K 表示卷积核大小。
        morph: 卷积核形态类型，沿x轴（0）或y轴（1）。
        extend_scope: 扩展范围，控制卷积的偏移范围，默认为 1。
        device: 数据所在设备，默认为 'cuda'。
    Return:
        y_coordinate_map: y轴的坐标映射，形状为 [B, K_H * H, K_W * W]
        x_coordinate_map: x轴的坐标映射，形状为 [B, K_H * H, K_W * W]
    """
    # 检查 morph 参数是否为有效值
    if morph not in (0, 1):
        raise ValueError("morph 应该是 0 或 1。")

    # 获取批大小、宽度和高度
    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2  # 计算卷积核大小
    center = kernel_size // 2  # 中心位置
    device = torch.device(device)  # 确定设备

    # 将偏移量拆分为 x 和 y 的偏移
    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)

    # 生成 y 轴的中心坐标，扩展到每个卷积核位置和高度
    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

    # 生成 x 轴的中心坐标，扩展到每个卷积核位置和宽度
    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

    # 根据 morph 值来处理不同形态的卷积
    if morph == 0:
        """
        初始化卷积核，并展开卷积核：
            y：只需要0
            x：范围为 -num_points//2 到 num_points//2（由卷积核大小决定）
        """
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        # 将 y 和 x 的扩展分布到对应的宽和高上
        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        # 计算新的 y 和 x 坐标
        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        # 重复 y 和 x 坐标，以适应批次维度
        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        # 调整 y 偏移并初始化偏移量
        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()

        # 中心位置保持不变，其他位置开始摇摆
        # 偏移量是一个迭代过程
        y_offset_new_[center] = 0

        for index in range(1, center + 1):
            y_offset_new_[center + index] = (
                y_offset_new_[center + index - 1] + y_offset_[center + index]
            )
            y_offset_new_[center - index] = (
                y_offset_new_[center - index + 1] + y_offset_[center - index]
            )

        # 调整 y 偏移后，将偏移量应用到坐标
        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")
        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        # 生成 y 和 x 坐标图
        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")

    elif morph == 1:
        """
        初始化卷积核，并展开卷积核：
            y：范围为 -num_points//2 到 num_points//2（由卷积核大小决定）
            x：只需要0
        """
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)

        # 将 y 和 x 的扩展分布到对应的宽和高上
        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        # 计算新的 y 和 x 坐标
        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        # 重复 y 和 x 坐标，以适应批次维度
        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        # 调整 x 偏移并初始化偏移量
        x_offset_ = einops.rearrange(x_offset_, "b k w h -> k b w h")
        x_offset_new_ = x_offset_.detach().clone()

        # 中心位置保持不变，其他位置开始摇摆
        # 偏移量是一个迭代过程
        x_offset_new_[center] = 0

        for index in range(1, center + 1):
            x_offset_new_[center + index] = (
                x_offset_new_[center + index - 1] + x_offset_[center + index]
            )
            x_offset_new_[center - index] = (
                x_offset_new_[center - index + 1] + x_offset_[center - index]
            )

        # 调整 x 偏移后，将偏移量应用到坐标
        x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")
        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))

        # 生成 y 和 x 坐标图
        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b w (h k)")

    return y_coordinate_map, x_coordinate_map

def get_interpolated_feature(
    input_feature: torch.Tensor,
    y_coordinate_map: torch.Tensor,
    x_coordinate_map: torch.Tensor,
    interpolate_mode: str = "bilinear",
):
    """根据坐标图插值DSCNet的特征

    Args:
        input_feature: 待插值的特征图，形状为 [B, C, H, W]
        y_coordinate_map: 沿y轴的坐标图，形状为 [B, K_H * H, K_W * W]
        x_coordinate_map: 沿x轴的坐标图，形状为 [B, K_H * H, K_W * W]
        interpolate_mode: nn.functional.grid_sample的插值模式，可以为 'bilinear' 或 'bicubic' ，默认是 'bilinear'。

    Return:
        interpolated_feature: 插值后的特征图，形状为 [B, C, K_H * H, K_W * W]
    """

    # 检查插值模式是否正确
    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode 应为 'bilinear' 或 'bicubic'。")

    # 获取 y 和 x 的最大值
    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1

    # 缩放 y 坐标图到指定范围
    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    # 缩放 x 坐标图到指定范围
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

    # 增加一维，使得坐标图的形状适配 grid_sample 的输入
    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

    # 合并 x 和 y 坐标图，生成 grid，形状为 [B, H, W, 2]，其中 [:, :, :, 2] 表示 [x ,y]
    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

    # 使用 grid_sample 进行插值
    interpolated_feature = nn.functional.grid_sample(
        input=input_feature,
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )

    return interpolated_feature


def _coordinate_map_scaling(
    coordinate_map: torch.Tensor,
    origin: list,
    target: list = [-1, 1],
):
    """将坐标图的值从 origin=[min, max] 映射到 target=[a,b]，用于 DSCNet

    Args:
        coordinate_map: 需要缩放的坐标图
        origin: 坐标图的原始值范围，例如 [coordinate_map.min(), coordinate_map.max()]
        target: 坐标图的目标值范围，默认是 [-1, 1]

    Return:
        coordinate_map_scaled: 缩放后的坐标图
    """
    min, max = origin
    a, b = target

    # 将坐标图限制在 [min, max] 范围内
    coordinate_map_scaled = torch.clamp(coordinate_map, min, max)

    # 计算缩放因子并应用到坐标图上
    scale_factor = (b - a) / (max - min)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min)

    return coordinate_map_scaled


if __name__ == '__main__':
    # 确定设备，如果有GPU则使用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建一个输入张量，形状为 [1, 32, 256, 256] 并放置到设备上
    input = torch.randn(1,32,256,256).to(device)
    # 实例化 DSConv_pro 卷积层
    dsconv = DSConv_pro(32, 64).to(device)
    # 通过卷积层计算输出
    output = dsconv(input)
    # 打印输入和输出的形状
    print(input.shape)
    print(output.shape)
