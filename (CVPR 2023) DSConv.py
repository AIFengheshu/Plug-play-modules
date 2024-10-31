import torch
from torch import nn
import einops
from typing import Union

# ������Ŀ��Dynamic Snake Convolution based on Topological Geometric Constraints for Tubular Structure Segmentation
# ������Ŀ:  ���˼���Լ����״�ṹ�ָ�Ķ�̬�߾��
# Ӣ���������ӣ�https://arxiv.org/pdf/2307.08388
# �����������ӣ�
# https://yaoleiqi.github.io/publication/2023_ICCV/DSCNet_Chinese.pdf
# �ٷ�github��https://github.com/YaoleiQi/DSCNet
# �������������ϴ�ѧ�˹�������һ�����������ѧ��Ӧ�ý������ص�ʵ���ң�����ʡҽѧ��Ϣ�����������ʵ���ң��з�����ҽѧ��Ϣ�о�����
# �ؼ��ʣ�����֪ʶ�ںϣ���̬���ξ�������ӽ������ںϣ�����ͬ������״�ṹ�ָ�

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
        ��̬���ξ��ģ�� (Dynamic Snake Convolution) ��ʵ�֡�
        Args:
            in_channels: ����ͨ������Ĭ��Ϊ1��
            out_channels: ���ͨ������Ĭ��Ϊ1��
            kernel_size: ����˴�С��Ĭ��Ϊ9��
            extend_scope: �������չ��Χ�����ڿ��ƾ��������Ӱ�췶Χ��Ĭ��Ϊ1��
            morph: �������̬���ͣ���x�ᣨ0����y�ᣨ1�����μ������˽����ϸ�ڡ�
            if_offset: �Ƿ�����α��������Ϊ False����Ϊ��׼�����Ĭ��Ϊ True��
        """
        super().__init__()
        if morph not in (0, 1):
            raise ValueError("morph Ӧ���� 0 �� 1��")
        
        # �����������
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device(device)
        self.to(device)

        # ƫ������һ���㣬ʹ�÷����һ������ƫ������
        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        # ���������һ���㣬ʹ�÷����һ�������������
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        # �����
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        # ƫ�ƾ���㣬��������ƫ���������ڶ�̬���
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)

        # ��̬���ξ����x��ľ���ˣ�����˴�СΪ (kernel_size, 1)
        self.dsc_conv_x = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        # ��̬���ξ����y��ľ���ˣ�����˴�СΪ (1, kernel_size)
        self.dsc_conv_y = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

    def forward(self, input: torch.Tensor):
        # ����ƫ����ͼ����ֵ��ΧΪ [-1, 1]
        offset = self.offset_conv(input)
        # ��ƫ�������з����һ������
        offset = self.gn_offset(offset)
        offset = self.tanh(offset)

        # ��ȡ���ξ����ƫ������ӳ��
        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(
            offset=offset,
            morph=self.morph,
            extend_scope=self.extend_scope,
            device=self.device,
        )
        # ����ƫ�������ȡ��ֵ�������
        deformed_feature = get_interpolated_feature(
            input,
            y_coordinate_map,
            x_coordinate_map,
        )

        # ���ݾ����̬ѡ����ʵľ������
        if self.morph == 0:
            # �� x ��Ķ�̬���ξ��
            output = self.dsc_conv_x(deformed_feature)
        elif self.morph == 1:
            # �� y ��Ķ�̬���ξ��
            output = self.dsc_conv_y(deformed_feature)

        # ʹ�÷����һ���� ReLU ��������������
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
    ���㶯̬���ξ����2D����ӳ�䡣
    Args:
        offset: ����Ԥ���ƫ��������״Ϊ [B, 2*K, W, H]������ K ��ʾ����˴�С��
        morph: �������̬���ͣ���x�ᣨ0����y�ᣨ1����
        extend_scope: ��չ��Χ�����ƾ����ƫ�Ʒ�Χ��Ĭ��Ϊ 1��
        device: ���������豸��Ĭ��Ϊ 'cuda'��
    Return:
        y_coordinate_map: y�������ӳ�䣬��״Ϊ [B, K_H * H, K_W * W]
        x_coordinate_map: x�������ӳ�䣬��״Ϊ [B, K_H * H, K_W * W]
    """
    # ��� morph �����Ƿ�Ϊ��Чֵ
    if morph not in (0, 1):
        raise ValueError("morph Ӧ���� 0 �� 1��")

    # ��ȡ����С����Ⱥ͸߶�
    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2  # �������˴�С
    center = kernel_size // 2  # ����λ��
    device = torch.device(device)  # ȷ���豸

    # ��ƫ�������Ϊ x �� y ��ƫ��
    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)

    # ���� y ����������꣬��չ��ÿ�������λ�ú͸߶�
    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

    # ���� x ����������꣬��չ��ÿ�������λ�úͿ��
    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

    # ���� morph ֵ������ͬ��̬�ľ��
    if morph == 0:
        """
        ��ʼ������ˣ���չ������ˣ�
            y��ֻ��Ҫ0
            x����ΧΪ -num_points//2 �� num_points//2���ɾ���˴�С������
        """
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        # �� y �� x ����չ�ֲ�����Ӧ�Ŀ�͸���
        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        # �����µ� y �� x ����
        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        # �ظ� y �� x ���꣬����Ӧ����ά��
        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        # ���� y ƫ�Ʋ���ʼ��ƫ����
        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()

        # ����λ�ñ��ֲ��䣬����λ�ÿ�ʼҡ��
        # ƫ������һ����������
        y_offset_new_[center] = 0

        for index in range(1, center + 1):
            y_offset_new_[center + index] = (
                y_offset_new_[center + index - 1] + y_offset_[center + index]
            )
            y_offset_new_[center - index] = (
                y_offset_new_[center - index + 1] + y_offset_[center - index]
            )

        # ���� y ƫ�ƺ󣬽�ƫ����Ӧ�õ�����
        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")
        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        # ���� y �� x ����ͼ
        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")

    elif morph == 1:
        """
        ��ʼ������ˣ���չ������ˣ�
            y����ΧΪ -num_points//2 �� num_points//2���ɾ���˴�С������
            x��ֻ��Ҫ0
        """
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)

        # �� y �� x ����չ�ֲ�����Ӧ�Ŀ�͸���
        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        # �����µ� y �� x ����
        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        # �ظ� y �� x ���꣬����Ӧ����ά��
        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        # ���� x ƫ�Ʋ���ʼ��ƫ����
        x_offset_ = einops.rearrange(x_offset_, "b k w h -> k b w h")
        x_offset_new_ = x_offset_.detach().clone()

        # ����λ�ñ��ֲ��䣬����λ�ÿ�ʼҡ��
        # ƫ������һ����������
        x_offset_new_[center] = 0

        for index in range(1, center + 1):
            x_offset_new_[center + index] = (
                x_offset_new_[center + index - 1] + x_offset_[center + index]
            )
            x_offset_new_[center - index] = (
                x_offset_new_[center - index + 1] + x_offset_[center - index]
            )

        # ���� x ƫ�ƺ󣬽�ƫ����Ӧ�õ�����
        x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")
        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))

        # ���� y �� x ����ͼ
        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b w (h k)")

    return y_coordinate_map, x_coordinate_map

def get_interpolated_feature(
    input_feature: torch.Tensor,
    y_coordinate_map: torch.Tensor,
    x_coordinate_map: torch.Tensor,
    interpolate_mode: str = "bilinear",
):
    """��������ͼ��ֵDSCNet������

    Args:
        input_feature: ����ֵ������ͼ����״Ϊ [B, C, H, W]
        y_coordinate_map: ��y�������ͼ����״Ϊ [B, K_H * H, K_W * W]
        x_coordinate_map: ��x�������ͼ����״Ϊ [B, K_H * H, K_W * W]
        interpolate_mode: nn.functional.grid_sample�Ĳ�ֵģʽ������Ϊ 'bilinear' �� 'bicubic' ��Ĭ���� 'bilinear'��

    Return:
        interpolated_feature: ��ֵ�������ͼ����״Ϊ [B, C, K_H * H, K_W * W]
    """

    # ����ֵģʽ�Ƿ���ȷ
    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode ӦΪ 'bilinear' �� 'bicubic'��")

    # ��ȡ y �� x �����ֵ
    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1

    # ���� y ����ͼ��ָ����Χ
    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    # ���� x ����ͼ��ָ����Χ
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

    # ����һά��ʹ������ͼ����״���� grid_sample ������
    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

    # �ϲ� x �� y ����ͼ������ grid����״Ϊ [B, H, W, 2]������ [:, :, :, 2] ��ʾ [x ,y]
    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

    # ʹ�� grid_sample ���в�ֵ
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
    """������ͼ��ֵ�� origin=[min, max] ӳ�䵽 target=[a,b]������ DSCNet

    Args:
        coordinate_map: ��Ҫ���ŵ�����ͼ
        origin: ����ͼ��ԭʼֵ��Χ������ [coordinate_map.min(), coordinate_map.max()]
        target: ����ͼ��Ŀ��ֵ��Χ��Ĭ���� [-1, 1]

    Return:
        coordinate_map_scaled: ���ź������ͼ
    """
    min, max = origin
    a, b = target

    # ������ͼ������ [min, max] ��Χ��
    coordinate_map_scaled = torch.clamp(coordinate_map, min, max)

    # �����������Ӳ�Ӧ�õ�����ͼ��
    scale_factor = (b - a) / (max - min)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min)

    return coordinate_map_scaled


if __name__ == '__main__':
    # ȷ���豸�������GPU��ʹ��
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ����һ��������������״Ϊ [1, 32, 256, 256] �����õ��豸��
    input = torch.randn(1,32,256,256).to(device)
    # ʵ���� DSConv_pro �����
    dsconv = DSConv_pro(32, 64).to(device)
    # ͨ�������������
    output = dsconv(input)
    # ��ӡ������������״
    print(input.shape)
    print(output.shape)
