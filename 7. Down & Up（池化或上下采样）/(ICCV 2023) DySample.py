import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文题目：Learning to Upsample by Learning to Sample
# 中文题目:  通过学习采样来学习上采样
# 论文链接：https://arxiv.org/pdf/2308.15085
# 官方github：https://github.com/tiny-smart/dysample
# 所属机构：华中科技大学人工智能与自动化学院
# 关键词：DySample, 动态上采样, 点采样, 密集预测任务, 资源效率
# 微信公众号：AI缝合术   https://github.com/AIFengheshu/Plug-play-modules

# 初始化模块的权重为正态分布
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)  # 权重初始化为正态分布
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # 偏置初始化为常数

# 初始化模块的权重为常数
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)  # 权重初始化为指定常数
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # 偏置初始化为指定常数

# 自适应采样模块
class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        """
        参数说明：
        - in_channels: 输入通道数
        - scale: 放大倍数
        - style: 模式，支持 'lp'（放大后处理）和 'pl'（先放大处理后卷积）
        - groups: 组卷积的组数
        - dyscope: 是否启用动态范围调整
        """
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl'], "style 必须为 'lp' 或 'pl'"  # 模式验证
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0, "通道数必须大于等于 scale² 并可整除"
        assert in_channels >= groups and in_channels % groups == 0, "通道数必须大于等于组数并可整除"

        # 根据 style 调整通道数
        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        # 偏移量卷积层
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)  # 初始化偏移量卷积层

        # 如果启用动态范围调整，则添加范围控制卷积层
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)  # 初始化范围控制卷积层

        # 注册初始位置
        self.register_buffer('init_pos', self._init_pos())

    # 初始化位置信息
    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    # 采样函数
    def sample(self, x, offset):
        """
        参数：
        - x: 输入特征图
        - offset: 偏移量
        返回：
        - 采样后的特征图
        """
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)  # 将偏移量分为坐标分量
        coords_h = torch.arange(H) + 0.5  # 高度坐标
        coords_w = torch.arange(W) + 0.5  # 宽度坐标
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)  # 坐标归一化因子
        coords = 2 * (coords + offset) / normalizer - 1  # 转换为归一化坐标
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).reshape(B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    # 'lp' 模式的前向传播
    def forward_lp(self, x):
        if hasattr(self, 'scope'):  # 如果启用动态范围调整
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:  # 普通偏移量计算
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    # 'pl' 模式的前向传播
    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)  # 放大后处理
        if hasattr(self, 'scope'):  # 如果启用动态范围调整
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:  # 普通偏移量计算
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    # 总的前向传播函数
    def forward(self, x):
        if self.style == 'pl':  # 根据 style 调用不同的前向传播
            return self.forward_pl(x)
        return self.forward_lp(x)

# 测试代码
if __name__ == '__main__':
    block = DySample(32)  # 创建 DySample 模块，输入通道为 32
    input = torch.rand(1, 32, 128, 128)  # 模拟输入张量，形状为 [1, 32, 128, 128]
    output = block(input)  # 前向传播
    print(input.size())  # 打印输入张量形状
    print(output.size())  # 打印输出张量形状
