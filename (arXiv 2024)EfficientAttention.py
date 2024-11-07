import torch
from torch import nn
from torch.nn import functional as f

# 论文题目：Efficient Attention: Attention with Linear Complexities
# 中文题目:  高效注意力：具有线性复杂度的注意力机制
# 论文链接：https://arxiv.org/pdf/1812.01243
# 官方github：https://github.com/cmsflash/efficient-attention

# 代码整理与注释：公众号：AI缝合术
# AI缝合术github：https://github.com/AIFengheshu/Plug-play-modules

# EfficientAttention 模块：一个高效的多头注意力机制
class EfficientAttention(nn.Module):
    
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        """
        初始化函数，定义了 EfficientAttention 模块的各个层
        :param in_channels: 输入特征的通道数
        :param key_channels: 键的通道数
        :param head_count: 注意力头的数量
        :param value_channels: 值的通道数
        """
        super().__init__()
        self.in_channels = in_channels  # 输入通道数
        self.key_channels = key_channels  # 键的通道数
        self.head_count = head_count  # 注意力头数
        self.value_channels = value_channels  # 值的通道数

        # 1x1 卷积层，用于生成键（keys）、查询（queries）和值（values）
        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)

        # 最后的 1x1 卷积用于将注意力输出映射回输入通道数
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        """
        前向传播函数，计算输入的注意力输出
        :param input_: 输入特征图 (batch_size, in_channels, height, width)
        :return: 输出经过注意力加权后的特征图
        """
        n, _, h, w = input_.size()  # 获取输入特征图的形状 (batch_size, channels, height, width)
        
        # 计算键、查询和值
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))  # reshape 为 (batch_size, key_channels, height * width)
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)  # reshape 为 (batch_size, key_channels, height * width)
        values = self.values(input_).reshape((n, self.value_channels, h * w))  # reshape 为 (batch_size, value_channels, height * width)
        
        # 计算每个头的键通道数和值通道数
        head_key_channels = self.key_channels // self.head_count  # 每个头的键的通道数
        head_value_channels = self.value_channels // self.head_count  # 每个头的值的通道数
        
        attended_values = []  # 用于存储每个头的输出

        # 对每个头进行注意力计算
        for i in range(self.head_count):
            # 从键和值中提取当前头的部分
            key = f.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = f.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            
            # 计算注意力
            context = key @ value.transpose(1, 2)  # 键和值的乘积，得到上下文信息
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)  # 查询和上下文的乘积
            attended_values.append(attended_value)  # 将每个头的输出添加到列表中

        # 合并所有头的输出
        aggregated_values = torch.cat(attended_values, dim=1)  # 将所有头的输出按通道拼接
        reprojected_value = self.reprojection(aggregated_values)  # 使用 1x1 卷积对拼接后的结果进行映射
        
        # 最终输出是加上输入的残差连接
        attention = reprojected_value + input_

        return attention  # 返回加权后的特征图

# 测试 EfficientAttention 模块
if __name__ == '__main__':
    # 创建一个 EfficientAttention 实例
    attention = EfficientAttention(in_channels=64, key_channels=128, head_count=4, value_channels=128)
    
    # 创建一个随机输入张量，模拟输入特征图 (batch_size=1, in_channels=64, height=32, width=32)
    input_tensor = torch.randn(1, 64, 32, 32)

    # 通过注意力模块进行前向传播
    output = attention(input_tensor)
    
    # 打印输入和输出张量的形状
    print(f'输入形状: {input_tensor.shape}')
    print(f'输出形状: {output.shape}')
