import torch
from torch import nn

# 论文题目：SLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized Batch Normalization
# 中文题目：SLAB：具有简化线性注意力和渐进式重参数化批量归一化的高效变换器
# 论文链接：https://arxiv.org/pdf/2405.11582
# 官方github：https://github.com/xinghaochen/SLAB
# 所属机构：华为诺亚方舟实验室

# 代码整理:微信公众号:AI缝合术

# 源代码, 处理三维数据
class RepBN(nn.Module):
    def __init__(self, channels):
        super(RepBN, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn(x) + self.alpha * x
        x = x.transpose(1, 2)
        return x

# 可扩展到四维, 代码整理:微信公众号:AI缝合术 
class RepBN2d(nn.Module):
    def __init__(self, channels):
        super(RepBN2d, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm2d(channels)  # 使用BatchNorm2d

    def forward(self, x):
        # BatchNorm2d处理四维输入数据
        x = self.bn(x) + self.alpha * x  # BatchNorm + alpha * input
        return x
        
# if __name__ == "__main__":
#     # 模块参数
#     batch_size = 1    # 批大小
#     channels = 32     # 输入特征通道数
#     N = 16 * 16      # 图像高度*宽度 height * width

#     model = RepBN(channels = channels)
#     print(model)
#     print("微信公众号:AI缝合术, nb!")

#     # 生成随机输入张量 (batch_size, channels, height * width (N))
#     x = torch.randn(batch_size, N, channels)
#     # 打印输入张量的形状
#     print("Input shape:", x.shape)
#     # 前向传播计算输出
#     output = model(x)
#     # 打印输出张量的形状
#     print("Output shape:", output.shape)

if __name__ == "__main__":
    # 模块参数
    batch_size = 1    # 批大小
    channels = 32     # 输入特征通道数
    height = 256      # 图像高度
    width = 256        # 图像宽度

    model = RepBN2d(channels = channels)
    print(model)
    print("微信公众号:AI缝合术, nb!")

    # 生成随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)
    # 打印输入张量的形状
    print("Input shape:", x.shape)
    # 前向传播计算输出
    output = model(x)
    # 打印输出张量的形状
    print("Output shape:", output.shape)