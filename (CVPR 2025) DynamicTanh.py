import torch
import torch.nn as nn

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5, channels_last=True):
        """
        动态 Tanh 激活层,  可以替代 LayerNorm 层。
        Args:
            normalized_shape (tuple): 输入张量的归一化形状
            alpha_init_value (float): 初始的 alpha 值,  用于调节 Tanh 激活的“陡峭度”
            channels_last (bool): 如果为 True,  则按最后一维通道进行加权,  否者按其他维度加权
        """
        super(DynamicTanh, self).__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        # 定义需要学习的参数
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        """
        前向传播：对输入应用动态 Tanh 激活,  并进行加权。
        """
        x = torch.tanh(self.alpha * x)  # 应用动态 Tanh 激活

        # 根据 channels_last 参数确定加权方式
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]

        return x

if __name__ == '__main__':
    # 参数设置
    batch_size = 1               # 批量大小
    in_channels = 32             # 输入通道数
    height, width = 256, 256     # 输入图像的高度和宽度

    # 创建随机输入张量, 形状为 (batch_size, in_channels, height, width)
    x = torch.randn(batch_size, in_channels, height, width)

    model = DynamicTanh(normalized_shape=(in_channels, height, width))

    # 打印模型结构
    print(model)
    print("微信公众号: AI缝合术!")

    # 进行前向传播, 得到输出
    output = model(x)

    # 打印输入和输出的形状
    print(f"输入张量的形状: {x.shape}")
    print(f"输出张量的形状: {output.shape}")
