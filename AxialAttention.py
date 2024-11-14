import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 论文题目：Medical Transformer: Gated Axial-Attention forMedical Image Segmentation
# 中文题目:  医疗Transformer：用于医学图像分割的门控轴向注意力机制
# 论文链接：https://arxiv.org/pdf/2102.10662
# 官方github：https://github.com/jeya-maria-jose/Medical-Transformer
# 所属机构：约翰霍普金斯大学, 新泽西州立大学
# 关键词： Transformer, 医学图像分割, 自注意力机制

# 定义1x1卷积，用于改变通道数
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# 定义一个一维卷积层，用于进行qkv变换
class qkv_transform(nn.Conv1d):
    """用于qkv变换的Conv1d"""

# 定义轴向注意力模块（Axial Attention）
class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        # 确保输入通道数和输出通道数都能被组数整除
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes  # 输入通道数
        self.out_planes = out_planes  # 输出通道数
        self.groups = groups  # 注意力头的组数
        self.group_planes = out_planes // groups  # 每组通道数
        self.kernel_size = kernel_size  # 卷积核大小
        self.stride = stride  # 步长
        self.bias = bias  # 是否使用偏置
        self.width = width  # 是否对宽度进行轴向注意力

        # 多头自注意力的qkv变换层
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        # 归一化层
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # 位置编码
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        # 相对位置索引
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))

        # 如果步长大于1，则添加池化层
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        # 初始化参数
        self.reset_parameters()

    def forward(self, x):
        # 根据宽度或高度调整输入张量的维度顺序
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H

        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # 进行qkv变换
        qkv = self.bn_qkv(self.qkv_transform(x))
        # 将qkv分解为q、k、v
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # 计算位置编码
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)

        # 计算q和位置编码的乘积
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        # 组合相似性矩阵并归一化
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked_similarity, dim=3)

        # 加权求和得到v的注意力输出
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        # 恢复张量的维度顺序
        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        # 如果步长大于1，则进行池化
        if self.stride > 1:
            output = self.pooling(output)

        return output

    # 参数初始化
    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

# 测试代码
if __name__ == '__main__':
    input = torch.randn(1, 64, 224, 224)  # 创建一个测试输入，大小为 1x64x224x224
    block = AxialAttention(in_planes=64, out_planes=64, groups=1, kernel_size=224, stride=1, bias=False, width=False)
    # in_planes、out_planes和channel一致，kernel_size和h,w一致
    output = block(input)
    print(input.size())  # 输出输入张量的尺寸
    print(output.size())  # 输出输出张量的尺寸
