import torch
from torch import nn

# 论文题目：CoordGate: Efficiently Computing Spatially-Varying Convolutions in Convolutional Neural Networks
# 中文题目：CoordGate：在卷积神经网络中高效计算空间变化卷积
# 论文链接：https://arxiv.org/pdf/2401.04680
# 官方github：无
# 所属机构：牛津大学克拉伦登实验室物理系，慕尼黑大学路德维希-马克西米利安物理学院，约翰·亚当斯加速器科学研究所
# 代码整理：微信公众号《AI缝合术》

class CoordGate(nn.Module):
    def __init__(self, enc_channels, out_channels, size: list = [256, 256], enctype='pos', **kwargs):
        super(CoordGate, self).__init__()
        '''
        参数解释：
        - enc_channels: 编码器的通道数。
        - out_channels: 输出的通道数。
        - size: 输入特征图的空间尺寸，默认为 [256, 256]。
        - enctype: 编码类型，可选 'pos' (位置编码), 'map', 或 'bilinear'。
        - **kwargs: 额外参数，根据不同编码类型使用。
        '''
        
        self.enctype = enctype  # 编码类型
        self.enc_channels = enc_channels  # 编码通道数

        if enctype == 'pos':  # 如果是位置编码类型
            encoding_layers = kwargs['encoding_layers']  # 获取编码层数
            
            # 创建 x 和 y 方向的坐标范围，取值范围为 [-1, 1]
            x_coord, y_coord = torch.linspace(-1, 1, int(size[0])), torch.linspace(-1, 1, int(size[1]))
            
            # 注册坐标网格缓冲区
            self.register_buffer('pos', torch.stack(torch.meshgrid((x_coord, y_coord), indexing='ij'), dim=-1).view(-1, 2))
            
            # 定义编码器，使用线性层实现
            self.encoder = nn.Sequential()
            for i in range(encoding_layers):
                if i == 0:
                    self.encoder.add_module('linear' + str(i), nn.Linear(2, enc_channels))
                else:
                    self.encoder.add_module('linear' + str(i), nn.Linear(enc_channels, enc_channels))

        elif (enctype == 'map') or (enctype == 'bilinear'):  # 如果是地图或双线性类型
            initialiser = kwargs['initialiser']  # 获取初始值
            
            # 下采样因子
            if 'downsample' in kwargs.keys():
                self.sample = kwargs['downsample']
            else:
                self.sample = [1, 1]
            
            # 将地图注册为可训练参数
            self.map = nn.Parameter(initialiser)

        # 通用卷积层
        self.conv = nn.Conv2d(enc_channels, out_channels, 1, padding='same')

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        前向传播
        输入 x 的形状为 (batch_size, num_channels, height, width)
        '''
        if self.enctype == 'pos':  # 位置编码处理
            # 使用编码器生成门控矩阵
            gate = self.encoder(self.pos).view(1, x.shape[2], x.shape[3], x.shape[1]).permute(0, 3, 1, 2)
            gate = torch.nn.functional.relu(gate)  # 应用 ReLU 激活
            x = self.conv(x * gate)  # 乘以门控矩阵并通过卷积
            return x

        elif self.enctype == 'map':  # 地图编码处理
            # 处理 map 参数并重复采样到输入尺寸
            map = self.relu(self.map).repeat_interleave(self.sample[0], dim=2).repeat_interleave(self.sample[1], dim=3)
            x = self.conv(x * map)  # 乘以 map 并通过卷积
            return x

        elif self.enctype == 'bilinear':  # 双线性插值编码处理
            # 计算双线性插值的权重
            map = create_bilinear_coeff_map_cart_3x3(self.map[:, 0:1], self.map[:, 1:2])
            map = self.relu(map).repeat_interleave(self.sample[0], dim=2).repeat_interleave(self.sample[1], dim=3)
            x = self.conv(x * map)  # 乘以 map 并通过卷积
            return x

def create_bilinear_coeff_map_cart_3x3(x_disp, y_disp):
    '''
    创建双线性插值的权重映射，适用于 3x3 栅格。
    输入：
    - x_disp: x 方向的位移。
    - y_disp: y 方向的位移。
    输出：
    - coeffs: 双线性插值权重张量。
    '''
    shape = x_disp.shape
    x_disp = x_disp.reshape(-1)
    y_disp = y_disp.reshape(-1)

    # 确定位移的象限
    primary_indices = torch.zeros_like(x_disp, dtype=torch.long)
    primary_indices[(x_disp >= 0) & (y_disp >= 0)] = 0  # 第一象限
    primary_indices[(x_disp < 0) & (y_disp >= 0)] = 2  # 第二象限
    primary_indices[(x_disp < 0) & (y_disp < 0)] = 4  # 第三象限
    primary_indices[(x_disp >= 0) & (y_disp < 0)] = 6  # 第四象限
    
    num_directions = 8  # 方向数
    
    # 计算主要方向和次要方向的索引
    secondary_indices = ((primary_indices + 1) % num_directions).long()
    tertiary_indices = (primary_indices - 1).long()
    tertiary_indices[tertiary_indices < 0] = num_directions - 1

    x_disp = x_disp.abs()
    y_disp = y_disp.abs()

    coeffs = torch.zeros((x_disp.size(0), num_directions + 1), device=x_disp.device)
    batch_indices = torch.arange(x_disp.size(0), device=x_disp.device)

    # 分配权重系数
    coeffs[batch_indices, primary_indices] = (x_disp * y_disp)
    coeffs[batch_indices, secondary_indices] = x_disp * (1 - y_disp)
    coeffs[batch_indices, tertiary_indices] = (1 - x_disp) * y_disp
    coeffs[batch_indices, -1] = (1 - x_disp) * (1 - y_disp)

    # 象限调整
    swappers = (primary_indices == 0) | (primary_indices == 4)
    coeffs[batch_indices[swappers], secondary_indices[swappers]] = (1 - x_disp[swappers]) * y_disp[swappers]
    coeffs[batch_indices[swappers], tertiary_indices[swappers]] = x_disp[swappers] * (1 - y_disp[swappers])

    coeffs = coeffs.view(shape[0], shape[2], shape[3], num_directions + 1).permute(0, 3, 1, 2)
    reorderer = [0, 1, 2, 7, 8, 3, 6, 5, 4]  # 重排顺序

    return coeffs[:, reorderer, :, :]

if __name__ == '__main__':
    # 创建 CoordGate 模块的实例

    in_size=[256,256]
    encoding_layers = 2
    initialiser = torch.rand((32, 2))
    kwargs = {'encoding_layers': encoding_layers, 'initialiser': initialiser}

    block = CoordGate(32, 32, in_size, enctype = 'pos', **kwargs)
 
    # 生成随机输入数据
    input_data = torch.rand(1,32,256,256)
    output = block(input_data)
 
    # 打印输入和输出形状
    print("Input size:", input_data.size())
    print("Output size:", output.size())
