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
        type can be:'pos' - position encoding
                    'regularised' 
        '''
 
        self.enctype = enctype
        self.enc_channels = enc_channels
 
        if enctype == 'pos':
 
            encoding_layers = kwargs['encoding_layers']
 
            x_coord, y_coord = torch.linspace(-1, 1, int(size[0])), torch.linspace(-1, 1, int(size[1]))
 
            self.register_buffer('pos', torch.stack(torch.meshgrid((x_coord, y_coord), indexing='ij'), dim=-1).view(-1,
                                                                                                                    2))  # .to(device)
 
            self.encoder = nn.Sequential()
            for i in range(encoding_layers):
                if i == 0:
                    self.encoder.add_module('linear' + str(i), nn.Linear(2, enc_channels))
                else:
                    self.encoder.add_module('linear' + str(i), nn.Linear(enc_channels, enc_channels))
 
        elif (enctype == 'map') or (enctype == 'bilinear'):
 
            initialiser = kwargs['initialiser']
 
            if 'downsample' in kwargs.keys():
                self.sample = kwargs['downsample']
            else:
                self.sample = [1, 1]
 
            self.map = nn.Parameter(initialiser)
 
        self.conv = nn.Conv2d(enc_channels, out_channels, 1, padding='same')
 
        self.relu = nn.ReLU()
 
    def forward(self, x):
        '''
        x is (bs,nc,nx,ny)
        '''
        if self.enctype == 'pos':
 
            gate = self.encoder(self.pos).view(1, x.shape[2], x.shape[3], x.shape[1]).permute(0, 3, 1, 2)
            gate = torch.nn.functional.relu(gate)  # ?
            x = self.conv(x * gate)
            return x
 
 
        elif self.enctype == 'map':
 
            map = self.relu(self.map).repeat_interleave(self.sample[0], dim=2).repeat_interleave(self.sample[1], dim=3)
 
            x = self.conv(x * map)
            return x
 
        elif self.enctype == 'bilinear':
 
            # if self.enc_channels == 9:
            map = create_bilinear_coeff_map_cart_3x3(self.map[:, 0:1], self.map[:, 1:2])
            # else:
            #     map = create_bilinear_coeff_map_cart_5x5(angles,distances)
 
            map = self.relu(map).repeat_interleave(self.sample[0], dim=2).repeat_interleave(self.sample[1], dim=3)
 
            x = self.conv(x * map)
            return x
 
 
def create_bilinear_coeff_map_cart_3x3(x_disp, y_disp):
    shape = x_disp.shape
    x_disp = x_disp.reshape(-1)
    y_disp = y_disp.reshape(-1)
 
    # Determine the quadrant based on the signs of the displacements
    primary_indices = torch.zeros_like(x_disp, dtype=torch.long)
    primary_indices[(x_disp >= 0) & (y_disp >= 0)] = 0  # Quadrant 1
    primary_indices[(x_disp < 0) & (y_disp >= 0)] = 2  # Quadrant 2
    primary_indices[(x_disp < 0) & (y_disp < 0)] = 4  # Quadrant 3
    primary_indices[(x_disp >= 0) & (y_disp < 0)] = 6  # Quadrant 4
    # Define the number of directions
    num_directions = 8
 
    # Compute the indices for the primary and secondary directions
    secondary_indices = ((primary_indices + 1) % num_directions).long()
    tertiary_indices = (primary_indices - 1).long()
    tertiary_indices[tertiary_indices < 0] = num_directions - 1
 
    x_disp = x_disp.abs()
    y_disp = y_disp.abs()
 
    coeffs = torch.zeros((x_disp.size(0), num_directions + 1), device=x_disp.device)
    batch_indices = torch.arange(x_disp.size(0), device=x_disp.device)
 
    coeffs[batch_indices, primary_indices] = (x_disp * y_disp)
    coeffs[batch_indices, secondary_indices] = x_disp * (1 - y_disp)
    coeffs[batch_indices, tertiary_indices] = (1 - x_disp) * y_disp
    coeffs[batch_indices, -1] = (1 - x_disp) * (1 - y_disp)
 
    swappers = (primary_indices == 0) | (primary_indices == 4)
 
    coeffs[batch_indices[swappers], secondary_indices[swappers]] = (1 - x_disp[swappers]) * y_disp[swappers]
    coeffs[batch_indices[swappers], tertiary_indices[swappers]] = x_disp[swappers] * (1 - y_disp[swappers])
 
    coeffs = coeffs.view(shape[0], shape[2], shape[3], num_directions + 1).permute(0, 3, 1, 2)
    reorderer = [0, 1, 2, 7, 8, 3, 6, 5, 4]
 
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