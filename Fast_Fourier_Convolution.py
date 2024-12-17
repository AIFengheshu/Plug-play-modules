import torch
import torch.nn as nn
import time

# 论文题目：Fast Fourier Convolution
# 中文题目：快速傅里叶卷积
# 论文链接：https://proceedings.neurips.cc/paper_files/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf
# 官方github：无
# 所属机构：王选计算机技术研究所，北京大学数据科学中心
# 核心速览：本文提出了一种名为快速傅里叶卷积（Fast Fourier Convolution, FFC）的新型卷积操作符，
# 旨在实现非局部感受野和跨尺度融合，以提高深度网络在处理图像和视频任务时的性能。
# 代码整理：微信公众号《AI缝合术》

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()
        # 进行傅里叶变换，返回复数结果
        ffted = torch.fft.fft2(x, norm="ortho")
        ffted_real = ffted.real  # 提取实部
        ffted_imag = ffted.imag  # 提取虚部
        # 将复数分解成两个通道：实部和虚部拼接
        ffted = torch.cat([ffted_real, ffted_imag], dim=1)
        # 卷积操作
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        # 将卷积后的实部和虚部重新分离
        ffted_real, ffted_imag = torch.chunk(ffted, 2, dim=1)
        ffted = torch.complex(ffted_real, ffted_imag)  # 重新组合为复数
        # 进行逆傅里叶变换
        output = torch.fft.ifft2(ffted, s=(h, w), norm="ortho").real  # 只保留实部输出
        return output

class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()
        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0
        output = self.conv2(x + output + xs)
        return output

class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin=0.5, ratio_gout=0.5, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin=0.5, ratio_gout=0.5,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 enable_lfu=True):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
        self.bn_g = gnorm(int(out_channels * ratio_gout))
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        # 手动将输入划分为局部和全局部分
        c = x.shape[1]  # 获取通道数
        c_l = int(c * (1 - self.ffc.ratio_gin))  # 局部通道数
        x_l, x_g = x[:, :c_l, :, :], x[:, c_l:, :, :]  # 分割为局部和全局

        # 调用 FFC 模块进行前向传播
        x_l, x_g = self.ffc((x_l, x_g))
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        
        # 将局部和全局部分合并为整体输出
        return torch.cat((x_l, x_g), dim=1)

if __name__ == '__main__':
    print("当前系统时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print("微信公众号:AI缝合术,The test was successful!")

    ffc = FFC_BN_ACT(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1).to(device) 
    input = torch.rand(1, 16, 128, 128).to(device)  # 输入张量
    output = ffc(input)  # 前向传播
    print(f"\n输入张量形状: {input.shape}")
    print(f"输出张量形状: {output.shape}")
