import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文题目：Rethinking Fast Fourier Convolution in Image Inpainting
# 中文题目：图像修复中快速傅里叶卷积的再思考
# 论文链接：https://openaccess.thecvf.com/content/ICCV2023/papers/Chu_Rethinking_Fast_Fourier_Convolution_in_Image_Inpainting_ICCV_2023_paper.pdf
# 官方github：https://github.com/1911cty/Unbiased-Fast-Fourier-Convolution
# 所属机构：浙江大学计算机科学与技术学院，浙江工商大学
# 代码整理：微信公众号《AI缝合术》


class FourierUnit_modified(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit_modified, self).__init__()
        self.groups = groups

        self.input_shape = 32  # change!!!!!it!!!!!!manually!!!!!!
        self.in_channels = in_channels

        self.locMap = nn.Parameter(torch.rand(self.input_shape, self.input_shape//2 + 1))

        self.lambda_base = nn.Parameter(torch.tensor(0.),requires_grad=True)

        
        self.conv_layer_down55 = torch.nn.Conv2d(in_channels=in_channels * 2 + 1, # +1 for locmap
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups, bias=False, padding_mode = 'reflect')
        self.conv_layer_down55_shift = torch.nn.Conv2d(in_channels=in_channels * 2 + 1, # +1 for locmap
                                          out_channels=out_channels * 2,
                                          kernel_size=3, stride=1, padding=2, dilation=2, groups=self.groups, bias=False, padding_mode = 'reflect')
        

        self.norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        
        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm
    

        self.img_freq = None
        self.distill = None


    def forward(self, x): 
        batch = x.shape[0]


        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)


        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])


        locMap = self.locMap.expand_as(ffted[:,:1,:,:]) # B 1 H' W'
        ffted_copy= ffted.clone()

        cat_img_mask_freq = torch.cat((ffted[:,:self.in_channels,:,:], 
                                    ffted[:,self.in_channels:,:,:], 
                                    locMap),dim = 1)

        ffted = self.conv_layer_down55( cat_img_mask_freq )
        ffted = torch.fft.fftshift(ffted, dim = -2)

        ffted = self.relu(ffted)
        

        locMap_shift = torch.fft.fftshift(locMap, dim = -2) ## ONLY IF NOT SHIFT BACK

        # REPEAT CONV
        cat_img_mask_freq1 = torch.cat((ffted[:,:self.in_channels,:,:], 
                                    ffted[:,self.in_channels:,:,:], 
                                    locMap_shift),dim = 1)                        

        ffted = self.conv_layer_down55_shift( cat_img_mask_freq1 )
        ffted = torch.fft.ifftshift(ffted, dim = -2)


        lambda_base = torch.sigmoid(self.lambda_base)

        ffted = ffted_copy * lambda_base + ffted * (1-lambda_base)


        # irfft
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)


        epsilon = 0.5
        output = output - torch.mean(output) + torch.mean(x)
        output = torch.clip(output ,  float(x.min()-epsilon), float(x.max()+epsilon))
        
        self.distill = output # for self perc
        return output
    
if __name__ == '__main__':

    # 实例化 FourierUnit_modified 模块
    model = FourierUnit_modified(
        in_channels=64,
        out_channels=64,
        groups=1,
        spatial_scale_factor=None,
        spectral_pos_encoding=False,
        use_se=False,
        ffc3d=False,
        fft_norm='ortho'
    )
    # 打印模型结构
    print(model)
    # 输入张量
    x = torch.randn(1, 64, 32,32)
    output = model(x)
    # 打印输入和输出的形状
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)