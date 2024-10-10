import numpy as np
import torch
from torch import nn
from torch.nn import init

# 论文地址：https://arxiv.org/pdf/2105.14447
# 论文：EPSANet: An Efficient Pyramid Squeeze Attention Block on Convolutional Neural Network
# 微信公众号：AI缝合术
"""
2024年全网最全即插即用模块,全部免费!包含各种卷积变种、最新注意力机制、特征融合模块、上下采样模块，
适用于人工智能(AI)、深度学习、计算机视觉(CV)领域，适用于图像分类、目标检测、实例分割、语义分割、
单目标跟踪(SOT)、多目标跟踪(MOT)、红外与可见光图像融合跟踪(RGBT)、图像去噪、去雨、去雾、去模糊、超分等任务，
模块库持续更新中......
https://github.com/AIFengheshu/Plug-play-modules
"""

class PSA(nn.Module):

    def __init__(self, channel=512,reduction=4,S=4):
        super().__init__()
        self.S=S

        self.convs=[]
        for i in range(S):
            self.convs.append(nn.Conv2d(channel//S,channel//S,kernel_size=2*(i+1)+1,padding=i+1))

        self.se_blocks=[]
        for i in range(S):
            self.se_blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel//S, channel // (S*reduction),kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // (S*reduction), channel//S,kernel_size=1, bias=False),
                nn.Sigmoid()
            ))
        
        self.softmax=nn.Softmax(dim=1)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()

        #Step1:SPC module
        SPC_out=x.view(b,self.S,c//self.S,h,w) #bs,s,ci,h,w
        for idx,conv in enumerate(self.convs):
            SPC_out[:,idx,:,:,:]=conv(SPC_out[:,idx,:,:,:])

        #Step2:SE weight
        se_out=[]
        for idx,se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out[:,idx,:,:,:]))
        SE_out=torch.stack(se_out,dim=1)
        SE_out=SE_out.expand_as(SPC_out)

        #Step3:Softmax
        softmax_out=self.softmax(SE_out)

        #Step4:SPA
        PSA_out=SPC_out*softmax_out
        PSA_out=PSA_out.view(b,-1,h,w)

        return PSA_out


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    psa = PSA(channel=512,reduction=8)
    output=psa(input)
    a=output.view(-1).sum()
    a.backward()
    print(output.shape)

    