import torch
from torch import nn
from torch.nn import init

# 论文题目：Spatial Group-wise Enhance: Improving Semantic
# Feature Learning in Convolutional Networks

# 中文题目:  空间分组增强：在卷积网络中改进语义特征学习
# 论文链接：https://arxiv.org/pdf/1905.09646
# 官方github：https://github.com/implus/PytorchInsight
# 所属机构：南京理工大学PCALab、Momenta、清华大学
# 关键词：卷积神经网络、注意力机制、图像分类、目标检测、特征增强

# 微信公众号：AI缝合术 https://github.com/AIFengheshu/Plug-play-modules

class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups=8):
        super().__init__()
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight=nn.Parameter(torch.zeros(1,groups,1,1))
        self.bias=nn.Parameter(torch.zeros(1,groups,1,1))
        self.sig=nn.Sigmoid()
        self.init_weights()

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
        b, c, h,w=x.shape
        x=x.view(b*self.groups,-1,h,w) #bs*g,dim//g,h,w
        xn=x*self.avg_pool(x) #bs*g,dim//g,h,w
        xn=xn.sum(dim=1,keepdim=True) #bs*g,1,h,w
        t=xn.view(b*self.groups,-1) #bs*g,h*w

        t=t-t.mean(dim=1,keepdim=True) #bs*g,h*w
        std=t.std(dim=1,keepdim=True)+1e-5
        t=t/std #bs*g,h*w
        t=t.view(b,self.groups,h,w) #bs,g,h*w
        
        t=t*self.weight+self.bias #bs,g,h*w
        t=t.view(b*self.groups,1,h,w) #bs*g,1,h*w
        x=x*self.sig(t)
        x=x.view(b,c,h,w)
        return x 

if __name__ == '__main__':
    input=torch.randn(1,32,256,256)
    sge = SpatialGroupEnhance(groups=8)
    output=sge(input)
    print(output.shape)
