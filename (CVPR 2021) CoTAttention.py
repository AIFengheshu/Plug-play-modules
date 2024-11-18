import torch
from torch import nn
from torch.nn import functional as F

# 论文题目：Contextual Transformer Networks for Visual Recognition
# 中文题目:  视觉识别的上下文变换网络
# 论文链接：https://arxiv.org/pdf/2107.12292
# 官方github：https://github.com/JDAI-CV/CoTNet
# 所属机构：京东AI研究院
# 关键词：视觉识别、上下文信息、Transformer、自注意力机制、计算机视觉
# 微信公众号：AI缝合术   https://github.com/AIFengheshu/Plug-play-modules

class CoTAttention(nn.Module):

    def __init__(self, dim=512,kernel_size=3):
        super().__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim)
        )
        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )

    def forward(self, x):
        bs,c,h,w=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w)
        return k1+k2

if __name__ == '__main__':
    input=torch.randn(1,32,256,256)
    cot = CoTAttention(dim=32,kernel_size=3)
    output=cot(input)
    print(input.shape)
    print(output.shape)
