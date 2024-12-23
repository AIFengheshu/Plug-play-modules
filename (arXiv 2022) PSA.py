import torch
import torch.nn as nn
from torch.nn import init

# 论文题目：EPSANet: An Efficient Pyramid Squeeze Attention Block on Convolutional Neural Network
# 中文题目：EPSANet：一种高效的金字塔挤压注意力块在卷积神经网络上
# 论文链接：https://arxiv.org/pdf/2105.14447
# 官方github：https://github.com/murufeng/EPSANet
# 所属机构：深圳大学，西安交通大学
# 代码整理：微信公众号《AI缝合术》

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.rand(1, 32, 256, 256)
    attention_module = PSA(channel=32,reduction=8)
    output_tensor = attention_module(input_tensor)
    # 打印输入和输出的形状
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")
