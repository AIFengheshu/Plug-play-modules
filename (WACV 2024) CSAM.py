import torch
import torch.nn as nn
import torch.distributions as td

# ������Ŀ��CSAM: A 2.5D Cross-Slice Attention Module for Anisotropic Volumetric Medical Image Segmentation
# ������Ŀ:  CSAM: һ�����ڸ����������ҽѧͼ��ָ��2.5D������Ƭע����ģ��
# �������ӣ�https://arxiv.org/pdf/2311.04942
# �ٷ�github��https://github.com/aL3x-O-o-Hung/CSAM
# �������������ݴ�ѧ��ɼ��У
# �ؼ��ʣ���Ƭע������U-Net��ҽѧͼ��ָ�

# ΢�Ź��ںţ�AI����� https://github.com/AIFengheshu/Plug-play-modules

def custom_max(x,dim,keepdim=True):
    temp_x=x
    for i in dim:
        temp_x=torch.max(temp_x,dim=i,keepdim=True)[0]
    if not keepdim:
        temp_x=temp_x.squeeze()
    return temp_x

class PositionalAttentionModule(nn.Module):
    def __init__(self):
        super(PositionalAttentionModule,self).__init__()
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=(7,7),padding=3)
    def forward(self,x):
        max_x=custom_max(x,dim=(0,1),keepdim=True)
        avg_x=torch.mean(x,dim=(0,1),keepdim=True)
        att=torch.cat((max_x,avg_x),dim=1)
        att=self.conv(att)
        att=torch.sigmoid(att)
        return x*att

class SemanticAttentionModule(nn.Module):
    def __init__(self,in_features,reduction_rate=16):
        super(SemanticAttentionModule,self).__init__()
        self.linear=[]
        self.linear.append(nn.Linear(in_features=in_features,out_features=in_features//reduction_rate))
        self.linear.append(nn.ReLU())
        self.linear.append(nn.Linear(in_features=in_features//reduction_rate,out_features=in_features))
        self.linear=nn.Sequential(*self.linear)
    def forward(self,x):
        max_x=custom_max(x,dim=(0,2,3),keepdim=False).unsqueeze(0)
        avg_x=torch.mean(x,dim=(0,2,3),keepdim=False).unsqueeze(0)
        max_x=self.linear(max_x)
        avg_x=self.linear(avg_x)
        att=max_x+avg_x
        att=torch.sigmoid(att).unsqueeze(-1).unsqueeze(-1)
        return x*att

class SliceAttentionModule(nn.Module):
    def __init__(self,in_features,rate=4,uncertainty=True,rank=5):
        super(SliceAttentionModule,self).__init__()
        self.uncertainty=uncertainty
        self.rank=rank
        self.linear=[]
        self.linear.append(nn.Linear(in_features=in_features,out_features=int(in_features*rate)))
        self.linear.append(nn.ReLU())
        self.linear.append(nn.Linear(in_features=int(in_features*rate),out_features=in_features))
        self.linear=nn.Sequential(*self.linear)
        if uncertainty:
            self.non_linear=nn.ReLU()
            self.mean=nn.Linear(in_features=in_features,out_features=in_features)
            self.log_diag=nn.Linear(in_features=in_features,out_features=in_features)
            self.factor=nn.Linear(in_features=in_features,out_features=in_features*rank)
    def forward(self,x):
        max_x=custom_max(x,dim=(1,2,3),keepdim=False).unsqueeze(0)
        avg_x=torch.mean(x,dim=(1,2,3),keepdim=False).unsqueeze(0)
        max_x=self.linear(max_x)
        avg_x=self.linear(avg_x)
        att=max_x+avg_x
        if self.uncertainty:
            temp=self.non_linear(att)
            mean=self.mean(temp)
            diag=self.log_diag(temp).exp()
            factor=self.factor(temp)
            factor=factor.view(1,-1,self.rank)
            dist=td.LowRankMultivariateNormal(loc=mean,cov_factor=factor,cov_diag=diag)
            att=dist.sample()
        att=torch.sigmoid(att).squeeze().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x*att


class CSAM(nn.Module):
    def __init__(self,num_slices,num_channels,semantic=True,positional=True,slice=True,uncertainty=True,rank=5):
        super(CSAM,self).__init__()
        self.semantic=semantic
        self.positional=positional
        self.slice=slice
        if semantic:
            self.semantic_att=SemanticAttentionModule(num_channels)
        if positional:
            self.positional_att=PositionalAttentionModule()
        if slice:
            self.slice_att=SliceAttentionModule(num_slices,uncertainty=uncertainty,rank=rank)
    def forward(self,x):
        if self.semantic:
            x=self.semantic_att(x)
        if self.positional:
            x=self.positional_att(x)
        if self.slice:
            x=self.slice_att(x)
        return x
    
if __name__ == "__main__":
    # ������Ҫע��batchsize������������������Ƭ��
    batch_size = 1
    csam = CSAM(batch_size, num_channels=32)
    # ��������������� (B, C, H, W)
    input_tensor = torch.rand(1, 32, 256, 256)
    # ��ӡ������������״
    print(f"������������״: {input_tensor.shape}")
    # ǰ�򴫲�
    output_tensor = csam(input_tensor)
    # ��ӡ�����������״
    print(f"�����������״: {output_tensor.shape}")