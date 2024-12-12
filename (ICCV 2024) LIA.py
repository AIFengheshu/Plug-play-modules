import torch
import torch.nn as nn
import torch.nn.functional as F

# ������Ŀ��PlainUSR: Chasing Faster ConvNet for Efficient Super-Resolution
# ������Ŀ:  PlainUSR: ׷�����ľ����������ʵ�ָ�Ч�ĳ��ֱ���
# �������ӣ�https://openaccess.thecvf.com/content/ACCV2024/papers/Wang_PlainUSR_Chasing_Faster_ConvNet_for_Efficient_Super-Resolution_ACCV_2024_paper.pdf
# �ٷ�github��https://github.com/icandle/PlainUSR
# �����������Ͽ���ѧ�������ѧѧԺ
# �ؼ��ʣ����ֱ��ʣ��ز�������ע����
# ��������΢�Ź��ںš�AI�������

class SoftPooling2D(torch.nn.Module):
    def __init__(self,kernel_size,stride=None,padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,stride,padding, count_include_pad=False)
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool 
    
class LocalAttention(nn.Module):
    ''' attention based on local importance'''
    def __init__(self, channels, f=16):
        super().__init__()
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )            
    def forward(self, x):
        ''' forward '''
        # interpolate the heat map
        g = self.gate(x[:,:1].clone())
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return x * w * g #(w + g) #self.gate(x, w) 

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    la = LocalAttention(channels=32).to(device)  # ��ģ���ƶ����豸
    input = torch.rand(1, 32, 256, 256).to(device)  # �����������ƶ����豸

    output = la(input)
    print(f"\nInput shape: {input.shape}")
    print(f"Output shape: {output.shape}")