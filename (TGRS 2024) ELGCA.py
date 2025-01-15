import torch
import torch.nn as nn

# ������Ŀ��ELGC-Net: Efficient Local-Global Context Aggregation for Remote Sensing Change Detection
# ������Ŀ��ELGC-Net������ң�б仯���ĸ�Ч�ֲ�-ȫ�������ľۺ�����
# �������ӣ�https://arxiv.org/pdf/2403.17909
# �ٷ�github��https://github.com/techmn/elgcnet
# �����������º�Ĭ�¡�������Ү���˹����ܴ�ѧ��IBM �о����Ĵ����ǹ�����ѧ����ѩƽ��ѧ
# ��������΢�Ź��ںš�AI�������

class ELGCA(nn.Module):
    """
    Efficient local global context aggregation module
    dim: number of channels of input
    heads: number of heads utilized in computing attention
    """
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.dwconv = nn.Conv2d(dim//2, dim//2, 3, padding=1, groups=dim//2)
        self.qkvl = nn.Conv2d(dim//2, (dim//4)*self.heads, 1, padding=0)
        self.pool_q = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.pool_k = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        
        x1, x2 = torch.split(x, [C//2, C//2], dim=1)
        # apply depth-wise convolution on half channels
        x1 = self.act(self.dwconv(x1))

        # linear projection of other half before computing attention
        x2 = self.act(self.qkvl(x2))

        x2 = x2.reshape(B, self.heads, C//4, H, W)
        
        q = torch.sum(x2[:, :-3, :, :, :], dim=1)
        k = x2[:,-3, :, :, :]

        q = self.pool_q(q)
        k = self.pool_k(k)
        
        v = x2[:,-2,:,:,:].flatten(2)
        lfeat = x2[:,-1,:,:,:]
        
        qk = torch.matmul(q.flatten(2), k.flatten(2).transpose(1,2))
        qk = torch.softmax(qk, dim=1).transpose(1,2)

        x2 = torch.matmul(qk, v).reshape(B, C//4, H, W)
        
        x = torch.cat([x1, lfeat, x2], dim=1)

        return x


    
if __name__ == "__main__":

    # ��ģ���ƶ��� GPU��������ã�
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ���������������� (batch_size, channels, height, width)
    x = torch.randn(1, 32, 256, 256).to(device)

    # ��ʼ�� elgca ģ��
    elgca = ELGCA(dim=32, heads=4)
    print(elgca)
    elgca = elgca.to(device)

    # ǰ�򴫲�
    output = elgca(x)

    # ��ӡ����������������״
    print("����������״:", x.shape)
    print("���������״:", output.shape)
