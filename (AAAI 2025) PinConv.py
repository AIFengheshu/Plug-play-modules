import torch
import torch.nn as nn

# ������Ŀ��Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection
# ������Ŀ���糵��״�ľ���ͻ��ڳ߶ȵĶ�̬��ʧ���ں���СĿ����
# �������ӣ�https://arxiv.org/pdf/2412.16986
# �ٷ�github��https://github.com/JN-Yang/PConv-SDloss-Data
# �������������ϿƼ���ѧ��Ϣ�빤��ѧԺ���Ͼ�����ѧ�������ѧ����ѧԺ
# ��������΢�Ź��ںš�AI�������

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Pinwheel_shapedConv(nn.Module):  
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''
    
    def __init__(self, c1, c2, k, s):
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


    
if __name__ == "__main__":

    # ��ģ���ƶ��� GPU��������ã�
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ���������������� (batch_size, channels, height, width)
    x = torch.randn(1, 32, 256, 256).to(device)

    # ��ʼ�� pconv ģ��
    pconv = Pinwheel_shapedConv(c1=32, c2=64, k=3, s=1)
    print(pconv)
    pconv = pconv.to(device)

    # ǰ�򴫲�
    output = pconv(x)

    # ��ӡ����������������״
    print("����������״:", x.shape)
    print("���������״:", output.shape)
