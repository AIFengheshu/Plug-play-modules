import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

# ������Ŀ��Restoring Images in Adverse Weather Conditionsvia Histogram Transformer
# ������Ŀ���ڶ�������������ͨ��ֱ��ͼ Transformer �ָ�ͼ��
# �������ӣ�https://arxiv.org/pdf/2407.10172
# �ٷ� github��https://github.com/sunshangquan/Histoformer
# �����������й���ѧԺ��Ϣ�����о������й���ѧԺ��ѧ����ռ䰲ȫѧԺ����ɽ��ѧ����У��
# ��������Ϣ��ȫѧԺ��
# ��������΢�Ź��ںţ�AI�����

## Dynamic-range Histogram Self-Attention (DHSA)
class Attention_histogram(nn.Module):
    def __init__(self, dim, num_heads, bias, ifBox=True):
        super(Attention_histogram, self).__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*5, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*5, dim*5, kernel_size=3, stride=1, padding=1, groups=dim*5, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw//factor+1)*factor-hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad
    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:,:,t_pad[0]:hw-t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit  = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) #* self.weight + self.bias
    

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b, head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        b,c,h,w = x.shape
        x_sort, idx_h = x[:,:c//2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:,:c//2] = x_sort
        qkv = self.qkv_dwconv(self.qkv(x))
        q1,k1,q2,k2,v = qkv.chunk(5, dim=1) # b,c,x,x

        v, idx = v.view(b,c,-1).sort(dim=-1)
        q1 = torch.gather(q1.view(b,c,-1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b,c,-1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b,c,-1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b,c,-1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)
        
        out1 = torch.scatter(out1, 2, idx, out1).view(b,c,h,w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b,c,h,w)
        out = out1 * out2
        out = self.project_out(out)
        out_replace = out[:,:c//2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:,:c//2] = out_replace
        return out
    
if __name__ == "__main__":
   # ģ�����
    batch_size = 1    # ����С
    channels = 32     # ��������ͨ����
    height = 256      # ͼ��߶�
    width = 256        # ͼ����
    num_heads = 8     # ע����ͷ����
    
    # ���� Attention_histogram ģ��
    attention_histogram = Attention_histogram(dim=channels, num_heads=num_heads, bias=True, ifBox=True)
    print(attention_histogram)
    print("΢�Ź��ں�:AI�����, nb!")
    
    # ��������������� (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)
    
    # ��ӡ������������״
    print("Input shape:", x.shape)
    
    # ǰ�򴫲��������
    output = attention_histogram(x)
    
    # ��ӡ�����������״
    print("Output shape:", output.shape)
