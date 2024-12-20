import torch
from torch import Tensor,nn

# ������Ŀ��OrthoNets : Orthogonal Channel Attention Networks
# ������Ŀ��OrthoNets : ����ͨ��ע��������
# �������ӣ�https://arxiv.org/pdf/2311.03071
# �ٷ�github��https://github.com/hady1011/OrthoNets
# ��������������ɫ��ѧ�������ѧ����������ϵ
# ��������΢�Ź��ںš�AI�������
# ע�����´�����Դ���Ż�����õ�����Դ���߼������������и���������ʵ�����������ܣ�Դ���Ϊɢ�ң���ϸ���Դ��

def gram_schmidt(input):
    def projection(u, v):
        return (torch.dot(u.view(-1), v.view(-1)) / torch.dot(u.view(-1), u.view(-1))) * u
    output = []
    for x in input:
        for y in output:
            x = x - projection(y, x)
        x = x / x.norm(p=2)
        output.append(x)
    return torch.stack(output)

def initialize_orthogonal_filters(c, h, w):
    if h * w < c:
        n = c // (h * w)
        gram = []
        for i in range(n):
            gram.append(gram_schmidt(torch.rand([h * w, 1, h, w])))
        return torch.cat(gram, dim=0)
    else:
        return gram_schmidt(torch.rand([c, 1, h, w]))

class GramSchmidtTransform(torch.nn.Module):
    instance = {}  # ��ʼ���ֵ�
    constant_filter: Tensor

    @staticmethod
    def build(c: int, h: int):
        if (c, h) not in GramSchmidtTransform.instance:
            GramSchmidtTransform.instance[(c, h)] = GramSchmidtTransform(c, h)
        return GramSchmidtTransform.instance[(c, h)]

    def __init__(self, c: int, h: int):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            rand_ortho_filters = initialize_orthogonal_filters(c, h, h).view(c, h, h)
        self.register_buffer("constant_filter", rand_ortho_filters.to(self.device).detach())

    def forward(self, x):
        _, _, h, w = x.shape
        _, H, W = self.constant_filter.shape
        if h != H or w != W: 
            x = torch.nn.functional.adaptive_avg_pool2d(x, (H, W))
        return (self.constant_filter * x).sum(dim=(-1, -2), keepdim=True)

# class Orthogonal_Channel_Attention(torch.nn.Module):
#     def __init__(self, c: int, h:int):
#         super().__init__()
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.FWT = GramSchmidtTransform.build(c, h)  # ��ʼ�� FWT

#     def forward(self, input: Tensor):
#         x = input
#         while input[0].size(-1) > 1:
#             input = self.FWT(input.to(self.device))
#         b = input.size(0)
#         return input.view(b, -1)
    
class Orthogonal_Channel_Attention(nn.Module):
    def __init__(self, channels: int, height: int):
        """
        ��ʼ��Orthogonal_Channel_Attentionģ��
        :param channels: ����������ͨ���� (C)
        :param height: Gram-Schmidt �任����ĸ߶� (���� H=W)
        """
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.channels = channels
        self.height = height

        # Gram-Schmidt �任��ʼ��
        self.F_C_A = GramSchmidtTransform.build(channels, height)
        
        # ͨ��ע����ӳ�䣨SE Block �ṹ��
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        ǰ�򴫲�
        :param x: �������� (B, C, H, W)
        :return: ������� (B, C, H, W)
        """
        B, C, H, W = x.shape
        # �������� H �� W ���ʼ����ƥ�䣬������Ӧ�ػ�
        if H != self.height or W != self.height:
            x = nn.functional.adaptive_avg_pool2d(x, (self.height, self.height))
        # Gram-Schmidt �任
        transformed = self.F_C_A(x)  # (B, C, 1, 1)
        # ȥ���ռ�ά�ȣ�����ͨ��ע��������
        compressed = transformed.view(B, C)
        # ͨ��ע��������
        excitation = self.channel_attention(compressed).view(B, C, 1, 1)
        # ��Ȩԭʼ��������
        output = x * excitation
        return output        

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ��ʼ�� Orthogonal_Channel_Attention
    channels = 32
    height = 256
    attention_module = Orthogonal_Channel_Attention(channels, height).to(device)
    # �������� (B, C, H, W)
    input_tensor = torch.rand(1, channels, 256, 256).to(device)
    # ǰ�򴫲�
    output_tensor = attention_module(input_tensor)
    print(f"����������״: {input_tensor.shape}")
    print(f"���������״: {output_tensor.shape}")

