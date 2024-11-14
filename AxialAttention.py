import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ������Ŀ��Medical Transformer: Gated Axial-Attention forMedical Image Segmentation
# ������Ŀ:  ҽ��Transformer������ҽѧͼ��ָ���ſ�����ע��������
# �������ӣ�https://arxiv.org/pdf/2102.10662
# �ٷ�github��https://github.com/jeya-maria-jose/Medical-Transformer
# ����������Լ�����ս�˹��ѧ, ������������ѧ
# �ؼ��ʣ� Transformer, ҽѧͼ��ָ�, ��ע��������

# ����1x1��������ڸı�ͨ����
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 ���"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# ����һ��һά����㣬���ڽ���qkv�任
class qkv_transform(nn.Conv1d):
    """����qkv�任��Conv1d"""

# ��������ע����ģ�飨Axial Attention��
class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        # ȷ������ͨ���������ͨ�������ܱ���������
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes  # ����ͨ����
        self.out_planes = out_planes  # ���ͨ����
        self.groups = groups  # ע����ͷ������
        self.group_planes = out_planes // groups  # ÿ��ͨ����
        self.kernel_size = kernel_size  # ����˴�С
        self.stride = stride  # ����
        self.bias = bias  # �Ƿ�ʹ��ƫ��
        self.width = width  # �Ƿ�Կ�Ƚ�������ע����

        # ��ͷ��ע������qkv�任��
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        # ��һ����
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # λ�ñ���
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        # ���λ������
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))

        # �����������1������ӳػ���
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        # ��ʼ������
        self.reset_parameters()

    def forward(self, x):
        # ���ݿ�Ȼ�߶ȵ�������������ά��˳��
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H

        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # ����qkv�任
        qkv = self.bn_qkv(self.qkv_transform(x))
        # ��qkv�ֽ�Ϊq��k��v
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # ����λ�ñ���
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)

        # ����q��λ�ñ���ĳ˻�
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        # ��������Ծ��󲢹�һ��
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked_similarity, dim=3)

        # ��Ȩ��͵õ�v��ע�������
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        # �ָ�������ά��˳��
        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        # �����������1������гػ�
        if self.stride > 1:
            output = self.pooling(output)

        return output

    # ������ʼ��
    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

# ���Դ���
if __name__ == '__main__':
    input = torch.randn(1, 64, 224, 224)  # ����һ���������룬��СΪ 1x64x224x224
    block = AxialAttention(in_planes=64, out_planes=64, groups=1, kernel_size=224, stride=1, bias=False, width=False)
    # in_planes��out_planes��channelһ�£�kernel_size��h,wһ��
    output = block(input)
    print(input.size())  # ������������ĳߴ�
    print(output.size())  # �����������ĳߴ�
