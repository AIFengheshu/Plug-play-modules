import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np

# 论文题目：CrossNorm and SelfNorm for Generalization under Distribution Shifts
# 中文题目：CrossNorm和SelfNorm在分布偏移下的泛化
# 论文链接：https://arxiv.org/pdf/2102.02811
# 官方github：https://github.com/amazon-science/crossnorm-selfnorm
# 所属机构：亚马逊网络服务，罗格斯大学
# 代码整理：微信公众号《AI缝合术》

def calc_ins_mean_std(x, eps=1e-5):
    """extract feature map statistics"""
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = x.size()
    assert (len(size) == 4)
    N, C = size[:2]
    var = x.contiguous().view(N, C, -1).var(dim=2) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = x.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std

def instance_norm_mix(content_feat, style_feat):
    """replace content statistics with style statistics"""
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_ins_mean_std(style_feat)
    content_mean, content_std = calc_ins_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def cn_rand_bbox(size, beta, bbx_thres):
    """sample a bounding box for cropping."""
    W = size[2]
    H = size[3]
    while True:
        ratio = np.random.beta(beta, beta)
        cut_rat = np.sqrt(ratio)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        ratio = float(bbx2 - bbx1) * (bby2 - bby1) / (W * H)
        if ratio > bbx_thres:
            break
    return bbx1, bby1, bbx2, bby2


def cn_op_2ins_space_chan(x, crop='neither', beta=1, bbx_thres=0.1, lam=None, chan=False):
    """2-instance crossnorm with cropping."""
    assert crop in ['neither', 'style', 'content', 'both']
    ins_idxs = torch.randperm(x.size()[0]).to(x.device)
    if crop in ['style', 'both']:
        bbx3, bby3, bbx4, bby4 = cn_rand_bbox(x.size(), beta=beta, bbx_thres=bbx_thres)
        x2 = x[ins_idxs, :, bbx3:bbx4, bby3:bby4]
    else:
        x2 = x[ins_idxs]
    if chan:
        chan_idxs = torch.randperm(x.size()[1]).to(x.device)
        x2 = x2[:, chan_idxs, :, :]
    if crop in ['content', 'both']:
        x_aug = torch.zeros_like(x)
        bbx1, bby1, bbx2, bby2 = cn_rand_bbox(x.size(), beta=beta, bbx_thres=bbx_thres)
        x_aug[:, :, bbx1:bbx2, bby1:bby2] = instance_norm_mix(content_feat=x[:, :, bbx1:bbx2, bby1:bby2],
                                                              style_feat=x2)
        mask = torch.ones_like(x, requires_grad=False)
        mask[:, :, bbx1:bbx2, bby1:bby2] = 0.
        x_aug = x * mask + x_aug
    else:
        x_aug = instance_norm_mix(content_feat=x, style_feat=x2)
    if lam is not None:
        x = x * lam + x_aug * (1-lam)
    else:
        x = x_aug
    return x

class CrossNorm(nn.Module):
    """CrossNorm module"""
    def __init__(self, crop=None, beta=None):
        super(CrossNorm, self).__init__()
        self.active = False
        self.cn_op = functools.partial(cn_op_2ins_space_chan,
                                       crop=crop, beta=beta)

    def forward(self, x):
        if self.training and self.active:
            x = self.cn_op(x)
        self.active = False
        return x


class SelfNorm(nn.Module):
    """SelfNorm module"""
    def __init__(self, chan_num, is_two=False):
        super(SelfNorm, self).__init__()
        # channel-wise fully connected layer
        self.g_fc = nn.Conv1d(chan_num, chan_num, kernel_size=2,
                              bias=False, groups=chan_num)
        self.g_bn = nn.BatchNorm1d(chan_num)
        if is_two is True:
            self.f_fc = nn.Conv1d(chan_num, chan_num, kernel_size=2,
                                  bias=False, groups=chan_num)
            self.f_bn = nn.BatchNorm1d(chan_num)
        else:
            self.f_fc = None

    def forward(self, x):
        b, c, _, _ = x.size()
        mean, std = calc_ins_mean_std(x, eps=1e-12)
        statistics = torch.cat((mean.squeeze(3), std.squeeze(3)), -1)
        g_y = self.g_fc(statistics)
        g_y = self.g_bn(g_y)
        g_y = torch.sigmoid(g_y)
        g_y = g_y.view(b, c, 1, 1)
        if self.f_fc is not None:
            f_y = self.f_fc(statistics)
            f_y = self.f_bn(f_y)
            f_y = torch.sigmoid(f_y)
            f_y = f_y.view(b, c, 1, 1)
            return x * g_y.expand_as(x) + mean.expand_as(x) * (f_y.expand_as(x)-g_y.expand_as(x))
        else:
            return x * g_y.expand_as(x)


if __name__ == '__main__':

    # 生成随机张量，形状为 (batch_size, channels, height, width)
    # 注意batch_size太小无法计算均值和方差，建议2以上
    batch_size, channels, height, width = 2, 32, 256, 256
    x = torch.randn(batch_size, channels, height, width)
    print("input shape:", x.shape)

    # 测试 CrossNorm
    cross_norm = CrossNorm(crop='style', beta=1)
    cross_norm.train()  # 设置为训练模式
    cross_norm.active = True  # 激活 CrossNorm
    x_cn = cross_norm(x)  # 使用 CrossNorm 对输入张量进行处理
    print("CrossNorm output shape:", x_cn.shape)
   
    # 测试 SelfNorm
    self_norm = SelfNorm(chan_num=channels, is_two=True)
    self_norm.train()  # 设置为训练模式
    x_sn = self_norm(x)  # 使用 SelfNorm 对输入张量进行处理
    print("SelfNorm output shape:", x_sn.shape)
