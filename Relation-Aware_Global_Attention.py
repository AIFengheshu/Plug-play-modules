import torch
from torch import nn

# 论文题目：Relation-Aware Global Attention for Person Re-identification
# 中文题目：关系感知全局注意力在人员重识别中的应用
# 论文链接：https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Relation-Aware_Global_Attention_for_Person_Re-Identification_CVPR_2020_paper.pdf
# 官方github：https://github.com/WenCongWu/DRANet
# 所属机构：中国科学技术大学，微软亚洲研究院
# 代码整理：微信公众号《AI缝合术》

class RGA_Module(nn.Module):
	def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=True, \
		cha_ratio=8, spa_ratio=8, down_ratio=8):
		super(RGA_Module, self).__init__()

		self.in_channel = in_channel
		self.in_spatial = in_spatial
		
		self.use_spatial = use_spatial
		self.use_channel = use_channel

		print ('Use_Spatial_Att: {};\tUse_Channel_Att: {}.'.format(self.use_spatial, self.use_channel))

		self.inter_channel = in_channel // cha_ratio
		self.inter_spatial = in_spatial // spa_ratio
		
		# Embedding functions for original features
		if self.use_spatial:
			self.gx_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
		if self.use_channel:
			self.gx_channel = nn.Sequential(
				nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)
		
		# Embedding functions for relation features
		if self.use_spatial:
			self.gg_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)
		if self.use_channel:
			self.gg_channel = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel*2, out_channels=self.inter_channel,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
		
		# Networks for learning attention weights
		if self.use_spatial:
			num_channel_s = 1 + self.inter_spatial
			self.W_spatial = nn.Sequential(
				nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(num_channel_s//down_ratio),
				nn.ReLU(),
				nn.Conv2d(in_channels=num_channel_s//down_ratio, out_channels=1,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(1)
			)
		if self.use_channel:	
			num_channel_c = 1 + self.inter_channel
			self.W_channel = nn.Sequential(
				nn.Conv2d(in_channels=num_channel_c, out_channels=num_channel_c//down_ratio,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(num_channel_c//down_ratio),
				nn.ReLU(),
				nn.Conv2d(in_channels=num_channel_c//down_ratio, out_channels=1,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(1)
			)

		# Embedding functions for modeling relations
		if self.use_spatial:
			self.theta_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
								kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
			self.phi_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
							kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
		if self.use_channel:
			self.theta_channel = nn.Sequential(
				nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
								kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)
			self.phi_channel = nn.Sequential(
				nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
							kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)
				
	def forward(self, x):
		b, c, h, w = x.size()
		
		if self.use_spatial:
			# spatial attention
			theta_xs = self.theta_spatial(x)	
			phi_xs = self.phi_spatial(x)
			theta_xs = theta_xs.view(b, self.inter_channel, -1)
			theta_xs = theta_xs.permute(0, 2, 1)
			phi_xs = phi_xs.view(b, self.inter_channel, -1)
			Gs = torch.matmul(theta_xs, phi_xs)
			Gs_in = Gs.permute(0, 2, 1).view(b, h*w, h, w)
			Gs_out = Gs.view(b, h*w, h, w)
			Gs_joint = torch.cat((Gs_in, Gs_out), 1)
			Gs_joint = self.gg_spatial(Gs_joint)
		
			g_xs = self.gx_spatial(x)
			g_xs = torch.mean(g_xs, dim=1, keepdim=True)
			ys = torch.cat((g_xs, Gs_joint), 1)

			W_ys = self.W_spatial(ys)
			if not self.use_channel:
				out = torch.sigmoid(W_ys.expand_as(x)) * x
				return out
			else:
				x = torch.sigmoid(W_ys.expand_as(x)) * x

		if self.use_channel:
			# channel attention
			xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)
			theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)
			phi_xc = self.phi_channel(xc).squeeze(-1)
			Gc = torch.matmul(theta_xc, phi_xc)
			Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)
			Gc_out = Gc.unsqueeze(-1)
			Gc_joint = torch.cat((Gc_in, Gc_out), 1)
			Gc_joint = self.gg_channel(Gc_joint)

			g_xc = self.gx_channel(xc)
			g_xc = torch.mean(g_xc, dim=1, keepdim=True)
			yc = torch.cat((g_xc, Gc_joint), 1)

			W_yc = self.W_channel(yc).transpose(1, 2)
			out = torch.sigmoid(W_yc) * x

			return out

if __name__ == '__main__':
    print("微信公众号:AI缝合术\n")
    # 初始化 RGA 模块
    rga_module = RGA_Module(in_channel=64, in_spatial=32*32) # in_spatial= H * W
    # 创建随机输入张量 (B, C, H, W)
    input_tensor = torch.randn(4, 64, 32, 32)
    # 打印输入形状
    print("Input shape:", input_tensor.shape)
    # 前向传播测试
    output = rga_module(input_tensor)
    # 打印输出形状
    print("Output shape:", output.shape)