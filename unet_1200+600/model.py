# -*- coding: utf-8 -*-
# @Time    : 2022/8/23 9:39
# @Author  : HongFei Wang
import torch

from decoder import *
from encoder import *


class UNet(nn.Module):
	def __init__(self, num_class):
		super(UNet, self).__init__()
		out_channels = [2 ** (i + 6) for i in range(5)]  # [64, 128, 256, 512, 1024]
		# 下采样
		# 后续 改数据
		self.d1 = DownsampleLayer(4, out_channels[0])  # 3-64
		self.d2 = DownsampleLayer(out_channels[0], out_channels[1])  # 64-128
		self.d3 = DownsampleLayer(out_channels[1], out_channels[2])  # 128-256
		self.d4 = DownsampleLayer(out_channels[2], out_channels[3])  # 256-512
		# 上采样
		self.u1 = UpSampleLayer(out_channels[3], out_channels[3])  # 512-1024-512
		self.u2 = UpSampleLayer(out_channels[4], out_channels[2])  # 1024-512-256
		self.u3 = UpSampleLayer(out_channels[3], out_channels[1])  # 512-256-128
		self.u4 = UpSampleLayer(out_channels[2], out_channels[0])  # 256-128-64
		# 输出
		self.o = nn.Sequential(
				nn.Conv2d(out_channels[1], out_channels[0], kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(out_channels[0]),
				nn.ReLU(),
				nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(out_channels[0]),
				nn.ReLU(),
				nn.Conv2d(out_channels[0], num_class, 3, 1, 1),
		)
	
	def forward(self, x):
		out_1, out1 = self.d1(x)
		out_2, out2 = self.d2(out1)
		out_3, out3 = self.d3(out2)
		out_4, out4 = self.d4(out3)
		out5 = self.u1(out4, out_4)
		out6 = self.u2(out5, out_3)
		out7 = self.u3(out6, out_2)
		out8 = self.u4(out7, out_1)
		out = self.o(out8)
		return out


# W B  (0-1)

class BasicseBlock(nn.Module):
	"""
		基础卷积结构 进行 Conv->bn->relu->Conv->bn->relu
	"""
	
	def __init__(self, in_channels, middle_channels, out_channels):
		super().__init__()
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(middle_channels)
		self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(out_channels)
	
	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		
		return out


class UNetadd(nn.Module):
	def __init__(self, num_classes, input_channels=3, deep_supervision=False):
		"""

		:param num_classes: 分类数量
		:param input_channels: 输入通道
		:param deep_supervision: 管理深度 把每个输出出来 默认 False
		"""
		super().__init__()
		
		nb_filter = [32, 64, 128, 256, 512]  # 2^n
		
		self.deep_supervision = deep_supervision
		
		self.pool = nn.MaxPool2d(2, 2)
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		
		self.conv0_0 = BasicseBlock(input_channels, nb_filter[0], nb_filter[0])
		self.conv1_0 = BasicseBlock(nb_filter[0], nb_filter[1], nb_filter[1])
		self.conv2_0 = BasicseBlock(nb_filter[1], nb_filter[2], nb_filter[2])
		self.conv3_0 = BasicseBlock(nb_filter[2], nb_filter[3], nb_filter[3])
		self.conv4_0 = BasicseBlock(nb_filter[3], nb_filter[4], nb_filter[4])
		
		self.conv0_1 = BasicseBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
		self.conv1_1 = BasicseBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
		self.conv2_1 = BasicseBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
		self.conv3_1 = BasicseBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
		
		self.conv0_2 = BasicseBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
		self.conv1_2 = BasicseBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
		self.conv2_2 = BasicseBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])
		
		self.conv0_3 = BasicseBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
		self.conv1_3 = BasicseBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])
		
		self.conv0_4 = BasicseBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])
		
		if self.deep_supervision:
			self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
			self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
			self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
			self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
		else:
			self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
	
	def forward(self, input):
		x0_0 = self.conv0_0(input)  # 1024
		x1_0 = self.conv1_0(self.pool(x0_0))  # 512
		x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))  # 1024
		
		x2_0 = self.conv2_0(self.pool(x1_0))  # 256
		x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))  # 512
		x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))  # 1024
		
		x3_0 = self.conv3_0(self.pool(x2_0))  # 128
		x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))  # 256
		x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))  # 512
		x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))  # 1024
		
		x4_0 = self.conv4_0(self.pool(x3_0))  # 64
		x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))  # 128
		x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))  # 256
		x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))  # 512
		x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))  # 1024
		
		if self.deep_supervision:
			output1 = self.final1(x0_1)
			output2 = self.final2(x0_2)
			output3 = self.final3(x0_3)
			output4 = self.final4(x0_4)
			return [output1, output2, output3, output4]
		
		else:
			output = self.final(x0_4)
			return output