# -*- coding: utf-8 -*-
# @Time    : 2022/8/23 9:05
# @Author  : HongFei Wang
from torch import nn


# SE 模块
class Attition(nn.Module):
	def __init__(self, channel, dit):
		super(Attition, self).__init__()
		self.pool = nn.AdaptiveAvgPool2d(1)
		# 64 128 256
		self.fc1 = nn.Linear(channel, channel // dit)
		self.fc2 = nn.Linear(channel // dit, channel)
		self.sigmod = nn.Sigmoid()  # 压缩 0-1
	
	def forward(self, x):
		# 池化： B C W H --> B C 1 1
		# B C 1 1 --> B 1 1 C
		x = self.pool(x).transpose(1, 3)
		x = self.fc2(self.fc1(x))
		# B 1 1 C-->B 1 1 C1 -->B 1 1 C
		x = self.sigmod(x).transpose(1, 3)  # sigmod 可删
		return x


class DownsampleLayer(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(DownsampleLayer, self).__init__()
		self.Conv_BN_ReLU_2 = nn.Sequential(
				nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(out_ch),
				nn.ReLU(),
				nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(out_ch),
				nn.ReLU()
		)
		self.downsample = nn.Sequential(
				nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
				nn.BatchNorm2d(out_ch),
				nn.ReLU()
		)
		self.att = Attition(out_ch, 8)  # 8 可以修改
	
	def forward(self, x):
		"""
		:param x:
		:return: out输出到深层，out_2输入到下一层，
		"""
		out = self.Conv_BN_ReLU_2(x)
		# 获取注意力
		out_ = self.att(out)  # out_  --> (0,1)
		# 注意力机制融合
		out = out_ * out
		out_2 = self.downsample(out)
		return out, out_2