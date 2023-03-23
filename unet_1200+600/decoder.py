# -*- coding: utf-8 -*-
# @Time    : 2022/8/23 9:38
# @Author  : HongFei Wang
import torch
from torch import nn


class UpSampleLayer(nn.Module):
	def __init__(self, in_ch, out_ch):
		# 512-1024-512
		# 1024-512-256
		# 512-256-128
		# 256-128-64
		super(UpSampleLayer, self).__init__()
		self.Conv_BN_ReLU_2 = nn.Sequential(
				nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 2, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(out_ch * 2),
				nn.ReLU(),
				nn.Conv2d(in_channels=out_ch * 2, out_channels=out_ch * 2, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(out_ch * 2),
				nn.ReLU()
		)
		self.upsample = nn.Sequential(
				nn.ConvTranspose2d(in_channels=out_ch * 2, out_channels=out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
				nn.BatchNorm2d(out_ch),
				nn.ReLU()
		)
	
	def forward(self, x, out):
		'''
		:param x: 输入卷积层
		:param out:与上采样层进行cat
		:return:
		'''
		x_out = self.Conv_BN_ReLU_2(x)
		x_out = self.upsample(x_out)
		# B C W H --> C 拼接 --> B c1+c2 W H
		cat_out = torch.cat((x_out, out), dim=1)
		return cat_out