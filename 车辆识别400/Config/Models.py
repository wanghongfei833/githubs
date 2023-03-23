# -*- coding: utf-8 -*-
# @Time    : 2022/12/15 14:42
# @Author  : HongFei Wang
import torch
from torch import nn


class Block(nn.Module):
	exception = 4
	
	def __init__(self, in_channe, channel, strid=1, downsample=None):
		super(Block, self).__init__()
		self.conv1 = nn.Conv2d(in_channe, channel, kernel_size=1, stride=1, bias=False)
		self.bn1 = nn.BatchNorm2d(channel)
		self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=strid, bias=False)
		self.bn2 = nn.BatchNorm2d(channel)
		self.conv3 = nn.Conv2d(channel, channel * self.exception, kernel_size=1, bias=False, stride=1)
		self.bn3 = nn.BatchNorm2d(channel * self.exception)
		self.re = nn.ReLU(True)
		self.downsample = downsample
	
	def forward(self, x):
		identy = x
		if self.downsample is not None:
			identy = self.downsample(identy)
		
		x = self.re(self.bn1(self.conv1(x)))
		x = self.re(self.bn2(self.conv2(x)))
		x = self.bn3(self.conv3(x))
		return self.re(x + identy)


class RestNet(nn.Module):
	def __init__(self, num_stages):
		super(RestNet, self).__init__()
		self.in_place = 64
		self.conv1 = nn.Conv2d(3, self.in_place, kernel_size=7, padding=3, bias=False, stride=2)
		self.bn1 = nn.BatchNorm2d(self.in_place)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(3, 2, 1)
		self.layer1 = self.make_layer(Block, 64, num_stage=num_stages[0])
		self.layer2 = self.make_layer(Block, 128, num_stage=num_stages[1], strid=2)
		self.layer3 = self.make_layer(Block, 256, num_stage=num_stages[2], strid=2)
		self.layer4 = self.make_layer(Block, 512, num_stage=num_stages[3], strid=2)
		self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
		self.head = nn.Linear(512 * 4, 7 * 34)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avg_pool(x)
		x = self.head(x)
		x = x.view(-1, 34, 7)
		return x
	
	def make_layer(self, block, channel, num_stage, strid=1):
		downsample = None
		if strid != 1 or self.in_place != channel * block.exception:
			downsample = nn.Sequential(
					nn.Conv2d(self.in_place, channel * block.exception, 1, strid, bias=False),
					nn.BatchNorm2d(channel * block.exception)
			)
		layer = [block(self.in_place, channel, strid, downsample)]
		self.in_place = channel * block.exception
		for _ in range(1, num_stage):
			layer.append(block(self.in_place, channel))  # 后续stage不进行跳连接故不需要strid 以及downsample参数
		return nn.Sequential(*layer)


def get_model(Read):
	file_path = r'C:\Users\10095\.cache\torch\hub\checkpoints\resnet50-0676ba61.pth'
	model = RestNet(num_stages=[3, 4, 6, 3])  #
	if Read:
		model.load_state_dict(torch.load(file_path), strict=False)
		print("load paths")
	return model