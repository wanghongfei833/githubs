# -*- coding: utf-8 -*-
# @Time    : 2023/2/5 16:18
# @Author  : HongFei Wang
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
	def __init__(self, numclass):
		super(Net, self).__init__()
		
		self.conv1 = nn.Conv2d(3, 6, 5)
		
		self.conv2 = nn.Conv2d(6, 16, 5)
		
		# 最后的三层全连接
		self.fc1 = nn.Linear(44944, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, numclass)
	
	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		
		x = x.view(-1, self.num_flot_features(x))
		
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
	
	def num_flot_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		
		return num_features


def LetNet(numclass):
	return Net(numclass)