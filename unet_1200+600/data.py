# -*- coding: utf-8 -*-
# @Time    : 2022/8/23 9:45
# @Author  : HongFei Wang
import os
import sys

import numpy as np
import torch
from osgeo import gdal
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class Datasets(Dataset):
	def __init__(self, train: bool, root: str, tfs: transforms, numclass: int):
		super(Datasets, self).__init__()
		self.root = root
		self.model = train
		self.cat = 0.8  # 80 % 作为训练集 20%作为测试集
		self.tfs = tfs
		self.num_class = numclass
		self.transform = transforms.Compose([
				transforms.ToTensor(),
		])
		if os.path.exists('train.txt') and os.path.exists('test.txt'):  # 判断 两个 txt 是否存在
			if self.model:
				self.img = open('train.txt', 'r').read()
				self.label = self.img.replace('image', 'label')  # 俩个文件之间的差异
				self.img = self.img.split('\n')
				self.label = self.label.split('\n')
			else:
				self.img = open('test.txt', 'r').read()
				self.label = self.img.replace('image', 'label')  # 俩个文件之间的差异
				self.img = self.img.split('\n')
				self.label = self.label.split('\n')
		else:
			self.make_txt()
	
	def __getitem__(self, item):
		
		img = os.path.join(os.path.join(self.root, 'image'), self.img[item])
		label = os.path.join(os.path.join(self.root, 'label'), self.label[item])
		img = gdal.Open(img)
		label = gdal.Open(label)
		img = img.ReadAsArray()
		maxs = np.max(img)
		img = img / maxs
		label = label.ReadAsArray()
		#  array--> tensor-->transpose(2,0 1)    array--> (256,191,3)---->tf.totensor-->(3,253,191)
		img = self.tfs(torch.from_numpy(img.astype(np.float32)))
		label = torch.from_numpy(label)
		return img, label.long()
	
	def __len__(self):
		return len(self.img)
	
	def make_txt(self):
		image_path = os.listdir(os.path.join(self.root, 'image'))  # 获取标签中有多少文件 返回列表
		cat = int(self.cat * len(image_path))
		train_img, test_path = image_path[:cat], image_path[cat:]
		f1 = open('train.txt', 'w')
		f2 = open('test.txt', 'w')
		f1.write('\n'.join(train_img))
		f1.close()
		f2.write('\n'.join(test_path))
		f2.close()
		print('数据集创建成功，重新运行')
		sys.exit()