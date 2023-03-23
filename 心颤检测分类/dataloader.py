# -*- coding: utf-8 -*-
# @Time    : 2023/2/16 8:57
# @Author  : HongFei Wang
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class Datasets(Dataset):
	"""
	数据文件...
	"""
	
	def __init__(self, root: str, model: str, transform=None):
		super(Datasets, self).__init__()
		if root is None:
			print("输入文件夹目录")
			sys.exit()
		if model not in ['train', 'test']: raise print('only "train" or "test" ,but get %s' % model)
		self.root = root
		self.dir_root = os.path.dirname(__file__)
		self.to_tensor = transforms.ToTensor()
		if not os.path.isdir(self.root): raise print('input right filepath ,not %s' % self.root)
		self.modle = model
		self.image, self.lables = [], []
		for index, name in enumerate(os.listdir(root)):
			namePath = os.path.join(root, name)
			name = os.listdir(namePath)
			if model == 'train':
				name = name[:int(len(name) * 0.8)]
			else:
				name = name[int(len(name) * 0.8):]
			self.image.extend([os.path.join(namePath, i) for i in name])
			self.lables.extend([index for _ in name])
		self.transform = transform
	
	def __getitem__(self, idx):
		image, labels = self.image[idx], int(self.lables[idx])
		image = np.load(image).astype(np.float32)  # 读取二进制文件
		if self.transform: image = self.transform(image)
		labers = torch.tensor(labels)
		return image, labers
	
	def __len__(self):
		return len(self.image)
	
	def read(self):
		# 判断文件是否存在。不存在就创建
		names = self.root.replace('\\', '/').split('/')[-1]
		f1 = open(f'./Data_/{names}_train.txt', "w")
		f2 = open(f'./Data_/{names}_test.txt', "w")
		for index, paths in enumerate(sorted(os.listdir(self.root))):
			paths_ = os.path.join(self.root, paths)
			dirs = os.listdir(paths_)
			random.shuffle(dirs)
			cat = int(len(dirs) * 0.8)
			for pata_file in dirs[:cat]:  # 写入训练集
				try:
					f1.write(os.path.join(paths_, pata_file) + '\t' + str(index) + '\n')
				except:
					print(os.path.join(paths_, pata_file) + '\t' + str(index))
			for pata_file in dirs[cat:]:  # 写入测试集
				try:
					f2.write(os.path.join(paths_, pata_file) + '\t' + str(index) + '\n')
				except:
					print(os.path.join(paths_, pata_file), '写入失败')
		f1.close()
		f2.close()
		print('数据集创建成功，请重新运行!!!')
		sys.exit()