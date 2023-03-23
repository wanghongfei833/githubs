# -*- coding: utf-8 -*-
# @Time    : 2023/2/4 22:47
# @Author  : HongFei Wang
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class DataSetsSelf(Dataset):
	def __init__(self, root, model, transform):
		super().__init__()
		self.root = root
		self.model = model
		self.transform = transform
		if not os.path.exists("train.txt") \
				or not os.path.exists("test.txt"):
			self.make_txt()
		if self.model == 'train':
			self.img = open('train.txt', 'r')
		else:
			self.img = open('test.txt', 'r')
		self.img = self.img.readlines()
		self.image, self.label = [], []
		for line in self.img:
			line = line.strip("\n").split("\t")
			self.image.append(line[0])
			self.label.append(line[1])
	
	def __getitem__(self, item):
		image = self.image[item]
		lable = self.label[item]
		image = Image.open(image)
		if image.mode != "RGB": image = image.convert('RGB')
		image = self.transform(image)
		lable = torch.tensor(int(lable))
		return image[:3], lable
	
	def __len__(self):
		return len(self.image)
	
	def make_txt(self):
		f1 = open("train.txt", 'w')
		f2 = open("test.txt", "w")
		filename = os.listdir(self.root)  # 获取文件夹
		for index, path in enumerate(filename):
			name_path = os.path.join(self.root, path)
			names_path = os.listdir(name_path)  # 获取每个类多少文件
			cat = int(len(names_path) * 0.8)  # 2/8 分数据
			for pata_file in names_path[:cat]:  # 写入训练集
				f1.write(os.path.join(name_path, pata_file) + '\t' + str(index) + '\n')
			for pata_file in names_path[cat:]:  # 写入测试集
				f2.write(os.path.join(name_path, pata_file) + '\t' + str(index) + '\n')