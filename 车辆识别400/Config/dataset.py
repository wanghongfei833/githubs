# -*- coding: utf-8 -*-
# @Time    : 2022/12/15 13:16
# @Author  : HongFei Wang
import os
import sys

import numpy as np
import torch
from PIL import Image as IMAGE
from torch.utils.data import Dataset


class DataSetMySelf(Dataset):
	def __init__(self, dirs="./data", train=True, trainsform=None):
		super().__init__()
		self.root = "./data"
		self.image_path = []
		self.label_path = []
		self.transform = trainsform
		make_txt = not os.path.exists("train.txt") or not os.path.exists("val.txt")
		if make_txt:
			self.make_txt()
			self.make_txt("val")
			sys.exit()
		if train:
			self.read_path("train.txt")
		else:
			self.read_path("val.txt")
	
	def __len__(self):
		return len(self.image_path)
	
	def __getitem__(self, item):
		image = self.image_path[item]
		label = self.label_path[item]
		label = torch.from_numpy(np.array([int(label[i * 2:i * 2 + 2]) for i in range(7)]))
		
		image = IMAGE.open(image)
		image = self.transform(image)
		return image, label.long()
	
	def read_path(self, txt_name):
		paths = open(txt_name, "r", encoding="utf-8")
		
		for line in paths.readlines():
			line = line.strip()
			line = line.split("\t")
			self.image_path.append(line[0])
			self.label_path.append(line[1])
	
	def make_txt(self, model="train", root="data"):
		file_path = os.path.join(os.getcwd().replace("\Config", ""), root)
		list_dir1 = os.listdir(os.path.join(file_path, model))  # 获取路径
		ch_1 = ["京", "津", "冀", "晋", "蒙", "辽", "吉", "黑", "沪", "苏", "浙", "皖", "闽",
		        "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "渝", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
		ch_2 = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T",
		        "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
		print(len(ch_2))
		f1 = open(model + ".txt", "w", encoding="utf-8")
		for name in list_dir1:
			temp = ""
			names = name.split(".")[0]
			for d in names:
				if d in ch_1:
					temp += "%2d" % ch_1.index(d)
				elif d in ch_2:
					temp += "%2d" % ch_2.index(d)
			f1.write(os.path.join(file_path, model) + "\\" + name + "\t" + temp.replace(" ", "0") + "\n")
		f1.close()