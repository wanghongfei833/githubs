# -*- coding: utf-8 -*-
# @Time    : 2023/2/5 16:09
# @Author  : HongFei Wang
import os

import matplotlib.pyplot as plt

paths = "./image"

classes = os.listdir(paths)

for clas in classes:
	name = os.path.join(paths, clas)  # 每个类路径
	for imgname in os.listdir(name):
		plt.clf()
		img = os.path.join(name, imgname)
		img = plt.imread(img)
		w, h, c = img.shape
		
		img = img[:-int(w * 0.3), int(h * 0.2):-int(h * 0.2), :]
		# plt.imshow(img)
		# plt.pause(0.5)
		plt.imsave(f'./returns/{imgname}', img)