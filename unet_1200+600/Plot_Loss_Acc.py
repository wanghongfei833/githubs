# -*- coding: utf-8 -*-
# @Time    : 2022/8/29 12:04
# @Author  : HongFei Wang
import sys

import matplotlib.pyplot as plt
import numpy as np

file_path = r'./LOG/unet.txt'


def plot_line(*args, **kwargs):
	f = open(file_path, 'r')
	red_lines = f.readlines()
	data = []
	for lin in red_lines[1:]: data.append([float(i) for i in lin.strip('\n').split('\t')])
	title = ['epoch', 'lr', 'train_Loss', 'train_OA', 'train_mIou', 'test_Loss', 'test_OA', 'test_mIOu']
	data = np.array(data)
	plt.title(kwargs['title'])
	if kwargs['bold']:
		plt.xlabel('Epoch', fontsize=kwargs['label_size'], weight='bold')
		plt.ylabel('Value', fontsize=kwargs['label_size'], weight='bold')
		plt.xticks(fontproperties='Times New Roman', size=kwargs['scale_size'], weight='bold')
		plt.yticks(fontproperties='Times New Roman', size=kwargs['scale_size'], weight='bold')
	else:
		plt.xlabel('Epoch', fontsize=kwargs['label_size'])
		plt.ylabel('Value', fontsize=kwargs['label_size'])
		plt.xticks(fontproperties='Times New Roman', size=kwargs['scale_size'])
		plt.yticks(fontproperties='Times New Roman', size=kwargs['scale_size'])
	for index in args:
		if 0 < index < 8:
			plt.plot(data[:, 0], data[:, index], label=title[index])
			plt.legend()
		else:
			print('input must in [1-7] but get %d' % index)
			sys.exit()
	plt.show()


if __name__ == '__main__':
	# 'epoch':0, 'lr':1, 'train_Loss', 'train_OA', 'train_mIou', 'test_Loss', 'test_OA', 'test_mIOu'
	plot_line(5, 2,
	          label_size=15,  # 坐标 大小
	          scale_size=20,  # 刻度值大小
	          bold=False,  # 坐标刻度是否加粗
	          title='Loss'  # 标题
	
	          )