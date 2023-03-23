# -*- coding: utf-8 -*-
# @Time    : 2022/11/20 23:31
# @Author  : HongFei Wang
# -*- coding: utf-8 -*-
# @Time    : 2022/11/20 23:23
# @Author  : HongFei Wang
import numpy as np


def count_nks(self, data: np.array):
	"""
	:param data: 输入数据
	:return: Log(unique),Log(count)
	"""
	assert str(type(data)) == "<class 'numpy.ndarray'>", ValueError('data should be array but get{}'.format(type(data)))
	data = np.round(data, self.decimals)  # 进行取端点
	unique, counts = np.unique(data, return_counts=True)
	counts = counts[::-1]  # 计算频次
	for index, count in enumerate(counts[1:], 1): counts[index] = counts[index - 1] + count  # 计算累计频次
	counts = counts[::-1]
	if np.min(unique) <= 0:
		self.translation = self.translation
		unique = unique - np.min(unique) + self.translation  # 归一化 transaltion+
	unique = np.log(unique) / np.log(self.log_base)
	counts = np.log(counts) / np.log(self.log_base)
	self.log_base = self.log_base
	return unique, counts