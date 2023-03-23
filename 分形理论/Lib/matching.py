# -*- coding: utf-8 -*-
# @Time    : 2022/11/20 23:25
# @Author  : HongFei Wang
import numpy as np


def matching(x, y):
	"""
	最小二乘法拟合
	:param x:
	:param y:
	:return: R2,k,b
	"""
	N = len(x)
	sumx = np.sum(x)
	sumy = np.sum(y)
	sumx2 = np.sum(x ** 2)
	sumxy = np.sum(x * y)
	A = np.array([[N, sumx], [sumx, sumx2]])
	b = np.array([sumy, sumxy])
	b, k = np.linalg.solve(A, b)
	pred = x * k + b
	SSE = np.mean((y - np.mean(y)) ** 2)
	SST = np.mean((y - pred) ** 2)
	R2 = 1 - SST / SSE
	return np.mean(R2), k, b, pred