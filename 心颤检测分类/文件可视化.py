# -*- coding: utf-8 -*-
# @Time    : 2023/2/14 10:20
# @Author  : HongFei Wang
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio


def showDataset(imgPath):
	datas = scio.loadmat(imgPath)
	data = datas['val']
	x = np.arange(len(data[0]))
	plt.subplot(311)
	plt.plot(x, data[0])
	plt.plot(x, data[1])
	plt.subplot(312)
	plt.plot(x, data[0])
	plt.subplot(313)
	plt.plot(x, data[1])
	plt.show()


if __name__ == '__main__':
	showDataset('./data/AF/3000701m_/3000701m_019.mat')