# -*- coding: utf-8 -*-
# @Time    : 2023/2/18 23:34
# @Author  : HongFei Wang
import os.path

import cv2
import numpy as np
import pywt
import scipy.io as scio
from tqdm import tqdm


def conversion(file, save):
	matdata = scio.loadmat(file)["val"]
	lens = len(matdata[0]) * 2
	wavename = 'cgau8'
	totalscal = lens + 1
	fc = pywt.central_frequency(wavename)
	matdata = np.concatenate((matdata[0], matdata[1]), axis=0)
	cparam = 2 * fc * totalscal
	scales = cparam / np.arange(totalscal, 1, -1)
	[cwtmatr, frequencies] = pywt.cwt(matdata, scales, wavename, 1.0 / lens)
	cwtmatr = np.abs(cwtmatr)
	cwtmatr = np.log(cwtmatr + 0.01)
	cwtmatr = cv2.resize(cwtmatr, (256, 256))  # 将数据重采样至256*256
	np.save(save, cwtmatr)


def mat2img(file: str, save):
	for cls in os.listdir(file):
		matPath = os.path.join(file, cls)
		for mat in tqdm(os.listdir(matPath)):
			name = os.path.join(matPath, mat)  # 拿到mat文件
			savePath = os.path.join(save, mat.replace('mat', 'npy'))
			conversion(name, savePath)  # 拿到数组


def main():
	# print('处理AF数据')
	# os.makedirs('./data/AFIMAGE', exist_ok=True)
	# mat2img(r'./data/AF', './data/AFIMAGE')
	print('处理noAF数据')
	os.makedirs('./data/nonAFIMAGE', exist_ok=True)
	mat2img(r'./data/nonAF', './data/nonAFIMAGE')


if __name__ == '__main__':
	main()