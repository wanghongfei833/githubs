# -*- coding: utf-8 -*-
# @Time    : 2022/11/21 10:05
# @Author  : HongFei Wang
import os

import matplotlib.pyplot as plt
from osgeo import gdal


def read_image(file_path):
	assert os.path.exists(file_path), ValueError('{} is not found'.format(file_path))
	if os.path.splitext(file_path)[-1] in ['jpg', 'png']:  # 普通图片
		image = plt.imread(file_path)
	
	elif '.' not in file_path or os.path.splitext(file_path)[-1] in ['.tif', '.dat']:
		image = gdal.Open(file_path)
		if image is None: print('文件打开失败!')
		image = image.ReadAsArray()
		if len(image.shape) == 3: image = np.transpose(image, (1, 2, 0))
	else:
		print('文件打开失败!')
		image = None
	return image