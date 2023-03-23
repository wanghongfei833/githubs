# -*- coding: utf-8 -*-
# @Time    : 2022/8/23 10:34
# @Author  : HongFei Wang
import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import xlwt
from osgeo import gdal

from main import hyper_parameter
from main import transform
from model import UNet

mdl_index = 40  # 模型索引
modle = UNet(hyper_parameter['num_class'])

# 预测图像
file_paths = [
		r'E:\taobao\unet_1200\datas\image\10.tif',
		r'E:\taobao\unet_1200\datas\image\11.tif',
		r'E:\taobao\unet_1200\datas\image\12.tif'
]

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'grey', 'pink']


def ConfusionMatrix(numClass, imgPredict, Label):
	"""

	:param numClass:
	:param imgPredict:
	:param Label:
	:return: numclass*numclass的混淆矩阵
	"""
	# 计算混淆矩阵
	mask = (Label >= 0) & (Label < numClass)
	label = numClass * Label[mask] + imgPredict[mask]
	count = np.bincount(label, minlength=numClass ** 2)
	confusionMatrix = count.reshape(numClass, numClass)
	return confusionMatrix


def OverallAccuracy(confusionMatrix):
	#  返回所有类的整体像素精度OA
	# acc = (TP + TN) / (TP + TN + FP + TN)
	"""
		:param confusionMatrix:
		:return: OA 值
		"""
	acc = np.diag(confusionMatrix).sum()
	alls = confusionMatrix.sum()
	OA = acc / alls
	return OA


def write_sheet(sheet, arrays):
	rows, cols = arrays.shape
	for row in range(rows):
		for col in range(cols):
			sheet.write(row, col, int(arrays[row][col]))


def readTif(fileName):
	dataset = gdal.Open(fileName)
	if dataset == None:
		print(fileName + "文件无法打开")
		return
	im_width = dataset.RasterXSize  # 栅格矩阵的列数
	im_height = dataset.RasterYSize  # 栅格矩阵的行数
	im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
	im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
	im_proj = dataset.GetProjection()  # 获取投影信息
	return im_data, im_proj, im_geotrans, im_width, im_height


def writeTiff(im_data, im_width, im_height, im_geotrans, im_proj, path):
	if 'int8' in im_data.dtype.name:
		datatype = gdal.GDT_Byte
	elif 'int16' in im_data.dtype.name:
		datatype = gdal.GDT_UInt16
	else:
		datatype = gdal.GDT_Float32
	im_data = np.array(im_data)
	driver = gdal.GetDriverByName("GTiff")
	dataset = driver.Create(path, im_width, im_height, 1, datatype)
	if dataset is not None:
		dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
		dataset.SetProjection(im_proj)  # 写入投影
	dataset.GetRasterBand(1).WriteArray(im_data)
	del dataset


def main(file_paths, nums_class, save_excle: str, save_path: str):
	"""

	:param file_paths:
	:param nums_class:
	:param save_path: 输出路径
	:return:
	"""
	# 分类数据的tiff
	wook = xlwt.Workbook()
	sheet_index = 0
	label_out = None
	for file_path in tqdm.tqdm(file_paths):
		# 读取数据
		img, im_proj, im_geotrans, im_width, im_height = readTif(file_path)
		maxs_ = np.max(img)
		img_file = img / maxs_  # 归一化
		img = transform(torch.from_numpy(img_file.astype(np.float32)))
		with torch.no_grad():
			pred = modle(img.unsqueeze(0))
			out = torch.argmax(pred, dim=1).squeeze().numpy()
		# 标签读取
		label = gdal.Open(file_path.replace('image', 'label'))
		label = label.ReadAsArray()
		plt.figure(figsize=(16, 9))
		
		# 标签绘制
		maxs_label, mins_label = label.max(), label.min()
		colors_temp_label = colors[mins_label:maxs_label + 1]
		plt.subplot(221)
		plt.title('label')
		plt.imshow(label, cmap=mpl.colors.ListedColormap(colors_temp_label))
		
		# 原图绘制
		plt.subplot(222)
		plt.title('image')
		img_file = np.transpose(img_file[:3, :, :], (1, 2, 0))
		plt.imshow(img_file)
		# 输出绘制
		plt.subplot(223)
		plt.title('out')
		maxs_out, mins_out = out.max(), out.min()
		colors_temp_out = colors[mins_out:maxs_out + 1]
		plt.imshow(out, cmap=mpl.colors.ListedColormap(colors_temp_out))
		writeTiff(out, im_width, im_height, im_geotrans, im_proj,
		          save_path + file_path.split('\\')[-1].replace('.tif', 'out.tif'))
		
		plt.subplot(224)
		plt.title('superposition')
		plt.imshow(img_file)
		plt.imshow(out, alpha=0.2, cmap=mpl.colors.ListedColormap(colors_temp_out))
		if label_out is not None:
			sheet_index += 1
			sheet = wook.add_sheet(str(sheet_index))
			comix = ConfusionMatrix(numClass=nums_class, imgPredict=out, Label=label_out)
			write_sheet(sheet, comix)
		else:
			label_out = out
	wook.save(save_excle)
	plt.show()


if __name__ == '__main__':
	if not os.path.exists(f'./Mdl_Data/unet/{mdl_index}.pth'):
		print(f'File not found  “./Mdl_Data/unet/{mdl_index}.pth”')
	else:
		print(f'load modle {mdl_index}.pth')
		modle.load_state_dict(torch.load(f'./Mdl_Data/unet/{mdl_index}.pth'))
	main(file_paths=file_paths, nums_class=hyper_parameter['num_class'], save_excle='result.xls', save_path='')