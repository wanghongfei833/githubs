import numpy as np
from PIL import Image

""" 
精度评价代码

二分类的混淆矩阵
P\L     P    N 
P      TP    FP 
N      FN    TN 
"""


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
	print(count)
	confusionMatrix = count.reshape(numClass, numClass)
	return confusionMatrix


def OverallAccuracy(confusionMatrix):
	#  返回所有类的整体像素精度OA
	# acc = (TP + TN) / (TP + TN + FP + TN)
	"""

	:param confusionMatrix:
	:return: OA 值
	"""
	OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
	return OA


def Precision(confusionMatrix):
	#  返回所有类别的精确率precision
	"""

	:param confusionMatrix:
	:return: （num_class，1）所有类的准确率
	"""
	precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
	return precision


def Recall(confusionMatrix):
	#  返回所有类别的召回率recall
	recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
	return recall


def F1Score(confusionMatrix):
	"""
	:param confusionMatrix:
	:return: (num_class,1)返回每个类的 F1
	"""
	precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
	recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
	f1score = 2 * precision * recall / (precision + recall)
	return f1score


def IntersectionOverUnion(confusionMatrix):
	"""

	:param confusionMatrix:
	:return: [numclass,1] iou
	"""
	#  返回交并比IoU
	intersection = np.diag(confusionMatrix)
	union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
	IoU = intersection / union
	return IoU


def MeanIntersectionOverUnion(confusionMatrix):
	"""

	:param confusionMatrix:
	:return: 平均 iou 值
	"""
	#  返回平均交并比mIoU
	intersection = np.diag(confusionMatrix)
	union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
	IoU = intersection / union
	mIoU = np.nanmean(IoU)
	return mIoU


def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
	"""

	:param confusionMatrix:
	:return: 返回 一个值 FWIOU
	"""
	#  返回频权交并比FWIoU
	freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
	iu = np.diag(confusionMatrix) / (
			np.sum(confusionMatrix, axis=1) +
			np.sum(confusionMatrix, axis=0) -
			np.diag(confusionMatrix))
	FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
	return FWIoU


import numba


@numba.njit()
def count(pred):
	w, h, bs = pred.shape
	for b in range(bs):
		for i in range(w):
			for j in range(h):
				if pred[i, j, b] < 70:
					pred[i, j, b] = 0
				elif 140 > pred[i, j, b] >= 70:
					pred[i, j, b] = 1
				else:
					pred[i, j, b] = 2
	return pred


if __name__ == "__main__":
	# 类别数为2，预测图片和标签图片的路径。
	classNum = 3
	pred_img = r"D:\data\comminute\potsdam\top_potsdam_2_10_image.png"
	label_img = r"D:\data\comminute\potsdam\top_potsdam_2_10_labels.png"
	
	# 读取图片并转换成数组
	pred = np.array(Image.open(pred_img))
	label = np.array(Image.open(label_img)) / 255
	label = label.astype(np.int64)
	
	pred = count(pred)
	
	# 展平成一维
	pred = pred.flatten()
	label = label.flatten()
	
	#  计算混淆矩阵及各精度参数
	confusionMatrix = ConfusionMatrix(classNum, pred, label)
	precision = Precision(confusionMatrix)
	recall = Recall(confusionMatrix)
	OA = OverallAccuracy(confusionMatrix)
	IoU = IntersectionOverUnion(confusionMatrix)
	FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
	mIOU = MeanIntersectionOverUnion(confusionMatrix)
	f1ccore = F1Score(confusionMatrix)
	print(f'confusionMatrix:\n{confusionMatrix}\n precision:\n{precision}\nrecall:\n{recall}\nOA:\n{OA}\nIoU:\n{IoU}'
	      f' \nFWIOU:\n{FWIOU}\n mIOU:\n{mIOU}\n f1ccore:\n{f1ccore}')