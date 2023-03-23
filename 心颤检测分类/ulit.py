# -*- coding: utf-8 -*-
# @Time    : 2023/2/15 8:51
# @Author  : HongFei Wang

import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import Datasets

device = torch.device('cuda:0')


def train(tran_data, epoch, criterion, scheduler, net, optimizer, best_data):
	pars_train = tqdm(tran_data, total=len(tran_data), file=sys.stdout)
	running_loss = 0.
	class_correct, class_total = 0, 0
	for index, data in enumerate(pars_train, 1):
		inputs, labels = data[0].to(device), data[1].to(device)
		y_pred = net(inputs)
		prediction = torch.argmax(y_pred, 1)
		reall = (prediction == labels).sum().cpu()
		class_correct += reall
		class_total += labels.size(0)
		optimizer.zero_grad(set_to_none=True)
		loss = criterion(y_pred, labels)
		loss.backward()
		optimizer.step()
		# 计算统计参数
		running_loss += loss.item()
		lr = "{:.3g}".format(scheduler.get_last_lr()[0])
		pars_train.set_description(">>| train-%.3d | Lr:%s | Loss:%.7f | Acc:%.2f%% |best:%.2f%%|" % (
				epoch + 1, lr, running_loss / index, class_correct * 100 / class_total, best_data))
		pars_train.update(1)
	pars_train.close()
	return running_loss / class_total, class_correct * 100 / class_total


def Test(test_data, epoch, scheduler, net, best_data, count=False, **kwargs):
	pars_train = tqdm(test_data, total=len(test_data), colour="blue")
	running_loss = 0.
	class_correct, class_total = 0, 0
	criterion = torch.nn.CrossEntropyLoss()
	if count: datares = np.zeros((len(kwargs['label']), len(kwargs['label'])))
	
	with torch.no_grad():
		for index, data in enumerate(pars_train, 1):
			inputs, labels = data[0].to(device), data[1].to(device)
			y_pred = net(inputs)
			prediction = torch.argmax(y_pred, -1)
			reall = (prediction == labels).sum().cpu()
			class_correct += reall
			classes = labels.size(0)
			class_total += classes
			loss = criterion(y_pred, labels)
			running_loss += loss.item()
			lr = "{:.3g}".format(scheduler.get_last_lr()[0])
			if kwargs['count']:
				datares += ConfusionMatrix(len(labels), y_pred, labels)
			pars_train.set_description(">>|  test-%.3d | Lr:%s | Loss:%.7f | Acc:%.2f%% |best:%.2f%%|" % (
					epoch + 1, lr, running_loss / index, class_correct * 100 / class_total, best_data))
			pars_train.update(1)
	if count:
		precision = Precision(datares)
		recall = Recall(datares)
		OA = OverallAccuracy(datares)
		IoU = IntersectionOverUnion(datares)
		f1ccore = F1Score(datares)
		print(f'\nprecision:{precision}\n'
		      f'recall:{recall}\n'
		      f'OA:{OA}\n'
		      f'IOU:{IoU}\n'
		      f'F1:{f1ccore}\n')
	return running_loss / class_total, class_correct * 100 / class_total


def OverallAccuracy(confusionMatrix):
	#  返回所有类的整体像素精度OA
	# acc = (TP + TN) / (TP + TN + FP + TN)
	"""

    :param confusionMatrix:
    :return: OA 值
    """
	OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
	return OA


def ConfusionMatrix(numClass, imgPredict, Label):
	"""

    :param numClass:
    :param imgPredict:
    :param Label:
    :return: numclass*numclass的混淆矩阵
    """
	# 计算混淆矩阵
	imgPredict = torch.argmax(imgPredict, dim=1).cpu().numpy()
	Label = Label.cpu().numpy()
	
	mask = (Label >= 0) & (Label < numClass)
	label = numClass * Label[mask] + imgPredict[mask]
	count = np.bincount(label, minlength=numClass ** 2)
	confusionMatrix = count.reshape(numClass, numClass)
	return confusionMatrix


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


def drawConfusionMatrix(net, model_path, datas, classes: list, sava_path):
	"""

    :param net:  网络
    :param model_path: 网络模型
    :param datas: 测试数据
    :param classes: 种类标签
    :param sava_path: 保存路径 npy
    :return:
    """
	# 首先定义一个 分类数*分类数 的空混淆矩阵
	conf_matrix = torch.zeros((len(classes), len(classes)))
	net.load_state_dict(torch.load(model_path))
	with torch.no_grad():
		for step, (imgs, targets) in enumerate(datas):
			targets = targets.to(device)
			imgs = imgs.to(device)
			out = net(imgs)
			# 记录混淆矩阵参数
			conf_matrix += ConfusionMatrix(len(classes), out, targets)
	plt.imshow(conf_matrix, cmap=plt.cm.Blues)
	corrects = conf_matrix.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
	np.save(sava_path, conf_matrix)
	per_kinds = np.sum(conf_matrix, axis=1)  # 抽取每个分类数据总的测试条数
	
	print("混淆矩阵总元素个数：{0}".format(int(np.sum(conf_matrix))))
	print(conf_matrix)
	
	# 获取每种Emotion的识别准确率
	print("每个种类的个数：", per_kinds)
	print("每种预测正确的个数：", corrects)
	print("每种的识别准确率为：{0}".format([rate * 100 for rate in corrects / per_kinds]))
	print('len:', len(classes))
	thresh = conf_matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
	for x in range(len(classes)):
		for y in range(len(classes)):
			# 注意这里的matrix[y, x]不是matrix[x, y]
			info = int(conf_matrix[y, x])
			plt.text(x, y, info,
			         verticalalignment='center',
			         horizontalalignment='center',
			         color="white" if info > thresh else "black")
	
	# plt.tight_layout()  # 保证图不重叠
	plt.subplots_adjust(left=0.25, bottom=0.3)
	plt.yticks(range(len(classes)), classes)
	plt.xticks(range(len(classes)), classes, rotation=90)  # X轴字体倾斜45°
	plt.show()


def mean_std(train_data):
	pop_mean = []
	pop_std0 = []
	for data in tqdm(train_data, total=len(train_data)):
		numpy_image = data[0].numpy()  # B C W H
		batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
		batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
		pop_mean.append(batch_mean)
		pop_std0.append(batch_std0)
	pop_mean = f"{np.array(pop_mean).mean(axis=0)}".replace(' ', ',')
	pop_std0 = f'{np.array(pop_std0).mean(axis=0)}'.replace(' ', ',')
	print(f" mean, std =torch.tensor({pop_mean}),torch.tensor({pop_std0})")
	sys.exit()


def count_param(model):
	param_count = 0
	for param in model.parameters():
		param_count += param.view(-1).size()[0]
	return param_count


def getDataloader(arg, transform):
	trainDataset = Datasets(arg.data_path, model='train', transform=transform['train'])
	testDataset = Datasets(arg.data_path, model='test', transform=transform['test'])
	
	trainLoader = DataLoader(trainDataset, arg.batch_size, shuffle=True, num_workers=arg.num_classes)
	testLoader = DataLoader(testDataset, arg.batch_size, shuffle=True, num_workers=arg.num_classes)
	
	return trainLoader, testLoader