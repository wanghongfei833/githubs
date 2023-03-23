# -*- coding: utf-8 -*-
# @Time    : 2022/8/23 9:57
# @Author  : HongFei Wang
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

np.seterr(divide='ignore', invalid='ignore')


class Units(object):
	def __init__(self, num_class: int, net: torch.nn.Module, net_name, EPOCH, train_data, test_data,
	             optimer: str, scheduler: str, best=0., mdl_path='', **kwargs):
		"""

		:param net: 网络
		:param net_name: 网络名称
		:param EPOCH: 迭代次数
		:param train_data: 训练数据
		:param test_data: 测试数据
		:param optimer: 优化器[SGD,ADAMW]
		:param scheduler: [COS,EXP,MUL]
		:param Log: 输出路径
		:param best: 最有精度
		:param mdl_path: 读取的 path 的索引
		:param kwargs: lr weight_decay momentum gamma eta_min milestones
		"""
		super(Units, self).__init__()
		self.num_class = num_class
		self.net_name = net_name
		self.criterion = torch.nn.CrossEntropyLoss()
		self.device = torch.device('cuda:0')
		self.net = net
		self.train_data = train_data
		self.test_data = test_data
		self.best = best
		self.net.to(self.device)
		if not os.path.exists('./LOG'): os.makedirs('./LOG')
		if not os.path.exists(f'./Mdl_Data/{self.net_name}'): os.makedirs(f'./Mdl_Data/{self.net_name}')
		if os.path.exists(f'./Mdl_Data/{self.net_name}/{mdl_path}.pth'):
			print(f'load modle ./Mdl_Data/{self.net_name}/{mdl_path}.pth')
			self.net.load_state_dict(torch.load(f'./Mdl_Data/{self.net_name}/{mdl_path}.pth'))
		if optimer.upper() == 'SGD':
			self.optimer = torch.optim.SGD(
					self.net.parameters(), lr=kwargs['lr'], momentum=kwargs['momentum'],
					weight_decay=kwargs['weight_decay'])
		elif optimer.upper() == 'ADAMW':
			self.optimer = torch.optim.AdamW(
					self.net.parameters(), lr=kwargs['lr'], weight_decay=kwargs['weight_decay']
			)
		
		self.epoch = EPOCH
		if scheduler.upper() == 'COS':
			self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
					self.optimer, eta_min=kwargs['eta_min'], last_epoch=-1, T_max=EPOCH)
		elif scheduler.upper() == 'EXP':
			self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
					self.optimer, gamma=kwargs['gamma'], last_epoch=-1)
		elif scheduler.upper() == 'MUL':
			self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
					self.optimer, kwargs['milestones'], gamma=kwargs['gamma'])
		else:
			print('余弦退火算法:COS\n 指数下降算法:EXP \n 阶梯下降算法:\n')
			sys.exit()
		
		# 创建对应网络的文件夹
		if not os.path.exists(f'./Mdl_Data/{net_name}'): os.makedirs(f'./Mdl_Data/{net_name}')
	
	def train(self, epoch, train_data, colour='yellow'):
		pars_train = tqdm(train_data, total=len(train_data), file=sys.stdout, colour=colour)
		oa, miou, total, class_total, running_loss = 0, 0, 0, 0, 0
		########
		# 添加 acc 信息
		##########
		for index, data in enumerate(pars_train, 1):
			inputs, labels = data[0].to(self.device), data[1].to(self.device)
			y_pred = self.net(inputs)
			class_total += labels.size(0)
			total += 1
			self.optimer.zero_grad()
			loss = self.criterion(y_pred, labels)
			loss.backward()
			self.optimer.step()
			running_loss += loss.item()
			lr = "{:g}".format(self.scheduler.get_last_lr()[0] * 100)
			oa_temp, miou_temp = self.count_oa_miou(pred=y_pred, label=labels)
			oa += oa_temp
			miou += miou_temp
			pars_train.set_description("|train| --> |epoch:%.3d | lr:%s%% | loss:%.3f | OA:%.3f | Miou:%.3f" %
			                           (epoch, lr, running_loss / index, oa / total, miou / total))
			
			pars_train.update(1)
		pars_train.close()
		return running_loss / class_total, oa, miou
	
	def test(self, epoch, test_data, colour='blue'):
		pars_train = tqdm(test_data, total=len(test_data), colour=colour)
		oa, miou, total, class_total, running_loss = 0, 0, 0, 0, 0
		with torch.no_grad():
			for index, data in enumerate(pars_train, 1):
				inputs, labels = data[0].to(self.device), data[1].to(self.device)
				y_pred = self.net(inputs)
				classes = labels.size(0)
				class_total += classes
				total += 1
				loss = self.criterion(y_pred, labels)
				running_loss += loss.item()
				oa_temp, miou_temp = self.count_oa_miou(pred=y_pred, label=labels)
				oa += oa_temp
				miou += miou_temp
				lr = "{:g}".format(self.scheduler.get_last_lr()[0] * 100)
				
				pars_train.set_description("|test | --> |epoch:%.3d | lr:%s%% | loss:%.3f | OA:%.3f | Miou:%.3f" %
				                           (epoch, lr, running_loss / index, oa / total, miou / total))
				pars_train.update(1)
		
		return running_loss / class_total, oa, miou
	
	def run(self, colour_train='yellow', colour_test='blue'):
		
		with open(f'./LOG/{self.net_name}.txt', 'w') as f1:
			f1.write(f'Epoch:\tlr\tt_loss:\tt_oa\tt_miou\tv_loss\tv_oa\tv_miou\n')
			for epoch in range(self.epoch):
				self.net.train()
				t_loss, t_oa, t_miou = self.train(epoch, self.train_data, colour=colour_train)
				self.scheduler.step()
				self.net.eval()
				v_loss, v_oa, v_miou = self.test(epoch, self.test_data, colour_test)
				paths = os.path.join(f"./Mdl_Data/{self.net_name}", f"{epoch}.pth")
				torch.save(self.net.state_dict(), paths)
				lr = "{:g}".format(self.scheduler.get_last_lr()[0] * 100)
				f1.write(f'{epoch}\t{lr}\t{t_loss}\t{t_oa}\t{t_miou}\t{v_loss}\t{v_oa}\t{v_miou}\n')
	
	def ConfusionMatrix(slef, numClass, imgPredict, Label):
		"""

		:param numClass:
		:param imgPredict:
		:param Label:
		:return: numclass*numclass的混淆矩阵
		"""
		# 计算混淆矩阵
		mask = (Label >= 0) & (Label < numClass)
		label = numClass * Label[mask] + imgPredict[mask]
		count = np.bincount(label, minlength=numClass * numClass)
		# count = count[:numClass**2]
		confusionMatrix = count.reshape(numClass, numClass)
		return confusionMatrix
	
	def OverallAccuracy(self, confusionMatrix):
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
	
	def MeanIntersectionOverUnion(self, confusionMatrix):
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
	
	def count_oa_miou(self, pred, label):
		
		pred = torch.argmax(pred, dim=1).cpu().numpy()
		label = label.cpu().numpy()
		matrix = self.ConfusionMatrix(self.num_class, pred, label)
		oa = self.OverallAccuracy(matrix)
		miou = self.MeanIntersectionOverUnion(matrix)
		return oa, miou
	
	def mean_std(self):
		pop_mean = []
		pop_std0 = []
		for data in tqdm(self.train_data, total=len(self.train_data)):
			numpy_image = data[0].numpy()
			batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
			batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
			pop_mean.append(batch_mean)
			pop_std0.append(batch_std0)
		pop_mean = np.array(pop_mean).mean(axis=0)
		pop_std0 = np.array(pop_std0).mean(axis=0)
		print(f" mean, std =torch.tensor({pop_mean}),torch.tensor({pop_std0})")
		sys.exit()