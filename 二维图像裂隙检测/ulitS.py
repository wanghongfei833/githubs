# -*- coding: utf-8 -*-
# @Time    : 2023/2/5 11:41
# @Author  : HongFei Wang
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataloader


def train(tran_data, scheduler, device, net, criterion, optimizer, epoch):
	pars_train = tqdm(tran_data, file=sys.stdout)
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
		running_loss += loss.item()
		pars_train.set_description("{train-%d}:\tlr:%.7f\tmean:%.4f\tacc:%.2f%%" %
		                           (epoch, scheduler.get_last_lr()[0], running_loss / index,
		                            class_correct * 100 / class_total))
		pars_train.update(1)
	return running_loss / class_total, class_correct * 100 / class_total


def Test(test_data, scheduler, device, net, criterion, epoch):
	pars_train = tqdm(test_data, total=len(test_data), colour="blue")
	running_loss = 0.
	class_correct, class_total = 0, 0
	with torch.no_grad():
		for index, data in enumerate(pars_train, 1):
			inputs, labels = data[0].to(device), data[1].to(device)
			y_pred = net(inputs)
			prediction = torch.argmax(y_pred, 1)
			reall = (prediction == labels).sum().cpu()
			class_correct += reall
			classes = labels.size(0)
			class_total += classes
			loss = criterion(y_pred, labels)
			running_loss += loss.item()
			pars_train.set_description("{test-%d}:\tlr:%.7f\tmean:%.4f\tacc:%.2f%%" %
			                           (epoch, scheduler.get_last_lr()[0], running_loss / index,
			                            class_correct * 100 / class_total))
			pars_train.update(1)
	
	return running_loss / class_total, class_correct * 100 / class_total


def datasets(tf, batch_size, root):
	train_set = dataloader.DataSetsSelf(model="train", root=fr"{root}", transform=tf)
	test_set = dataloader.DataSetsSelf(model="test", root=fr"{root}", transform=tf)
	train_loader = DataLoader(train_set, batch_size, shuffle=True, pin_memory=True)
	test_loader = DataLoader(test_set, batch_size, shuffle=True, pin_memory=True)
	return train_loader, test_loader


def mean_std(dataloader):
	pop_mean = []
	pop_std0 = []
	for data in tqdm(dataloader, total=len(dataloader)):
		numpy_image = data[0].numpy()
		batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
		batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
		pop_mean.append(batch_mean)
		pop_std0.append(batch_std0)
	pop_mean = np.array(pop_mean).mean(axis=0)
	pop_std0 = np.array(pop_std0).mean(axis=0)
	# 'mean, std = torch.tensor([0.1925, 0.1423, 0.1081]), torch.tensor([0.4675, 0.4374, 0.3624])'
	print(f" mean, std =torch.tensor({pop_mean}),torch.tensor({pop_std0})")
	sys.exit()