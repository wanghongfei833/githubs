# -*- coding: utf-8 -*-
# @Time    : 2023/2/15 8:59
# @Author  : HongFei Wang
import datetime
import os
import time

from torch import optim
from torchvision.transforms import transforms as tf

from model import getModel
from ulit import *


def main(arg):
	transform = {
			'train': tf.Compose([
					tf.ToTensor(),
					tf.Normalize(arg.mean, arg.std),
					tf.RandomCrop(224, 224),
					tf.RandomRotation(90),
			
			]),
			'test': tf.Compose([
					tf.ToTensor(),
					tf.Normalize(arg.mean, arg.std),
			])
	}
	model = getModel(2)
	trainloader, testloader = getDataloader(arg, transform=transform)
	lossFunction = torch.nn.CrossEntropyLoss()
	optimer = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimer, [arg.epoch * 2 // 10, arg.epoch * 5 // 10])
	best = 0.
	os.makedirs('./log', exist_ok=True)
	os.makedirs('./checkpoint', exist_ok=True)
	
	timestamp = time.time()  # 时间戳格式
	now = datetime.datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
	f = open(f'./log/{now}.txt', 'w')
	f.write(f'epoch\ttrainLoss\ttestLoss\ttrainAcc\ttestAcc\n')
	f.close()
	for epoch in range(arg.epoch):
		f = open(f'./log/{now}.txt', 'a')
		tl, ta = train(trainloader, epoch, lossFunction, scheduler, model, optimer, best)
		vl, va = Test(testloader, epoch, scheduler, model, best)
		scheduler.step()
		f.write(f'{epoch}\t{tl:.7f}\t{vl:.7f}\t{ta:.2f}\t{va:.2f}\t')
		f.close()
		if va > best:
			best = va
			torch.save(model.state_dict(), f'./checkpoint/{epoch + 1}.pth')


if __name__ == '__main__':
	arg = config()
	main(arg)