# -*- coding: utf-8 -*-
# @Time    : 2022/12/15 14:51
# @Author  : HongFei Wang
import sys

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from Models import get_model
from dataset import DataSetMySelf


def train(tran_data, epoch, scheduler, device, net, optimizer, criterion):
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
		optimizer.zero_grad()
		loss = criterion(y_pred, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		lr = "{:g}".format(scheduler.get_last_lr()[0])
		pars_train.set_description("{train-%d}:\tlr:%s\tloss:%.3f\tacc:%.2f%% \t%d/%d\tlayer:%d" %
		                           (epoch, lr, running_loss / index, class_correct * 100 / class_total, class_correct, class_total, index))
		
		pars_train.update(1)
	pars_train.close()
	return running_loss / class_total


def Test(test_data, epoch, scheduler, device, net, optimizer, criterion):
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
			lr = "{:g}".format(scheduler.get_last_lr()[0])
			
			pars_train.set_description("{test-%d}:\tlr:%s\tloss:%.3f\tacc:%.2f%% \t%d/%dlayers:%d" %
			                           (epoch, lr, running_loss / index,
			                            class_correct * 100 / class_total,
			                            class_correct, class_total, index))
			pars_train.update()
	
	return class_correct * 100 / class_total


def main():
	# 参数
	lr = 1e-5
	Epoch = 50
	star_epoch = 0
	net = get_model(True)
	milestones = [100, 200]
	
	# 获取数据
	transform = transforms.ToTensor()
	tra_data = DataSetMySelf(train=True, trainsform=transform)
	val_data = DataSetMySelf(train=False, trainsform=transform)
	tra_loader = DataLoader(tra_data, batch_size=32, shuffle=True, num_workers=2)
	val_loader = DataLoader(val_data, batch_size=32, shuffle=True, num_workers=2)
	# 定义信息
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-4)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	net.to(device)
	best = 0.
	for epoch in range(star_epoch, star_epoch + Epoch):
		f = open(f'../log/LOG.txt', 'a')
		net.train()
		t_loss = train(tra_loader, epoch, scheduler, device, net, optimizer, criterion)
		scheduler.step()
		net.eval()
		v_acc = Test(val_loader, epoch, scheduler, device, net, optimizer, criterion)
		if v_acc > best:
			best = v_acc
			torch.save(net.state_dict(), f"../weight/modelResult.pth")
		lr = "{:g}".format(scheduler.get_last_lr()[0])
		texts = "epoch\t%d\tt_loss\t%.5f\tval_acc\t%.2f%%\t%s\n" % (epoch, t_loss, v_acc, lr)
		f.write(str(texts))
		f.close()


if __name__ == '__main__':
	# main()
	
	prebs = torch.tensor([[0.5, 0.5, 0.5],
	                      [0.5, 0.5, 0.5],
	                      [0.5, 0.5, 0.5]])
	
	targets = torch.tensor([[0.5, 0.4, 0.5],
	                        [0.5, 0.5, 0.5],
	                        [0.5, 0.5, 0.5]])
	
	print(prebs.size())
	pred = prebs == targets
	pred = torch.sum(pred, dim=1)
	print(pred.size())
	print((pred == 3).sum())