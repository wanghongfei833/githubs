# -*- coding: utf-8 -*-
# @Time    : 2022/8/23 9:49
# @Author  : HongFei Wang
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from data import Datasets
from model import UNet
from ulits import Units

# todesk
hyper_parameter = {
		'lr': 1e-4,  # 学习率  ****
		'weight_decay': 5e-4,  # 权重衰减
		'epoch': 100,  # 训练次数         ****
		'pth_path': '5',  # 预训练权重 不使用则空就行  ****
		'data_root': './datas',  # 数据路径
		# datas /{ image label}
		'mean_std_count': False,  # 计算均值方差
		'num_class': 5,  # *
		'num_workers': 0,
		'batch_size': 4
	
}

transform = transforms.Compose([
		transforms.Normalize(mean=torch.tensor([0.3706616, 0.2910536, 0.21465479, 0.47991952]),
		                     std=torch.tensor([0.04082108, 0.03933582, 0.04168272, 0.14526395])),
])

train_data = Datasets(train=True,
                      root=hyper_parameter['data_root'],
                      tfs=transform,
                      numclass=hyper_parameter['num_class'])
test_data = Datasets(train=False,
                     root=hyper_parameter['data_root'],
                     tfs=transform,
                     numclass=hyper_parameter['num_class'])
train_loader = DataLoader(dataset=train_data,
                          batch_size=hyper_parameter['batch_size'],
                          num_workers=hyper_parameter['num_workers'],
                          shuffle=True, pin_memory=True)
test_loader = DataLoader(dataset=test_data,
                         batch_size=hyper_parameter['batch_size'],
                         num_workers=hyper_parameter['num_workers'],
                         shuffle=True, pin_memory=True)

mains = Units(net=UNet(hyper_parameter['num_class']),
              num_class=hyper_parameter['num_class'],
              net_name='unet',
              EPOCH=hyper_parameter['epoch'],
              train_data=train_loader,
              test_data=test_loader,
              optimer='adamw',
              scheduler='MUL',
              best=0, mdl_path=hyper_parameter['pth_path'],
              lr=hyper_parameter['lr'],
              milestones=[250, 300],
              gamma=0.1,
              weight_decay=hyper_parameter['weight_decay'])

if __name__ == '__main__':
	if hyper_parameter['mean_std_count']:
		mains.mean_std()
	else:
		mains.run()