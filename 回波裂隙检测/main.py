# -*- coding: utf-8 -*-
# @Time    : 2023/2/5 11:43
# @Author  : HongFei Wang
import torch
from torchvision.transforms import transforms as tf

from model import RestNets50, swinT
from ulitS import *

# 基础参数
mean_stds = False
batch_size = 8
numclass = 8  # 分类数量
EPOCH = 100
best = 0.  # 最优秀模型
modelName = "cnn"  # cnn transform comform effic
transform = tf.Compose(
		[
				tf.Resize((512, 512)),
				tf.ToTensor(),
				tf.Normalize(torch.tensor([0.9108745, 0.90717196, 0.78152937]),
				             torch.tensor([0.15088639, 0.15373805, 0.1442216]))
		]
)

if __name__ == '__main__':
	trainLoader, testLoader = datasets(transform, batch_size=batch_size, root="./image")
	if mean_stds: mean_std(trainLoader)
	if modelName == "cnn":
		models = RestNets50.restNet50(numclass)
	elif modelName == "transform":
		models = swinT.swinT(num_classes=numclass)
	elif modelName == "comform":
		pass
	else:
		raise "网络错误!!"
	optimer = torch.optim.AdamW(models.parameters(), lr=1e-5, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimer, 0.95)
	device = torch.device("cuda:0")
	criterion = torch.nn.CrossEntropyLoss()
	models.to(device)
	f = open(f"log{modelName}.log", 'w')
	for epoch in range(EPOCH):
		tloss, tacc = train(trainLoader, scheduler, device, models, criterion, optimer, epoch)
		scheduler.step()
		vloss, vacc = Test(testLoader, scheduler, device, models, criterion, epoch)
		f.write(f"{epoch}\t{tloss}\t{tacc}\t{vloss}\t{vacc}")
		if vacc >= best:
			best = vacc
			torch.save(models.state_dict(), f"{modelName}.pth")
	f.close()