# -*- coding: utf-8 -*-
# @Time    : 2023/2/5 11:43
# @Author  : HongFei Wang
import torch
from torchvision.transforms import transforms as tf

from model import RestNets50, lenet, swinT
from ulitS import *

# 基础参数
mean_stds = True
batch_size = 32
numclass = 15  # 分类数量
EPOCH = 100
best = 0.  # 最优秀模型
modelName = "transform"  # dcnn transform cnn
dataFile = "D:\data\Radiolarian"
transform = tf.Compose(
		[
				tf.ToTensor(),
				tf.Normalize(mean=[0.23885676, 0.23925619, 0.23907427],
				             std=[0.304022, 0.3047145, 0.30398583]),
		]
)
# import matplotlib.pyplot as plt
# a = np.random.random((4,4))
# plt.imshow(a)
# plt.show()


if __name__ == '__main__':
	trainLoader, testLoader = datasets(transform, batch_size=batch_size, root=dataFile)
	if mean_stds: mean_std(trainLoader)
	if modelName == "dcnn":
		models = RestNets50.restNet50(numclass)
	elif modelName == "transform":
		models = swinT.swinT(num_classes=numclass)
	elif modelName == "cnn":
		models = lenet.LetNet(numclass)
	else:
		raise "网络错误!!"
	optimer = torch.optim.AdamW(models.parameters(), lr=1e-2, weight_decay=1e-5)
	# 9e-6-->1.8e-6
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimer, 0.95)  # 可以修改
	device = torch.device("cuda:0")
	criterion = torch.nn.CrossEntropyLoss()
	models.to(device)
	
	# models.load_state_dict(torch.load("cnn.pth"))
	# for data,lable in trainLoader:
	# 	pred,sigma_ouput = models(data)
	#######################################3
	f = open(f"log_{modelName}.log", 'w')
	f.write(f"epoch\ttloss\ttacc\tvloss\tvacc\n")
	for epoch in range(EPOCH):
		tloss, tacc = train(trainLoader, scheduler, device, models, criterion, optimer, epoch)
		scheduler.step()
		vloss, vacc = Test(testLoader, scheduler, device, models, criterion, epoch)
		f.write(f"{epoch}\t{tloss}\t{tacc}\t{vloss}\t{vacc}\n")
		if vacc >= best:
			best = vacc
			torch.save(models.state_dict(), f"{modelName}.pth")
	f.close()