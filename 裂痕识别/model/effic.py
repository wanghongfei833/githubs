import torch
from torchvision import models


def effic(numclass):
	model = models.resnet18(pretrained=True)
	model.fc = torch.nn.Linear(512, numclass)
	# model.classifier = torch.nn.Sequential(
	# 	torch.nn.Dropout(0.99), torch.nn.Linear(2560, numclass)
	# )
	return model