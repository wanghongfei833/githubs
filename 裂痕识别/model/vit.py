import torch
from torchvision import models


def vit(numclass):
	modle = models.vit_b_16(num_classes=numclass)
	model_dicts = torch.load("./model/vit_b_16.pth")
	loads = {}
	for k, v in model_dicts.items():
		if "heads" not in k:
			loads[k] = v
	modle.load_state_dict(loads, strict=False)
	del model_dicts
	return modle