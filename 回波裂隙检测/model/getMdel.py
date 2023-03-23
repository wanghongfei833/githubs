# -*- coding: utf-8 -*-
# @Time    : 2023/2/5 11:52
# @Author  : HongFei Wang
import RestNets50
import swinT


def model(name, numclass):
	if name == "restnet50":
		return RestNets50.restNet50(numclass)
	elif name == "swinT":
		return swinT.swinT(num_classes=numclass)
	else:
		raise "错误 只能 restnet50 或者 swinT , 输入为{}".format(name)