# -*- coding: utf-8 -*-
# @Time    : 2023/2/14 11:53
# @Author  : HongFei Wang


import os

filePath = r'E:\opencv'
for i, j, k in os.walk(filePath):
	print(i, j, k)
# if len(k) == 0:
# 	continue
# else:
# 	for name in k:
# 		path = os.path.join(i, name)
# 		print(path)