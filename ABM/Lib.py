# -*- coding: utf-8 -*-
# @Time    : 2022/12/4 9:51
# @Author  : HongFei Wang
import random

import numpy as np


def updateMap(map_size: int,
              people_ones_list,
              people_twos_list, maps):
	for i in range(map_size):
		for j in range(map_size):
			if [i, j] in people_ones_list:
				maps[i, j] = 1
			elif [i, j] in people_twos_list:
				maps[i, j] = 2
			else:
				maps[i, j] = 0
	return maps


def setpoint(map_size, space, ones):
	POS = [i for i in range(map_size * map_size)]
	random.shuffle(POS)
	space_list = POS[:space]
	POS = POS[space:]
	people_ones_list = POS[:ones]
	people_twos_list = POS[ones:]
	people_ones_list = [[i // map_size, i % map_size] for i in people_ones_list]
	people_twos_list = [[i // map_size, i % map_size] for i in people_twos_list]
	space_list = [[i // map_size, i % map_size] for i in space_list]
	return people_ones_list, people_twos_list, space_list


def randomMove(Ones, Twos, Space, Linjie, Maps: np.array):
	"""

	:param Maps: 底图数组
	:param Ones:第一类列表
	:param Twos: 第二类列表
	:param Space: 第三类列表
	:param Linjie: 临界满意值
	:return:
	"""
	
	width, height = Maps.shape
	tmep_map = np.zeros((width + 2, height + 2))
	tmep_map[1:-1, 1:-1] += Maps
	manyi_tmp = np.zeros_like(Maps)
	for i in range(width):
		if i == 0 or i == width - 1:
			x_temp = -1
		else:
			x_temp = 0
		for j in range(height):
			if j == 0 or j == width - 1:
				y_tmp = -1
			else:
				y_tmp = 0
			sums = (3 + x_temp) * (3 + y_tmp)
			if Maps[i, j] == 0:  # 背景
				continue
			else:
				tmp = tmep_map[i:i + 3, j:j + 3]
				if Maps[i, j] == 1 and (np.sum(tmp == 1) / sums) >= Linjie:  # 第一类
					manyi_tmp[i, j] = (np.sum(tmp == 1) / sums)
					continue
				elif Maps[i, j] == 2 and (np.sum(tmp == 2) / sums) >= Linjie:  # 第二类
					manyi_tmp[i, j] = (np.sum(tmp == 2) / sums)
					continue
				elif Maps[i, j] == 1 and (np.sum(tmp == 1) / sums) < Linjie:  # 搬家
					manyi_tmp[i, j] = (np.sum(tmp == 1) / sums)
					temp_space = Space.pop(0)
					Space.append([i, j])
					Ones.remove([i, j])
					Ones.append(temp_space)
					random.shuffle(Space)
				elif Maps[i, j] == 2 and (np.sum(tmp == 2) / sums) < Linjie:  # 搬家
					manyi_tmp[i, j] = (np.sum(tmp == 2) / sums)
					temp_space = Space.pop(0)
					Space.append([i, j])
					Twos.remove([i, j])
					Twos.append(temp_space)
					random.shuffle(Space)
	
	manyidu = np.sum(manyi_tmp) / (len(Ones) + len(Twos))
	return Ones, Twos, Space, Maps, manyidu if manyidu < 0.98 else None