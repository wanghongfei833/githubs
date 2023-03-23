# -*- coding: utf-8 -*-
# @Time    : 2022/11/20 23:24
# @Author  : HongFei Wang
import numpy as np
from numba import njit, prange
from tqdm import tqdm


@njit(cache=True, parallel=True, nogil=True)
def count_accelerate(image_data):
	return_data = np.zeros_like(image_data, dtype=np.float64)
	rows, cols = image_data.shape
	for row in prange(rows):
		for col in prange(cols):
			return_data[row][col] = image_data[row][col].real ** 2 + \
			                        image_data[row][col].imag ** 2
	return return_data


def count_no_accelerate(image_data):
	return_data = np.zeros_like(image_data, dtype=np.float64)
	rows, cols = image_data.shape
	par = tqdm(range(rows))
	for row in par:
		for col in range(cols):
			return_data[row][col] = image_data[row][col].real ** 2 + \
			                        image_data[row][col].imag ** 2
	return return_data