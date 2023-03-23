# -*- coding: utf-8 -*-
# @Time    : 2022/9/4 10:33
# @Author  : HongFei Wang
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

a = np.array(
		[[0, 2, 1],
		 [2, 1, 0],
		 [1, 0, 2]]
)

b = np.array(
		[[1, 2, 1],
		 [2, 1, 2],
		 [1, 2, 2]]
)

colors = ['white', 'blue', 'cyan', 'Lime', 'yellow']
# cmap = mpl.colors.ListedColormap(colors)
a_max, a_min = a.max(), a.min()
b_max, b_min = b.max(), b.min()
plt.figure()
plt.imshow(a, cmap=mpl.colors.ListedColormap(colors[a_min:a_max + 1]))
plt.figure()
plt.imshow(b, cmap=mpl.colors.ListedColormap(colors[b_min:b_max + 1]))
plt.show()