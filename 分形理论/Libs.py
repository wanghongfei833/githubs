# -*- coding: utf-8 -*-
# @Time    : 2022/11/20 22:19
# @Author  : HongFei Wang

import matplotlib.pyplot as plt
import numpy as np

from Lib.coun_nks import count_nks
from Lib.count_fft import count_accelerate, count_no_accelerate
from Lib.file_data import read_image
from Lib.matching import matching


class Libs(object):
	def __init__(self, classes: str = 'CA',
	             accelerate: bool = False,
	             decimals: int = 3,
	             translation: float = 0.01,
	             log_base: int = 10,
	             energy_plot=False,
	             plot_class='plot'):
		"""

		:param classes: 类别
		:param accelerate: 是否加速
		:param decimals: 保留小数
		:param translation: 是否平移
		:param log_base: Log的底数
		:param energy_plot: 是否绘制能谱图
		:param plot_class: 绘制点还是线


		:return
		"""
		super(Libs, self).__init__()
		
		assert type(decimals) == int, ValueError('decimals should be int but get {}'.format(decimals))
		assert decimals >= 0, ValueError('decimals should >=0 but get {}'.format(decimals))
		assert plot_class.upper() in ['PLOT', 'SCATTER'], ValueError('plot_class should in "PLOT or SCATTER" but get {}'.format(plot_class))
		self.plot_class = plot_class.upper()
		self.energy_plot = energy_plot
		self.decimals = decimals
		self.translation = translation
		self.log_base = log_base
		classes = classes.upper()
		self.classes = classes
		self.accelerate = accelerate
		
		self.fig, self.axes = plt.subplots(dpi=100)
		if self.energy_plot and self.classes == 'SA': self.fig_energy, self.axes_energy = plt.subplots(dpi=100)
		assert classes in ['CA', 'SA'], ValueError('classes only support "CA","SA" But get {}'.format(classes))
	
	def make_data(self, data: np.array):
		"""

		:param data:
		:return:
		"""
		if self.classes == 'SA': data = np.fft.fft2(data)  # 判断是否SA
		if self.accelerate:
			data = count_accelerate(data)  # 加速获取能谱图
		else:
			data = count_no_accelerate(data)  # 不加速获取能谱图
		return data
	
	def Plt_LogToLog(self, image_file_path):
		"""

		:param image_file_path:
		:return:
		"""
		image = read_image(image_file_path)
		assert len(image.shape) == 2, ValueError('Only grayscale images are supported but get {}'.format(image.shape))
		if self.classes == 'SA':
			image = self.make_data(image)
		logx, logy = count_nks(self, data=image)
		if self.energy_plot and self.classes == 'SA':
			im = self.axes_energy.imshow(np.log(image))
			self.fig_energy.colorbar(im)
		if self.plot_class == 'PLOT':
			self.axes.plot(logx, logy)
		elif self.plot_class == 'SCATTER':
			self.axes.scatter(logx, logy)
		self.axes.set_xlabel("Log C" if self.classes == 'CA' else 'Log S')
		self.axes.set_ylabel('log N')
		self.axes.set_title('C-A' if self.classes == 'CA' else 'S-A')
		
		return logx, logy
	
	def plt_line(self, inputs, labels, cat, plt_fig):
		r2, k, b, pred = matching(inputs, labels)
		if cat: plt_fig.vlines(x=cat,
		                       ymax=max(np.max(pred), np.max(labels)) + 1,
		                       ymin=min(np.min(pred), np.min(labels)) - 1,
		                       linestyles='dashed', linewidth=0.9, colors='#3C323C')  # 垂线图
		plt_fig.plot(inputs, pred, label=f'{k:.2f}X+{b:.2f},r2={r2:.2f}')
		plt_fig.legend()
	
	def main(self, image_file_path, args: list, **kwargs):
		log_x, log_y = self.Plt_LogToLog(image_file_path)
		# 异常点移除
		if 'star' in kwargs.keys(): log_x, log_y = log_x[kwargs['star']:], log_y[kwargs['star']:]
		# 进行截断
		cats = [int(len(log_x) * cat) for cat in args]  # 收集折断点
		cats.append(len(log_x))
		cats.insert(0, 0)
		for star, end in zip(cats[:-1], cats[1:]):
			x_tmep, y_temp = log_x[star:end], log_y[star:end]
			try:
				self.plt_line(x_tmep, y_temp, cat=log_x[end], plt_fig=self.axes)
			except:
				self.plt_line(x_tmep, y_temp, cat='', plt_fig=self.axes)
		plt.show()
		
		return 1, 2

# functions = Libs(classes='sa', accelerate=True, energy_plot=True)
# x, y = functions.main('band1.dat', [0.45, 0.9])