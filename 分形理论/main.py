# -*- coding: utf-8 -*-
# @Time    : 2022/11/21 14:20
# @Author  : HongFei Wang
import tkinter as tk

import Libs


class Mains:
	def __init__(self, win_size: tuple, ):
		super(Mains, self).__init__()
		self.lib = None
		self.root = tk.Tk()
		screenwidth = self.root.winfo_screenwidth()  # 获取显示屏宽度
		screenheight = self.root.winfo_screenheight()  # 获取显示屏高度
		self.pos_window = [(screenwidth - win_size[0]) / 2, (screenheight - win_size[1]) / 2]
		size = '%dx%d+%d+%d' % (win_size[0], win_size[1], self.pos_window[0], self.pos_window[1])  # 设置窗口居中参数
		self.root.geometry(size)  # 让窗口居中显示
		# 截断点
		self.cats = []
		self.scale = tk.StringVar()  # 用以接受进度条的位置
		# 定义Lib类的基本信息
		self.file_image = tk.StringVar()  # 读取影像路径
		self.classes = tk.StringVar()  # 存储CA 还是S A
		self.plot_class = tk.StringVar()  # 存储绘制 点 or 线
		self.decimals = tk.IntVar()  # 保留小数
		self.log_base = tk.IntVar()  # log底数
		self.energy_plot = tk.BooleanVar()  # 能谱图是否绘制
		self.accelerate = tk.BooleanVar()  # 是否加速
		self.translation = tk.DoubleVar()  # 平移数量
		# 进行默认值设置
		self.classes.set('CA')
		self.plot_class.set('plot')
		self.decimals.set(3)
		self.log_base.set(10)
		self.energy_plot.set(False)
		self.accelerate.set(False)
		self.translation.set(1.)
	
	def set_canvs(self):
		print('调用')
		self.lib = Libs.Libs(classes=self.classes.get(),
		                     accelerate=self.accelerate.get(),
		                     decimals=self.decimals.get(),
		                     translation=self.translation.get(),
		                     log_base=self.log_base.get(),
		                     energy_plot=self.energy_plot.get(),
		                     plot_class=self.plot_class.get())
		self.lib.main(self.file_image.get(), self.cats)
	
	def set_farm(self):
		tk.Checkbutton(self.root, text="加速", variable=self.accelerate).place(x=0, y=200)
		tk.Checkbutton(self.root, text="能谱图", variable=self.energy_plot).place(x=60, y=200)
		tk.Radiobutton(self.root, text='CA', variable=self.classes, value='CA').place(x=120, y=200)
		tk.Radiobutton(self.root, text='SA', variable=self.classes, value='SA').place(x=180, y=200)
		
		tk.Scale(self.root, from_=0, to=100, resolution=1, orient=tk.HORIZONTAL,
		         length=200, sliderlength=5, width=20, label='横截断', command=self.scale.set).place(x=0, y=230)
		tk.Button(self.root, text='打印', font=('宋体', 9), command=lambda: print(self.scale.get())).place(x=100, y=250)
	
	def main(self):
		self.set_farm()
		self.root.mainloop()


if __name__ == '__main__':
	ui = Mains(win_size=(300, 300))
	ui.main()