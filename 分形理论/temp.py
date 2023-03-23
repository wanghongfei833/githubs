import tkinter as tk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Test:
	def __init__(self, master):
		self.master = master
		self.b1 = tk.Button(self.master, text='绘制饼状图', command=self.b)
		self.b1.pack()
		self.b2 = tk.Button(self.master, text='绘制柱状图', command=self.z)
		self.b2.pack()
	
	def b(self):
		win1 = tk.Toplevel(self.master)
		win1.title('饼图')
		win1.geometry('600x400')
		x = ['猫', '狗', '鸡', '鸭', '鹅']
		y = [27, 35, 20, 48, 36]
		# 画布大小及分辨率
		fig1 = plt.figure(figsize=(6, 4), dpi=100)
		ax1 = fig1.add_subplot(111)
		# 创建画布
		canvas1 = FigureCanvasTkAgg(fig1, master=win1)
		canvas1.draw()
		canvas1.get_tk_widget().grid()
		# 显示中文
		plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用指定的汉字字体类型（此处为黑体）
		# 绘图
		ax1.pie(y, labels=x, autopct='%0.2f%%')
		ax1.set_title('饼状图test')
	
	def z(self):
		win2 = tk.Toplevel(self.master)
		win2.title('柱状图图')
		win2.geometry('600x400')
		x = ['猫', '狗', '鸡', '鸭', '鹅']
		y = [27, 35, 20, 48, 36]
		# 显示中文
		plt.rcParams['font.sans-serif'] = ['SimHei']
		##画布大小及分辨率
		fig2 = plt.figure(figsize=(6, 4), dpi=100)
		ax2 = fig2.add_subplot(111)
		# 创建画布
		canvas2 = FigureCanvasTkAgg(fig2, master=win2)
		canvas2.draw()
		canvas2.get_tk_widget().grid()
		# 绘图
		for i in range(len(x)):
			ax2.bar(x[i], y[i])
		for a, b in zip(x, y):
			plt.text(a, b, b, ha='center', va='bottom')
		ax2.set_xlabel('动物')
		ax2.set_ylabel('数量')
		ax2.set_title('柱状图test')


if __name__ == '__main__':
	root = tk.Tk()
	root.title('测试脚本')
	root.geometry('400x200')
	Test(root)
	root.mainloop()