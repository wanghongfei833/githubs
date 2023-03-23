# -*- coding: utf-8 -*-
# @Time    : 2022/10/16 8:26
# @Author  : HongFei Wang
import tkinter as tk

from PIL import Image, ImageTk

from CreatNextMonth import CompeteWookCreat
from GenerateCompeteWook import GenerateCompeteWook
from ImageToPdf import ImageToPdf


class MainUi(object):
	"""
	主窗口
	"""
	
	def __init__(self):
		self.win = tk.Tk()
		# self.win.geometry('420x400+500+200')
		self.width, self.height = 420, 400
		self.win.geometry(str(self.width) + 'x' + str(self.height) + '+'
		                  + str((self.win.winfo_screenwidth() - self.width) // 2) + '+'
		                  + str((self.win.winfo_screenheight() - self.height) // 2 - 18))
		# self.win.bind('<Configure>', self.change_win_size)  # 注册窗口变动事件
		self.win.resizable(False, False)
		self.CWC = CompeteWookCreat
		self.GCW = GenerateCompeteWook
		self.ImageToPdf = ImageToPdf
	
	def set_bg(self):
		# -----------图像显示----------------
		self.win.update()
		self.width, self.height = self.win.winfo_width(), self.win.winfo_height()
		self.bg = tk.Canvas(self.win, width=self.width, height=self.height)
		self.bg.place(x=0, y=0)
		self.img = Image.open('BGS.jpg')
		self.img = self.img.resize((self.width, self.height))
		self.bg_image = ImageTk.PhotoImage(self.img)
		self.bg.create_image(0, 0, anchor='nw', image=self.bg_image)
	
	def creat_fram(self):
		tk.Button(self.win, text='竞品生成', bd=0, bg='#E0EEEE', font=('等线', 15), fg='blue',
		          activeforeground='red', activebackground='yellow',
		          command=lambda: self.GCW(self.win).main()).place(x=10, y=300)
		tk.Button(self.win, text='下月竞品表', bd=0, bg='#E0EEEE', font=('等线', 15), fg='blue',
		          activeforeground='red', activebackground='yellow',
		          command=lambda: self.CWC(self.win).main()).place(x=300, y=300)
		
		tk.Button(self.win, text='图像转pdf', bd=0, bg='#E0EEEE', font=('等线', 15), fg='blue',
		          activeforeground='red', activebackground='yellow',
		          command=lambda: self.ImageToPdf(self.win).main()).place(x=150, y=300)
	
	def run(self):
		self.set_bg()
		self.creat_fram()
		self.win.mainloop()


if __name__ == '__main__':
	main = MainUi()
	main.run()