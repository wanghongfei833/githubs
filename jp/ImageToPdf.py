# -*- coding: utf-8 -*-
# @Time    : 2022/10/23 14:19
# @Author  : HongFei Wang
import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.messagebox import *

from PIL import Image


class ImageToPdf(object):
	def __init__(self, win):
		self.win = tk.Toplevel(win)
		self.win.geometry('300x230+600+300')
		self.win.resizable(False, False)
		self.win.title('竞品生成')
		self.root = win
		self.file = StringVar()
		self.save = StringVar()
		
		win.withdraw()
		self.win.protocol('WM_DELETE_WINDOW', self.close)
	
	def main(self):
		Button(self.win, text='选择图片文件', command=self.file_path).grid(row=0, column=0)
		Entry(self.win, textvariable=self.file).grid(row=0, column=1)
		
		Button(self.win, text='保存pdf路径', command=self.save_path).grid(row=1, column=0)
		Entry(self.win, textvariable=self.save).grid(row=1, column=1)
		
		Button(self.win, text='生成PDF', command=self.run).grid(row=2, column=1)
	
	def run(self):
		if not self.file.get():
			showerror('错误', '请选择图片文件')
			self.file_path()
		if not self.save.get():
			showerror('错误', '未指定保存位置')
			self.save_path()
		img = self.file.get()
		img = Image.open(self.file.get())
		img.save(self.save.get(), 'pdf', save_all=True)
		showinfo('完成', 'PDF 已经保存至 %s' % self.save.get())
	
	def close(self):
		self.win.destroy()
		self.root.wm_deiconify()
	
	def file_path(self):
		str = askopenfilename(filetypes=(("jpg files", "*.jpg"),
		                                 ("png files", "*.png")))
		self.file.set(str)
	
	def save_path(self):
		str = asksaveasfilename(defaultextension='.pdf')
		self.save.set(str)