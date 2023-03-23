# -*- coding: utf-8 -*-
# @Time    : 2022/12/6 11:35
# @Author  : HongFei Wang
import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import askdirectory, asksaveasfilename

from DocToDocx import run as docrun
from dels import run as delrun


def open_file(path):
	path_ = askdirectory()
	path.set(path_)


def save_path(path):
	path_ = asksaveasfilename(defaultextension='.docx')
	path.set(path_)


def main(file, save, stop, docs):
	assert type(stop) == float, messagebox.showerror('错误', 'stop应该是小数')
	path = docrun(file, stops=stop, docs=docs)
	delrun(path, save)
	messagebox.showinfo('完成', '文件保存至' + save)


win_size = [300, 300]
win = tk.Tk()
screenwidth = win.winfo_screenwidth()  # 获取显示屏宽度
screenheight = win.winfo_screenheight()  # 获取显示屏高度
pos_window = [(screenwidth - win_size[0]) / 2, (screenheight - win_size[1]) / 2]
size = '%dx%d+%d+%d' % (win_size[0], win_size[1], pos_window[0], pos_window[1])  # 设置窗口居中参数
win.geometry(size)  # 让窗口居中显示
file = tk.StringVar()
file.set('E:/taobao/ZhangHao/二组')
save = tk.StringVar()
save.set('E:/taobao/ZhangHao/test.docx')
stops = tk.DoubleVar()
stops.set(0.1)
docs = tk.IntVar()
docs.set(1)
tk.Button(win, text='选择文件夹', command=lambda: open_file(file)).grid(row=0, column=0, pady=10)
tk.Entry(win, textvariable=file, width=30).grid(row=0, column=1, pady=10, padx=0)
tk.Button(win, text='存储路径', command=lambda: save_path(save)).grid(row=1, column=0, pady=10, padx=0)
tk.Entry(win, textvariable=save, width=30).grid(row=1, column=1, pady=10, padx=0)
tk.Label(win, text='停顿').grid(row=2, column=0, pady=10, padx=0)
tk.Entry(win, textvariable=stops).grid(row=2, column=1, pady=10, padx=0)
tk.Label(win, text='最大doc').grid(row=3, column=0, pady=10, padx=0)
tk.Entry(win, textvariable=docs).grid(row=3, column=1, pady=10, padx=0)
tk.Button(win, text='运行', font=('宋体', 15),
          command=lambda: main(file.get(), save.get(), stops.get(), 1)).grid(row=5, column=0, pady=10)
tk.mainloop()