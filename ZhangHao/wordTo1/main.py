# -*- coding: utf-8 -*-
# @Time    : 2023/2/7 9:22
# @Author  : HongFei Wang
import os
import shutil
import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import *

import doc2docx
import words2ines


def file(string):
	path = askdirectory()
	string.set(path)


# def save(string):
# 	path = asksaveasfilename()
# 	if ".docx" not in path: path += ".docx"
# 	string.set(path)


def main(filepath: str, savepath: str, show):
	# 摸底-->镇-->村-->组
	temp_make = os.path.join("C:\\temp_this")
	
	try:
		shutil.rmtree(temp_make)
		print("整理文件夹")
	except:
		pass
	os.makedirs(temp_make, exist_ok=True)  # 清理文件夹
	for i, j, k in os.walk(filepath):
		if len(k) == 0:
			continue
		else:
			for name in k:
				path = os.path.join(i, name)  # 获取所有路径
				save_name = path.replace('/', '\\')
				save_name = save_name.split('/')[:-1]
				save_name = '-'.join(save_name)
				save_name = save_name + '.docx'
				shutil.copy(os.path.join(path, name), os.path.join(temp_make, name[:4] + name[-4:]))  # 把文件移动到临时文件夹
				# 执行运行 doc转docx
				doc2docx.doc_to_docx(temp_make, show)
				# 执行合并word
				sourList = [os.path.join(temp_make, i) for i in os.listdir(temp_make) if ".docx" in i]
				words2ines.merge_doc(sourList, os.path.join(savepath, save_name))
				# 最后删除临时文件
				print("{:*^50}".format("删除临时文件中"))
				for name in os.listdir(temp_make):
					os.remove(os.path.join(temp_make, name))
				print("{:*^50}".format("删除完成"))
				print("{:*^50}".format(f"{save_name} 执行完成.."))
	
	messagebox.showinfo("完成", "文件已保存至{}".format(savepath))


def changebllo(data, data2):
	if data.get():
		data.set(False)
		data2.set("不显示")
	else:
		data.set(True)
		data2.set(" 显示")


if __name__ == '__main__':
	root = tk.Tk()
	screen_width = root.winfo_screenwidth()
	screen_height = root.winfo_screenheight()
	files = tk.StringVar()
	saves = tk.StringVar()
	shows = tk.BooleanVar()
	shows.set(False)
	call = tk.StringVar()
	call.set("不显示")
	x = int(screen_width / 2 - 300 / 2)
	y = int(screen_height / 2 - 300 / 2)
	size = '{}x{}+{}+{}'.format(300, 350, x, y)
	root.geometry(size)
	tk.Button(root, text="选择文件夹", font=("宋体", 20), command=lambda: file(files)).grid(row=0, column=0, padx=80, pady=20)
	tk.Button(root, text="保存文件名", font=("宋体", 20), command=lambda: file(saves)).grid(row=1, column=0, padx=80, pady=20)
	tk.Button(root, text="开始运行", font=("宋体", 20), command=lambda: main(files.get(), saves.get(), shows.get())).grid(row=2, column=0, padx=80,
	                                                                                                                      pady=20)
	tk.Radiobutton(root, text="  显示word", variable=shows, value=True).grid(row=3, column=0)
	tk.Radiobutton(root, text="不显示word", variable=shows, value=False).grid(row=4, column=0)
	
	root.mainloop()