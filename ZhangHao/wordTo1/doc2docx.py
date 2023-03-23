# -*- coding: utf-8 -*-
# @Time    : 2023/2/7 9:24
# @Author  : HongFei Wang
import os
import time

import tqdm
from win32com import client


def changeName(path):
	for file in os.listdir(path):
		file_path = os.path.join(path, file)
		saves = os.path.join(path, file[:4] + file[-4:])
		os.rename(file_path, saves)


def doc_to_docx(file, show):
	filename_list = tqdm.tqdm([i for i in os.listdir(file) if i.split(".")[-1] == "doc"])
	word = client.Dispatch("Word.Application")  # 打开word应用程序
	word.Visible = show  # 后台运行,不显示
	word.DisplayAlerts = True  # 不警告
	try:
		for files in filename_list:
			# 将doc的文件名换成后缀为docx的文件
			file_open = os.path.join(file, files)
			# 将我们的docx与文件保存位置拼接起来，获得绝对路径
			doc = word.Documents.Open(file_open)  # 打开word文件
			doc.SaveAs("{}".format(file_open + "x"), 12, False, "", True, "", False,
			           False, False,
			           False)  # 转换后的文件,12代表转换后为docx文件
			doc.Close()  # 关闭原来word文件
			time.sleep(0.2)
	except Exception as e:
		print(e)
	word.Quit()