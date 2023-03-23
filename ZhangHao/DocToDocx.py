# -*- coding: utf-8 -*-
# @Time    : 2022/12/6 11:42
# @Author  : HongFei Wang

# -*- coding:utf-8 -*-
# @Author : yyzhang
import os
import shutil
import time

import tqdm
from win32com import client


# 转换doc为docx
def doc2docx(file, save, word, stop):
	try:
		word = client.Dispatch("Word.Application")  # 打开word应用程序
		word.Visible = True  # 后台运行,不显示
		word.DisplayAlerts = True  # 不警告
		doc = word.Documents.Open(file)
		doc.SaveAs(save, 12)
		doc.Close()
		# 文件另存为，也可以自定义目录，12表示docx格式
	except:
		time.sleep(stop)
		doc2docx(file, save, word, stop)
		print(file + '执行重掉')


def fun_doc2docx(files, saves, stop=0.1):
	for file, save in zip(files, saves):
		time.sleep(stop)
		doc2docx(file, save, 0, stop=stop)
	
	time.sleep(stop)


def run(file_path, stops=0.1, docs=5):
	firle_path = file_path
	save_path = firle_path.replace('/', '\\')
	save_path = "\\".join(save_path.split('\\')[:-1]) + "\\Change"
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	file_dir = os.listdir(firle_path)
	dicts = {}
	remove = []
	# 文件重命名
	for path in file_dir:
		if '.doc' not in path or '$' in path: continue
		path_title = path[:4]
		dicts[path_title] = path
		save_path_temp = os.path.join(save_path, path_title + '.doc')
		file_tmep = os.path.join(firle_path, path)
		try:
			shutil.copy(file_tmep, save_path_temp)
		except:
			print(f'{path_title}已存在')
	# 进行doc转docx
	
	change_par = tqdm.tqdm(dicts.keys())
	lasts = list(dicts.keys())[-1]
	filess, savess = [], []
	for path in change_par:
		file_tmep = os.path.join(save_path, path + ".doc")
		saves = os.path.join(save_path, dicts[path] + 'x')
		filess.append(file_tmep)
		savess.append(saves)
		# doc2docx(file_tmep,saves,0,stops)
		if path == lasts or len(filess) == docs:
			fun_doc2docx(filess, savess, stops)
			filess, savess = [], []
		
		remove.append(os.path.join(save_path, path + '.doc'))
		change_par.set_description_str('文件转换')
	
	remove_par = tqdm.tqdm(remove)
	for remo in remove_par:
		os.remove(remo)
		remove_par.set_description_str('删除临时文件:' + remo)
	
	return save_path