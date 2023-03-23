# -*- coding: utf-8 -*-
# @Time    : 2022/10/16 8:24
# @Author  : HongFei Wang
import datetime
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.messagebox import *

import numpy as np
import xlrd
import xlwt


class GenerateCompeteWook:
	def __init__(self, win):
		self.win = tk.Toplevel(win)
		self.win.geometry('550x300+550+300')
		self.win.resizable(True, False)
		self.win.title('竞品表格生产')
		self.add_bool = False
		self.root = win
		self.root.withdraw()
		self.win.protocol('WM_DELETE_WINDOW', self.close)
		# --------------定义一系列参数------------
		# 今年业绩 去年业绩 land路径 保存路径
		# 今年业绩数据 去年业绩数据 mtd今年 mtd去年 ytd今年 ytd去年
		self.path = [StringVar() for _ in range(4)]
		self.sheet_name = [StringVar() for _ in range(4)]
		self.excel = ['' for _ in range(3)]
		self.sheet_names = ['' for _ in range(3)]
		self.performance = [{} for _ in range(6)]
		
		# 时间
		
		data_time = datetime.datetime.now()
		year, month, day = [int(i) for i in str(data_time).split(' ')[0].split('-')]
		self.year = IntVar()
		self.month = IntVar()
		self.day = IntVar()
		self.year.set(year)
		self.month.set(month)
		self.day.set(day)
	
	def set_fram(self):
		Button(self.win, text='今年业绩', command=lambda: self.file_path(self.path[0], 0, 0)).grid(row=0, column=0, pady=10)
		Entry(self.win, textvariable=self.path[0]).grid(row=0, column=1, pady=10)
		Button(self.win, text='今年竞品').grid(row=0, column=2)
		self.set_pool_down(Strs=self.sheet_name[0], Value=self.sheet_names[0], row=0, col=3, padx=5, default_index=2)
		
		Button(self.win, text='去年业绩', command=lambda: self.file_path(self.path[1], 1, 1)).grid(row=1, column=0, pady=10)
		Entry(self.win, textvariable=self.path[1]).grid(row=1, column=1, pady=10)
		Button(self.win, text='去年竞品').grid(row=1, column=2)
		self.set_pool_down(Strs=self.sheet_name[1], Value=self.sheet_names[1], row=1, col=3, padx=5, default_index=2)
		
		Button(self.win, text='landing', command=lambda: self.file_path(self.path[2], 2, 2)).grid(row=2, column=0, pady=10)
		Entry(self.win, textvariable=self.path[2]).grid(row=2, column=1, pady=10)
		Button(self.win, text='去年竞品').grid(row=1, column=2)
		self.set_pool_down(Strs=self.sheet_name[1], Value=self.sheet_names[1], row=1, col=3, padx=5, default_index=2)
		Button(self.win, text='MTD信息').grid(row=2, column=2)
		self.set_pool_down(Strs=self.sheet_name[2], Value=self.sheet_names[2], row=2, col=3, padx=5, default_index=3)
		Button(self.win, text='YTD信息').grid(row=3, column=2)
		self.set_pool_down(Strs=self.sheet_name[3], Value=self.sheet_names[2], row=3, col=3, padx=5, default_index=4)
		
		Button(self.win, text='保存路径', command=lambda: self.save_path(self.path[3])).grid(row=3, column=0, pady=10)
		Entry(self.win, textvariable=self.path[3]).grid(row=3, column=1, pady=10)
		
		Button(self.win, text='生成竞品文件', command=lambda:
		self.write_excel(self.this_year, self.last_year, self.month, self.year) if self.add_bool else showerror('错误',
		                                                                                                        '请先点击增加数据,\n补充去年对应数据')).grid(
				row=4, column=1, pady=10)
		# ---------------------------------增加信息筛选------------------------------------------------------------------------------
		
		Button(self.win, text='新增数据', command=self.select_data).grid(row=0, column=4, pady=10)
		Entry(self.win, textvariable=self.month, width=2).grid(row=0, column=5, pady=10)
		Label(self.win, text='月', width=2).grid(row=0, column=6, pady=10)
		Entry(self.win, textvariable=self.day, width=2).grid(row=0, column=7, pady=10)
	
	def set_pool_down(self, **kwargs):
		cbox = ttk.Combobox(
				master=self.win,  # 父容器
				height=10,  # 高度,下拉显示的条目数量
				width=10,  # 宽度
				state="readonly",  # 设置状态 normal(可选可输入)、readonly(只可选)、 disabled
				cursor="arrow",  # 鼠标移动时样式 arrow, circle, cross, plus...
				font=("等线", 12),  # 字体
				textvariable=kwargs['Strs'],  # 通过StringVar设置可改变的值
				values=kwargs['Value'],  # 设置下拉框的选项
		)
		cbox.grid(row=kwargs['row'], column=kwargs['col'], padx=kwargs['padx'])
		try:
			cbox.current(kwargs['default_index'])
		except:
			pass
	
	def select_data(self):
		
		self.this_year = self.MTD(self.excel[0], self.sheet_name[0].get())
		self.last_year = self.MTD(self.excel[1], self.sheet_name[1].get())
		self.month = self.Landong(self.excel[2], self.sheet_name[2].get())
		self.year = self.Landong(self.excel[2], self.sheet_name[3].get())
		# ----------------添加信息----------------
		self.add_week_key = set(self.this_year.keys()) - set(self.last_year.keys())  # 需要添加的旧业绩
		add_month = set(self.this_year.keys()) - set(self.month.keys())
		add_year = set(self.this_year.keys()) - set(self.year.keys())
		# 定义add参数
		self.add_week_str = [StringVar() for _ in self.add_week_key]
		self.add_month_str = [StringVar() for _ in add_month]
		self.add_year_str = [StringVar() for _ in add_year]
		if self.add_month_str:
			showerror('MTD错误', 'MTD:缺少 %s 业绩' % '、'.join(add_month))
		if self.add_year_str:
			showerror('YTD错误', 'YTD:缺少 %s 业绩' % '、'.join(add_year))
		if not self.add_month_str and not self.add_year_str:
			Label(self.win, text='去年业绩', font=('等线', 10)).grid(row=1, column=4, pady=10)
			for index, name in enumerate(self.add_week_key):
				Label(self.win, text=name, font=('等线', 10)).grid(row=2 + index, column=4, pady=2)
				Entry(self.win, textvariable=self.add_week_str[index], font=('等线', 10), width=5).grid(row=2 + index, column=5, pady=2)
				index += 1
		self.add_bool = True
	
	# -------------------基础函数----------
	
	def MTD(self, excel_: xlrd.open_workbook, sheet_name):
		"""

		:param excel_:表格文件
		:param sheet_name: 表格名称
		:param data_time: 时间
		:return: dict {产品:本周业绩}
		"""
		sheet = excel_.sheet_by_name(sheet_name)
		rows, cols = sheet.nrows, sheet.ncols
		pos = self.find_data(sheet, find_name='日期')
		pos = pos[0] if len(pos) == 1 else print('错误')
		titles = sheet.row_values(pos[0])[1:]
		title = []
		for i in titles:
			if 'TT' in i.upper():
				break
			else:
				title.append(i.replace('la_prairie', 'la prairie'))
		# title = '\t'.join(title).replace('la_prairie','la prairie').split('\t')
		# ------------获取时间---------------------
		# if not data_time:
		#     data_time = datetime.datetime.now()
		#     year, month, day = [int(i) for i in str(data_time).split(' ')[0].split('-')]
		# else:
		#     temp = str(datetime.datetime.now()).split(' ')[0].split('-')
		#     year, month, day = int(temp[0]), int(temp[1]), data_time
		week = datetime.date(self.year.get(), self.month.get(), self.day.get()).weekday()  # 周几
		index_start = self.day.get() - week  # 从 第几天开始算
		index_end = self.day.get() + 1  # 结束日期
		# --------------------筛选数据
		data_excle = []
		day_index = 0
		for row in range(pos[0], rows):
			try:
				# 获取每天日期
				int(sheet.cell(row, pos[1]).value)
				day_index += 1  # 第几天
				if index_end > day_index >= index_start:  # 处于需要的本周日期
					data_excle.append([int(i) if i else 0 for i in sheet.row_values(row)])
			except:
				pass
				# print(sheet.cell(row, pos[1]).value)
		data_excle = np.array(data_excle)[:, 1:len(title) + 1]
		sumer = np.sum(data_excle, axis=0)
		res = {}
		assert len(sumer) == len(title), ValueError(f'长度不一致{len(sumer)}--{len(title)}')
		for i in range(len(title)):
			res[title[i]] = sumer[i]
		return res
	
	def Landong(self, excel_, sheet_name, find_name='成都王府井二店'):
		"""

		:param find_name:
		:param excel_:
		:param sheet_name:
		:return: {brand:[this_year,last_year]}
		"""
		sheet = excel_.sheet_by_name(sheet_name)
		pos = self.find_data(sheet, find_name)[0]
		pos[0] += 1
		pos[1] += 1
		data = {}
		row = 0
		while True:
			row += 1
			values = sheet.cell(row + pos[0], pos[1]).value  # 获取品牌
			if values:
				brand = sheet.cell(row + pos[0], pos[1]).value
				brand = brand.replace('la_prairie', 'la prairie')
				try:
					this_year_temp = float(sheet.cell(row + pos[0], pos[1] + 1).value)
				except:
					this_year_temp = 0
				try:
					last_year_temp = float(sheet.cell(row + pos[0], pos[1] + 2).value)
				except:
					last_year_temp = 0
				data[brand] = [this_year_temp if this_year_temp else 0,
				               last_year_temp if last_year_temp else 0]
			elif row < 5:
				continue
			else:
				break
		return data
	
	def write_excel(self, this_year: dict, last_year: dict, land_mtd: dict, land_ytd: dict):
		"""

		:param land_ytd:
		:param this_year: 今年本周业绩
		:param last_year: 去年本周业绩
		:param land: landing数据
		:return:
		"""
		assert self.path[3].get(), showerror('错误', '请输入保存路径')
		if len(self.add_week_str) != len(self.add_week_key):
			showerror('错误', '请将去年没有的品牌补齐!')
		else:
			for k, v in zip(self.add_week_key, self.add_week_str):
				last_year[k] = v.get()
		for k, v in this_year.items():
			land_mtd[k][0] = float(this_year[k]) + float(land_mtd[k][0])  # 今年本周
			land_mtd[k][1] = float(last_year[k]) + float(land_mtd[k][1])  # 去年本周
			
			land_ytd[k][0] = float(this_year[k]) + float(land_ytd[k][0])  # 今年本周
			land_ytd[k][1] = float(last_year[k]) + float(land_ytd[k][1])  # 去年本周
		# ---------------------排序------------------
		land_mtd_rank = sorted(land_mtd.items(), key=lambda x: x[1][0], reverse=True)  # 按照今年
		land_ytd_rank = sorted(land_ytd.items(), key=lambda x: x[1][0], reverse=True)
		wook = xlwt.Workbook()
		sheet_mtd = wook.add_sheet('MTD')
		sheet_ytd = wook.add_sheet('YTD')
		
		rows = len(land_mtd_rank)
		style = xlwt.easyxf(num_format_str='0%')  # 百分数（显示整数）
		for row in range(rows):
			sheet_mtd.write(row, 0, str(land_mtd_rank[row][0]))
			sheet_mtd.write(row, 1, float(land_mtd_rank[row][1][0]))
			sheet_mtd.write(row, 2, float(land_mtd_rank[row][1][1]))
			person = f'B{row + 1}/C{row + 1}-1'
			sheet_mtd.write(row, 3, xlwt.Formula(person), style)
			ranks = f'RANK(C{row + 1},c1:c{rows + 1})'
			sheet_mtd.write(row, 4, xlwt.Formula(ranks))
			
			sheet_ytd.write(row, 0, str(land_ytd_rank[row][0]))
			sheet_ytd.write(row, 1, float(land_ytd_rank[row][1][0]))
			sheet_ytd.write(row, 2, float(land_ytd_rank[row][1][1]))
			person = f'B{row + 1}/C{row + 1}-1'
			sheet_ytd.write(row, 3, xlwt.Formula(person), style)
			ranks = f'RANK(C{row + 1},c1:c{rows + 1})'
			sheet_ytd.write(row, 4, xlwt.Formula(ranks))
		wook.save(self.path[3].get())
		showinfo('完成', '文件已经保存至%s' % (self.path[3].get()))
		self.close()
	
	def find_data(self, sheet, find_name):
		"""
		查询某名称的位置
		:param sheet:
		:param find_name:
		:return: [pos[0],pos[1]...]
		"""
		rows, cols = sheet.nrows, sheet.ncols
		pos = []
		for row in range(rows):
			for col in range(cols):
				sheet_value = str(sheet.cell(row, col).value)
				if find_name in sheet_value:
					pos.append([row, col])
					return pos
		else:
			showerror('错误！', '没找到%s' % find_name)
	
	def file_path(self, strs, index_excle, index_sheet_names):
		str = askopenfilename(filetypes=(("xls files", "*.xls"), ("xls files", "*.xlsx")))
		excels = xlrd.open_workbook(str)
		names = excels.sheet_names()
		self.excel[index_excle] = excels
		self.sheet_names[index_sheet_names] = names
		strs.set(str)
		self.set_fram()
	
	def save_path(self, strs):
		str = asksaveasfilename(filetypes=(("xls files", "*.xls"), ("xls files", "*.xlsx")),
		                        defaultextension='.xls')
		strs.set(str)
	
	def close(self):
		self.win.destroy()
		self.root.wm_deiconify()
	
	def main(self):
		self.set_fram()
		self.win.mainloop()