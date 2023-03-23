# -*- coding: utf-8 -*-
# @Time    : 2022/10/16 8:20
# @Author  : HongFei Wang
import calendar
import datetime
import os.path
import tkinter as tk
from tkinter import *
from tkinter.filedialog import askdirectory, askopenfilename
from tkinter.messagebox import *

import xlrd
import xlwt


class CompeteWookCreat:
	"""
	生成下个月的竞品表的函数
	"""
	
	def __init__(self, win):
		# 定义窗口
		self.win = tk.Toplevel(win)
		self.win.geometry('300x230+600+300')
		self.win.resizable(False, False)
		self.win.title('竞品生成')
		self.root = win
		self.today = datetime.datetime.today()
		self.year = StringVar()
		self.month = StringVar()
		self.year.set(self.today.year)
		self.month.set(self.today.month + 1 if self.today.month != 12 else 1)
		self.fave_path = StringVar()
		
		win.withdraw()
		self.win.protocol('WM_DELETE_WINDOW', self.close)
		
		self.fave_path.set(f'{self.today.year}年{self.today.month + 1 if self.today.month + 1 <= 12 else 1}月竞品.xls')
		self.config = StringVar()  # 配置文件
		self.config.set('config.xls')
	
	def creat_fram(self):
		Label(self.win, text='保存路径', font=('宋体', 13)).grid(row=0, column=0, pady=20)
		Entry(self.win, textvariable=self.fave_path, font=('宋体', 13), width=15).grid(row=0, column=1, pady=20)
		Button(self.win, text='选择文件', command=lambda: self.save_path(self.fave_path)).grid(row=0, column=2, pady=20)
		Label(self.win, text='标准文件', font=('宋体', 13)).grid(row=1, column=0, pady=20)
		Entry(self.win, textvariable=self.config, font=('宋体', 13), width=15).grid(row=1, column=1, pady=20)
		Button(self.win, text='选择文件', command=lambda: self.file_path(self.config)).grid(row=1, column=2, pady=20)
		Button(self.win, text='生成竞品', command=self.create_wook, font=('宋体', 15), fg='yellow', bg='gray').grid(row=3, column=1, pady=20)
		Entry(self.win, textvariable=self.year, width=5).grid(row=2, column=0)
		Entry(self.win, textvariable=self.month, width=5).grid(row=2, column=1)
	
	def create_wook(self):
		# ------------------------------------------------格式设置-----------------------------------
		# -----------------字体----------------------
		font = xlwt.Font()
		font.name = 'Calibri'  # 设置字体
		font.colour_index = 0  # 设置字体颜色
		font.height = 200  # 字体大小
		# ------------------------对其------------------
		alignment = xlwt.Alignment()
		alignment.horz = 2  # 设置水平位置，1是左对齐，2是居中，3是右对齐
		# 设置自动换行
		alignment.wrap = 1
		# --------------------边框-----------------
		# 设置边框
		borders = xlwt.Borders()  # Create Borders
		# DASHED虚线
		# NO_LINE没有
		# THIN实线
		borders.left = xlwt.Borders.THIN
		borders.right = xlwt.Borders.THIN
		borders.top = xlwt.Borders.THIN
		borders.bottom = xlwt.Borders.THIN
		# --------------------------------普通位置格式---------------------------
		style1 = xlwt.XFStyle()
		style1.font = font
		style1.alignment = alignment
		style1.borders = borders
		
		# -----------------------------周末格式------------------------------------
		style2 = xlwt.XFStyle()
		pattern = xlwt.Pattern()
		pattern.pattern = xlwt.Pattern.SOLID_PATTERN  # May be: NO_PATTERN, SOLID_PATTERN, or 0x00 through 0x12
		pattern.pattern_fore_colour = 5  # 给背景颜色赋值 # 黄色背景
		style2.pattern = pattern  # 把背景颜色加到表格样式里去
		style2.font = font
		style2.alignment = alignment
		style2.borders = borders
		# ----------------------------统计格式---------------------------------
		style3 = xlwt.XFStyle()
		pattern = xlwt.Pattern()
		pattern.pattern = xlwt.Pattern.SOLID_PATTERN  # May be: NO_PATTERN, SOLID_PATTERN, or 0x00 through 0x12
		pattern.pattern_fore_colour = 22  # 给背景颜色赋值 # 黄色背景
		style3.pattern = pattern  # 把背景颜色加到表格样式里去
		style3.font = font
		style3.alignment = alignment
		style3.borders = borders
		# -------------------------------------------------------------
		style4 = xlwt.XFStyle()
		pattern = xlwt.Pattern()
		pattern.pattern = xlwt.Pattern.SOLID_PATTERN  # May be: NO_PATTERN, SOLID_PATTERN, or 0x00 through 0x12
		pattern.pattern_fore_colour = 22  # 给背景颜色赋值 # 黄色背景
		style4.pattern = pattern  # 把背景颜色加到表格样式里去
		style4.font = font
		style4.alignment = alignment
		style4.borders = borders
		style4.num_format_str = '0.%'
		# -------------------------------------------------------------------
		year = int(self.year.get())
		month = int(self.month.get())
		week_day, days = calendar.monthrange(year, month)
		week_day_ = week_day
		title_excel = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
		               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG',
		               'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV',
		               'AW', 'AX', 'AY', 'AZ', 'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK',
		               'BL', 'BM', 'BN', 'BO', 'BP', 'BQ', 'BR', 'BS', 'BT', 'BU', 'BV', 'BW', 'BX', 'BY', 'BZ',
		               'CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG', 'CH', 'CI', 'CJ', 'CK', 'CL', 'CM', ]
		Wooks = xlwt.Workbook()
		sheet = Wooks.add_sheet('成都王府井二店')
		title = xlrd.open_workbook(self.config.get()).sheet_by_index(0).row_values(0)
		sheet.write_merge(0, 0, 0, len(title) - 1, '成都王府井二店', style3)
		cols = len(title)
		for col in range(cols):
			sheet.write(1, col, title[col], style1)
		index = 0  # 应该加多少行
		row_temp = 1  # 每周1的row标签
		
		# --------------------------开始写入新信息------------------
		
		Sunday = []
		for row in range(2, days + 2):
			sheet.write(row + index, 0, row - 1, style1)
			week_day += 1
			sheet.write(row + index, cols - 2,
			            xlwt.Formula(f'RANK(B{row + index + 1},$B${row + index + 1}:${title_excel[cols - 5]}${row + index + 1})'), style3)
			sheet.write(row + index, cols - 1, xlwt.Formula(f'B{row + index + 1}/sum(B{row + index + 1}:{title_excel[cols - 5]}{row + index + 1})'),
			            style4)
			sheet.write(row + index, cols - 3, xlwt.Formula(f'sum(B{row + index + 1}:{title_excel[cols - 5]}{row + index + 1})'), style3)
			for col in range(1, cols - 3):
				sheet.write(row + index, col, 0, style1)
			if week_day % 7 == 0:  # 判断是否周末
				index += 1
				Sunday.append(row + index + 1)  # 获取求和信息
				sheet.write(row + index, 0, f'第{index}周', style2)
				sheet.write(row + index, cols - 2,
				            xlwt.Formula(f'RANK(B{row + index + 1},$B${row + index + 1}:${title_excel[cols - 5]}${row + index + 1})'), style3)
				sheet.write(row + index, cols - 1,
				            xlwt.Formula(f'B{row + index + 1}/sum(B{row + index + 1}:{title_excel[cols - 5]}{row + index + 1})'), style4)
				for col in range(1, cols - 3):
					texts = f'SUM({title_excel[col]}{row_temp + 1}:{title_excel[col]}{row + index})'
					sheet.write(row + index, col, xlwt.Formula(texts), style2)
				texts = f'SUM({title_excel[cols - 4]}{row_temp + 1}:{title_excel[cols - 4]}{row + index})'
				sheet.write(row + index, cols - 3, xlwt.Formula(texts), style3)
				row_temp = row + index + 1
		# 判断最后一天是否是周末
		if row % 7 + 7 - week_day_ != 0:
			index += 1
			sheet.write(row + index, cols - 2,
			            xlwt.Formula(f'RANK(B{row + index + 1},$B${row + index + 1}:${title_excel[cols - 5]}${row + index + 1})'), style3)
			sheet.write(row + index, cols - 1, xlwt.Formula(f'B{row + index + 1}/sum(B{row + index + 1}:{title_excel[cols - 5]}{row + index + 1})'),
			            style4)
			sheet.write(row + index, 0, f'第{index}周', style2)
			Sunday.append(row + index + 1)
			for col in range(1, cols - 3):
				texts = f'SUM({title_excel[col]}{row_temp + 1}:{title_excel[col]}{row + index})'
				sheet.write(row + index, col, xlwt.Formula(texts), style2)
			texts = f'SUM({title_excel[cols - 4]}{row_temp + 1}:{title_excel[cols - 4]}{row + index})'
			sheet.write(row + index, cols - 3, xlwt.Formula(texts), style3)
		# 写入总求和
		sheet.write(row + index + 1, 0, 'TOTAL', style3)
		sheet.write(row + index + 1, cols - 2,
		            xlwt.Formula(f'RANK(B{row + index + 1},$B${row + index + 1}:${title_excel[cols - 5]}${row + index + 1})'), style3)
		sheet.write(row + index + 1, cols - 1, xlwt.Formula(f'B{row + index + 1}/sum(B{row + index + 2}:{title_excel[cols - 5]}{row + index + 1})'),
		            style4)
		for col in range(cols - 3):
			texts = '+'.join([title_excel[col] + str(i) for i in Sunday])
			sheet.write(row + index + 1, col + 1, xlwt.Formula(texts), style3)  # 报错
		# --------------------------写入排序----------------------
		sheet.write(row + index + 2, 0, '排名', style3)
		sheet.write(row + index + 2, cols - 2,
		            xlwt.Formula(f'RANK(B{row + index + 2},$B${row + index + 2}:${title_excel[cols - 5]}${row + index + 1})'), style3)
		sheet.write(row + index + 2, cols - 1, xlwt.Formula(f'B{row + index + 2}/sum(B{row + index + 2}:{title_excel[cols - 5]}{row + index + 2})'),
		            style4)
		"""=RANK(B39,$B$39:$AC$39)"""
		for col in range(cols - 3):
			texts = f'RANK(${title_excel[col]}{row + index + 2},${title_excel[0]}${row + index + 2}:${title_excel[cols - 5]}${row + index + 2})'
			sheet.write(row + index + 2, col + 1, xlwt.Formula(texts), style3)
		try:
			Wooks.save(os.path.join(self.fave_path.get(), self.year.get() + '年_' + self.month.get() + '月竞品表.xls'))
			showinfo('完成', f"保存到{os.path.join(self.fave_path.get(), self.year.get() + '年_' + self.month.get() + '月竞品表.xls')}")
			self.close()
		except:
			showerror('错误', '保存失败，请查看文件是否已经打开！！')
	
	def file_path(self, strs):
		str = askopenfilename(filetypes=(("xls files", "*.xls"), ("xls files", "*.xlsx")))
		strs.set(str)
	
	def save_path(self, strs):
		str = askdirectory()
		strs.set(str)
	
	def close(self):
		self.win.destroy()
		self.root.wm_deiconify()
	
	def main(self):
		self.creat_fram()
		self.win.mainloop()