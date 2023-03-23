# -*- coding: utf-8 -*-
# @Time    : 2022/12/27 10:46
# @Author  : HongFei Wang
import xlrd


class ExcelS(object):
	def __init__(self, wins):
		self.excel = None  # excel文件
		self.excelsheet = None  # 对应excel得某个表格
		self.excelSheets = []  # 所有表格
		self.win = wins  # 获取窗口
	
	def fileExcel(self, filename):
		"""

		:param filename: excel 名字
		"""
		# 打开excels
		self.excel = xlrd.open_workbook(filename)
		self.excelSheets = self.excel.sheets()
		print(self.excelSheets)
	
	def getExcelData(self, sheet_name):
		"""
		:param sheet_name: 表格名称
		"""
		self.excelsheet = self.excel.sheet_by_name(sheet_name)