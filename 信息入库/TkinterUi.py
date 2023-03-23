# -*- coding: utf-8 -*-
# @Time    : 2023/2/6 18:46
# @Author  : HongFei Wang
import re
import tkinter as tk
from tkinter import messagebox

import pymysql.err

import ulitSql


class GUI(object):
	def __init__(self, width=1200, height=600):
		self.opens_Finde = False
		self.window = tk.Tk()
		self.window.title("账务POS")
		self.window.config(background="#CAE1FF")
		screen_width = self.window.winfo_screenwidth()
		screen_height = self.window.winfo_screenheight()
		x = int(screen_width / 2 - width / 2)
		y = int(screen_height / 2 - height / 2)
		size = '{}x{}+{}+{}'.format(width, height, x, y)
		self.window.geometry(size)
		self.width = width
		self.height = height
		self.addVar = [tk.StringVar() for _ in range(6)]  # 新增记录
		self.sql = ulitSql.MySQLs()
		
		self.findAllData()
	
	# ------------------------------------------------
	# 主界面函数
	def findAllData(self):
		# 可视化至主界面
		
		datas = self.sql.findData("select * from pending WHERE DOING=1")
		title = ["ID", "  时 间", "   名称", "  描述", "价格", "进度"]
		draw = []
		for data in datas:
			draw.append(list(data))
		self.drawExcel(title, draw, 0, 100)
	
	def drawExcel(self, title, datas, starx, stary, windon=None):
		self.drawconfig(title[0], starx, stary, width=3, window=windon)
		self.drawconfig(title[1], starx + 50, stary, width=8, window=windon)
		self.drawconfig(title[2], starx + 260, stary, width=10, window=windon)
		self.drawconfig(title[3], starx + 700, stary, width=8, window=windon)
		self.drawconfig(title[4], starx + 1000, stary, width=5, window=windon)
		self.drawconfig(title[5], starx + 1150, stary, width=5, window=windon)
		
		#### ((3, 2023, 2, 6, '是法拉火山', '昂贵代价ask发', 500, 0),)
		row_index = 24
		for index, data in enumerate(datas):  # 设置详情按钮 查看详细信息
			times = "-".join([str(i) for i in data[1:4]])
			self.drawconfig(data[0], 0, y_pos=index * row_index + stary + row_index, width=3, window=windon)
			self.drawconfig(times, 50, y_pos=index * row_index + stary + row_index, width=8, window=windon)
			self.drawconfig(data[4], 160, y_pos=index * row_index + stary + row_index, width=30, height=1 + len(data[4]) // 16, window=windon)
			self.drawconfig(data[5], 450, y_pos=index * row_index + stary + row_index, width=70, height=1 + len(data[5]) // 30, window=windon)
			self.drawconfig(data[6], 1000, y_pos=index * row_index + stary + row_index, width=5, window=windon)
			self.drawconfig("未完成" if data[7] else "完成", 1150,
			                y_pos=index * row_index + stary + row_index, width=6,
			                bg="#EE2C2C" if data[7] else "#66ff66", window=windon)
	
	def drawconfig(self, DATA, x_pos, y_pos, width=10, height=1, bg="#CAE1FF", font_size=10, window=None):
		# 时间
		cell_time = tk.Text(window if window else self.window, width=width, height=height, wrap="char", bg=bg, font=("", font_size), bd=3)
		cell_time.place(x=x_pos, y=y_pos)
		cell_time.insert("end", str(DATA))
	
	def Farms(self):
		"""
		用以绘制基础按钮
		:return:
		"""
		tk.Button(self.window, text="{:^10}".format("新增工程"), font=("微软雅黑", 12), bg="#ffff99", command=self.addDatas).place(x=0, y=50)
		tk.Button(self.window, text="{:^10}".format("结算入库"), font=("微软雅黑", 12), bg="#ffff99").place(x=300, y=50)
		tk.Button(self.window, text="{:^10}".format("删除工程"), font=("微软雅黑", 12), bg="#ffff99").place(x=600, y=50)
		tk.Button(self.window, text="{:^10}".format("查询工程"), font=("微软雅黑", 12), bg="#ffff99", command=self.findData).place(x=900, y=50)
		self.window.config(background="#CAE1FF")
	
	# 主界面函数
	# -------------------------------------------
	
	# -------------------------------------------
	# 添加 项目
	def addDatas(self):
		addwindow = tk.Frame(self.window, width=1200, height=600)
		addwindow.pack(side='bottom')
		# 添加组件信息
		tk.Label(addwindow, text="年", font=("微软雅黑", 10)).place(x=20, y=200)
		tk.Label(addwindow, text="月", font=("微软雅黑", 10)).place(x=60, y=200)
		tk.Label(addwindow, text="日", font=("微软雅黑", 10)).place(x=100, y=200)
		tk.Label(addwindow, text="名称", font=("微软雅黑", 10)).place(x=260, y=200)
		tk.Label(addwindow, text="描述", font=("微软雅黑", 10)).place(x=600, y=200)
		tk.Label(addwindow, text="价钱", font=("微软雅黑", 10)).place(x=1000, y=200)
		tk.Button(addwindow, text="工程记录入库", font=("微软雅黑", 15), fg="#EE2C2C", command=self.addDataClick).place(x=800, y=400)
		tk.Button(addwindow, text="返回主页面", font=("微软雅黑", 15), fg="#EE2C2C", command=lambda: self.getback(addwindow)).place(x=0, y=400)
		self.drawAddData(addwindow, 30)
	
	def getback(self, farm):
		farm.destroy()
		self.Farms()
	
	# ------------------------------------------------------
	##### 新增 工程栏
	def intOnly(self, s: str) -> bool:
		return re.match(r"[-+]?\d*\.?\d*", s).group() == s
	
	def strOnly(self, s: str) -> bool:
		return re.match(r"[-+]?\d*", s).group() == s
	
	def drawAddData(self, addwindow, yPos=10):
		# 年
		tk.Entry(addwindow, bd=5, textvariable=self.addVar[0], width=5,
		         validate="key", validatecommand=(self.window.register(self.intOnly), "%P")).place(x=0, y=yPos + 200)
		# 月
		tk.Entry(addwindow, bd=5, textvariable=self.addVar[1], width=2,
		         validate="key", validatecommand=(self.window.register(self.intOnly), "%P")).place(x=55, y=yPos + 200)
		# 日
		tk.Entry(addwindow, bd=5, textvariable=self.addVar[2], width=2,
		         validate="key", validatecommand=(self.window.register(self.intOnly), "%P")).place(x=95, y=yPos + 200)
		# 名称
		tk.Entry(addwindow, bd=5, textvariable=self.addVar[3], width=40).place(x=140, y=yPos + 200)
		# 描述
		tk.Entry(addwindow, bd=5, textvariable=self.addVar[4], width=70).place(x=450, y=yPos + 200)
		# 价格
		tk.Entry(addwindow, bd=5, textvariable=self.addVar[5], font=("微软雅黑", 12), width=5,
		         validate="key", validatecommand=(self.window.register(self.intOnly), "%P")).place(x=1000, y=yPos + 200)
	
	def addDataClick(self):
		data = [i.get() for i in self.addVar]
		for i in range(len(data)):
			try:
				data[i] = int(data[i])
			except:
				pass
		data.append(0)
		data = tuple(data)
		try:
			self.sql.addData(data)
			self.sql.closeConn()
			messagebox.showinfo("完成", "写入成功")
		except pymysql.err.DataError as e:
			messagebox.showerror("错误", "检查输入是否存在空值")
	
	##### 新增工程栏
	# -----------------------------------------------------------
	
	#
	# print(draw)
	# ------- 显示完成/未完成任务
	def findData(self):
		addwindow = tk.Frame(self.window, width=1200, height=800)
		addwindow.pack(side='bottom')
		addwindow.config(bg="#CAE1FF")
		tk.Button(addwindow, text="返回主页面", font=("微软雅黑", 12), fg="#EE2C2C", command=lambda: self.getback(addwindow)).place(x=500, y=20)
		# 查询收益
		tk.Button(addwindow, text="查询累计收益", font=("微软雅黑", 12), bg="#CAE1FF", command=lambda: self.getback(addwindow)).place(x=50, y=100)
		tk.Button(addwindow, text="查询月份收益", font=("微软雅黑", 12), bg="#CAE1FF", command=lambda: self.findeMinth(2, addwindow)).place(x=300,
		                                                                                                                                    y=100)
		tk.Button(addwindow, text="查询年份收益", font=("微软雅黑", 12), bg="#CAE1FF", command=lambda: self.findYear(2023, addwindow)).place(x=550,
		                                                                                                                                     y=100)
		tk.Button(addwindow, text="查询完成订单", font=("微软雅黑", 12), bg="#CAE1FF", command=lambda: self.getback(addwindow)).place(x=750, y=100)
		tk.Button(addwindow, text="查询未完成单", font=("微软雅黑", 12), bg="#CAE1FF", command=lambda: self.getback(addwindow)).place(x=1000, y=100)
	
	def clearnFind(self, farm):
		"""
		清除 页面信息
		:param farm:
		:return:
		"""
		farm.destroy()
		self.findData()
	
	def findYear(self, year, window):
		if self.opens_Finde is False:
			self.opens_Finde = True
		else:
			pass
		data = self.sql.findData("select * from pending WHERE YEAR={}".format(year))
		datas = [[i for i in j] for j in data]
		title = ["ID", "  时 间", "   名称", "  描述", "价格", "进度"]
		draw = []
		sums = 0
		for data in datas:
			draw.append(list(data))
			sums += float(data[6])
		self.drawExcel(title, draw, 0, 200, windon=window)
		messagebox.showinfo("统计完成", f"{year}共计赚取{sums}元")
	
	def findeMinth(self, month, window):
		data = self.sql.findData("select * from pending WHERE MONTH={}".format(month))
		datas = [[i for i in j] for j in data]
		title = ["ID", "  时 间", "   名称", "  描述", "价格", "进度"]
		draw = []
		sums = 0
		for data in datas:
			draw.append(list(data))
			sums += float(data[6])
		self.drawExcel(title, draw, 0, 200, windon=window)
		messagebox.showinfo("统计完成", f"{month}共计赚取{sums}元")
	
	# 订单查询系列
	# ----------------------------------------------------------
	def main(self):
		self.Farms()
		self.window.mainloop()


if __name__ == '__main__':
	ui = GUI()
	ui.main()